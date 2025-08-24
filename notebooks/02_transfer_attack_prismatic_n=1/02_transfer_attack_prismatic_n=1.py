import ast
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import wandb

import src.analyze
import src.globals
import src.plot


refresh = True
# refresh = False
finished_only = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)


# sweep_ids = [
#     "zyf0lb9y",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 1)
#     "s754hflc",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 2)
#     "jl9as45o",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 3)
#     "1yoxmmrk",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 4)
#     "bjg1o5ko",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 5)
#     "8nrhoa2q",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 6)
# ]

sweep_ids = ["4cdnwkvi", "l4z6vrlq", "i3ng4vgj"]


wandb_username = "ishagupta2000"
eval_runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    finished_only=finished_only,
    wandb_username=wandb_username,
    filetype="csv",
)
eval_runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=eval_runs_configs_df,
    col_name="data",
    key_in_dict="dataset",
    new_col_name="eval_dataset",
)
eval_runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=eval_runs_configs_df,
    col_name="data",
    key_in_dict="split",
    new_col_name="eval_dataset_split",
)

eval_runs_configs_df.rename(
    columns={"run_id": "eval_run_id", "wandb_attack_run_id": "attack_run_id"},
    inplace=True,
)

# Switch attack_model_names and eval_model_name to nice strings.
eval_runs_configs_df["model_to_eval"] = eval_runs_configs_df["model_to_eval"].apply(
    src.analyze.map_string_set_of_models_to_nice_string
)
eval_runs_configs_df["models_to_attack"] = eval_runs_configs_df[
    "models_to_attack"
].apply(src.analyze.map_string_set_of_models_to_nice_string)

# Download attack runs.
unique_attack_run_ids = eval_runs_configs_df["attack_run_id"].unique()
print("Attack Run IDs: ", unique_attack_run_ids.tolist())
print("\nDebug: Number of evaluation runs before filtering:", len(eval_runs_configs_df))
print("Debug: Unique models to attack:", eval_runs_configs_df["models_to_attack"].unique())
print("Debug: Unique models to eval:", eval_runs_configs_df["model_to_eval"].unique())

attack_runs_configs_df = src.analyze.download_wandb_project_runs_configs_by_run_ids(
    wandb_project_path="universal-vlm-jailbreak",
    wandb_username=wandb_username,
    data_dir=data_dir,
    run_ids=unique_attack_run_ids,
    refresh=refresh,
    finished_only=finished_only,
    filetype="csv",
)

print("\nDebug: Number of attack runs:", len(attack_runs_configs_df))

attack_runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=attack_runs_configs_df,
    col_name="data",
    key_in_dict="dataset",
    new_col_name="attack_dataset",
)
attack_runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=attack_runs_configs_df,
    col_name="image_kwargs",
    key_in_dict="image_initialization",
    new_col_name="image_initialization",
)
attack_runs_configs_df.rename(
    columns={"run_id": "attack_run_id"},
    inplace=True,
)
attack_runs_configs_df["image_initialization"] = attack_runs_configs_df[
    "image_initialization"
].map(src.globals.IMAGE_INITIALIZATION_TO_STRINGS_DICT)

# Join attack run data into to evals df.
eval_runs_configs_df = eval_runs_configs_df.merge(
    right=attack_runs_configs_df[
        ["attack_run_id", "attack_dataset", "image_initialization"]
    ],
    how="left",
    left_on="attack_run_id",
    right_on="attack_run_id",
)

print("\nDebug: Number of evaluation runs after merge:", len(eval_runs_configs_df))

eval_runs_configs_df["Attacked"] = eval_runs_configs_df.apply(
    lambda row: row["model_to_eval"] in row["models_to_attack"], axis=1
)

print("\nDebug: Number of runs marked as Attacked:", eval_runs_configs_df["Attacked"].sum())

# Load the heftier runs' histories dataframe.
eval_runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    wandb_username=wandb_username,
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_run_history_samples=1000000,
    filetype="csv",
)

print("\nDebug: Number of history entries:", len(eval_runs_histories_df))
print("Debug: Unique run IDs in history:", len(eval_runs_histories_df["run_id"].unique()))

# Check steps for different metrics
print("\nDebug: Steps per run for different metrics:")
for col in eval_runs_histories_df.columns:
    if 'loss' in col or 'score' in col:
        print(f"\nMetric: {col}")
        print(eval_runs_histories_df.groupby("run_id")[col].agg(["min", "max", "count"]).head())

# Rename run_id to eval_run_id first
eval_runs_histories_df.rename(columns={"run_id": "eval_run_id"}, inplace=True)

print("\nDebug: After Rename")
print("Sample of optimizer steps per run:")
print(eval_runs_histories_df.groupby("eval_run_id")["optimizer_step_counter_epoch"].agg(["min", "max", "count"]).head())
print("\nSample of scores per run:")
print(eval_runs_histories_df.groupby("eval_run_id")["loss/score_model=claude3opus"].agg(["min", "max", "count"]).head())

# This col is not populated on this df.
eval_runs_histories_df.drop(columns=["models_to_attack"], inplace=True)

eval_runs_histories_df = eval_runs_histories_df.merge(
    right=eval_runs_configs_df[
        [
            "eval_run_id",
            "attack_run_id",
            "model_to_eval",
            "models_to_attack",
            "attack_dataset",
            "eval_dataset",
            "image_initialization",
            "Attacked",
        ]
    ],
    how="inner",
    on="eval_run_id",
)

print("\nDebug: Number of history entries after merge:", len(eval_runs_histories_df))
print("Debug: Unique run IDs after merge:", len(eval_runs_histories_df["eval_run_id"].unique()))

unique_metrics_order = [
    "loss/score_model=claude3opus",
]

eval_runs_histories_tall_df = eval_runs_histories_df.melt(
    id_vars=[
        "eval_run_id",
        "attack_run_id",
        "attack_dataset",
        "eval_dataset",
        "model_to_eval",
        "models_to_attack",
        "optimizer_step_counter_epoch",
        "image_initialization",
        "Attacked",
    ],
    value_vars=unique_metrics_order,
    var_name="Metric",
    value_name="Score",
)

print("\nDebug: Number of entries in tall dataframe:", len(eval_runs_histories_tall_df))
print("Debug: Unique run IDs in tall dataframe:", len(eval_runs_histories_tall_df["eval_run_id"].unique()))

eval_runs_histories_tall_df.rename(
    columns={
        "model_to_eval": "Eval VLM",
        "image_initialization": "Image Initialization",
    },
    inplace=True,
)

sorted_unique_attacked_models = list(
    sorted(eval_runs_histories_tall_df["models_to_attack"].unique())
)

# Convert metrics to nice strings.
eval_runs_histories_tall_df["Original Metric"] = eval_runs_histories_tall_df["Metric"]
eval_runs_histories_tall_df["Metric"] = eval_runs_histories_tall_df["Metric"].map(
    lambda k: src.globals.METRICS_TO_TITLE_STRINGS_DICT.get(k, k)
)

# Obtain the first optimizer_step_counter_epoch per eval_run_id.
first_optimizer_step = (
    eval_runs_histories_tall_df.groupby("eval_run_id")["optimizer_step_counter_epoch"]
    .min()
    .reset_index()
)
last_optimizer_step = (
    eval_runs_histories_tall_df.groupby("eval_run_id")["optimizer_step_counter_epoch"]
    .max()
    .reset_index()
)

print("\nDebug: Number of first steps:", len(first_optimizer_step))
print("Debug: Number of last steps:", len(last_optimizer_step))

# Merge these with the original dataframe to get the corresponding rows
first_optimizer_step_rows_df = (
    pd.merge(
        eval_runs_histories_tall_df,
        first_optimizer_step,
        on=["eval_run_id", "optimizer_step_counter_epoch"],
        how="inner",
    )
    .rename(columns={"Score": "Initial Score"})
    .drop(columns=["optimizer_step_counter_epoch"])
    .drop_duplicates(subset=["eval_run_id", "Initial Score"])  # Remove duplicates
)

last_optimizer_step_rows_df = (
    pd.merge(
        eval_runs_histories_tall_df,
        last_optimizer_step,
        on=["eval_run_id", "optimizer_step_counter_epoch"],
        how="inner",
    )
    .rename(columns={"Score": "Final Score"})
    .drop(columns=["optimizer_step_counter_epoch"])
    .drop_duplicates(subset=["eval_run_id", "Final Score"])  # Remove duplicates
)

print("\nDebug: Number of first step rows:", len(first_optimizer_step_rows_df))
print("Debug: Number of last step rows:", len(last_optimizer_step_rows_df))

# Combine first and last rows into a single dataframe
first_and_last_optimizer_step_df = pd.merge(
    first_optimizer_step_rows_df,
    last_optimizer_step_rows_df,
    on=[
        "eval_run_id",
        "attack_run_id",
        "attack_dataset",
        "eval_dataset",
        "Eval VLM",
        "models_to_attack",
        "Image Initialization",
        "Attacked",
        "Metric",
        "Original Metric",
    ],
    how="inner",
)

print("\nDebug: Final number of points for plotting:", len(first_and_last_optimizer_step_df))
print("Debug: Unique run IDs in final data:", len(first_and_last_optimizer_step_df["eval_run_id"].unique()))
print("Debug: Distribution of points by models_to_attack:")
print(first_and_last_optimizer_step_df["models_to_attack"].value_counts())
print("\nDebug: Distribution of points by Eval VLM:")
print(first_and_last_optimizer_step_df["Eval VLM"].value_counts())

# Drop any rows where Initial Score or Final Score is NaN
first_and_last_optimizer_step_df = first_and_last_optimizer_step_df.dropna(subset=["Initial Score", "Final Score"])

print("\nDebug: Number of points after dropping NaN:", len(first_and_last_optimizer_step_df))

# Add detailed debug prints for the plotting data
print("\nDebug: Detailed data for plotting:")
for models_to_attack in sorted_unique_attacked_models:
    print(f"\nData for models_to_attack={models_to_attack}:")
    subset = first_and_last_optimizer_step_df[first_and_last_optimizer_step_df["models_to_attack"] == models_to_attack]
    print(subset[["Eval VLM", "Initial Score", "Final Score", "Attacked"]].to_string())

# --- BEGIN: VLM color mapping by LM family ---
def get_lm_family(vlm_name):
    name = vlm_name.lower()
    if "llama3" in name:
        return "llama3"
    elif "llama2" in name:
        return "llama2"
    elif "mistral" in name:
        return "mistral"
    else:
        return "other"

unique_vlms = first_and_last_optimizer_step_df["Eval VLM"].unique()
vlm_families = {vlm: get_lm_family(vlm) for vlm in unique_vlms}

from collections import Counter
family_counts = Counter(vlm_families.values())

# Use more distinct palettes
family_palettes = {
    "llama3": sns.color_palette("Greens", n_colors=family_counts["llama3"] or 1),
    "llama2": sns.color_palette("Reds", n_colors=family_counts["llama2"] or 1),
    "mistral": sns.color_palette("Blues", n_colors=family_counts["mistral"] or 1),
    "other": sns.color_palette("Greys", n_colors=family_counts["other"] or 1),
}

vlm_color_map = {}
for family, palette in family_palettes.items():
    family_vlms = [vlm for vlm, fam in vlm_families.items() if fam == family]
    for vlm, color in zip(family_vlms, palette):
        vlm_color_map[vlm] = color
# --- END: VLM color mapping by LM family ---

plt.close()
g = sns.relplot(
    data=first_and_last_optimizer_step_df,
    kind="scatter",
    x="Initial Score",
    y="Final Score",
    col="models_to_attack",
    col_order=sorted_unique_attacked_models,
    style="Attacked",
    style_order=[False, True],
    size="Attacked",
    size_order=[False, True],
    sizes=[200, 300],
    hue="Eval VLM",
    palette=vlm_color_map,
    col_wrap=5,
    s=250,
    aspect=0.75,
)
line = np.linspace(0.0, 1.0, 100)
for ax in g.axes.flat:
    ax.plot(line, line, "k--")
g.set_axis_labels("Harmful-Yet-Helpful (Initial)", "Harmful-Yet-Helpful (Final)")
g.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
# Add identity line to each axis.
g.set_titles(col_template="{col_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
g.fig.suptitle(
    "Claude 3 Opus Scores of Transfer From Single VLM to New VLM", y=1.0, fontsize=65
)
# Make space for the title.
plt.subplots_adjust(top=0.8)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title=f"final_score_vs_initial_score_by_attacked_split_models_to_attack",
)
# plt.show()


learning_curves_results_dir = os.path.join(results_dir, "learning_curves")
os.makedirs(learning_curves_results_dir, exist_ok=True)


for eval_dataset in eval_runs_histories_tall_df["eval_dataset"].unique():
    learning_curves_eval_dataset_results_dir = os.path.join(
        learning_curves_results_dir, f"eval_dataset={eval_dataset}"
    )
    os.makedirs(learning_curves_eval_dataset_results_dir, exist_ok=True)
    for attack_dataset in eval_runs_histories_tall_df["attack_dataset"].unique():
        learning_curves_eval_dataset_attack_dataset_results_dir = os.path.join(
            learning_curves_eval_dataset_results_dir,
            f"attack_dataset={attack_dataset}",
        )
        os.makedirs(
            learning_curves_eval_dataset_attack_dataset_results_dir, exist_ok=True
        )
        eval_runs_histories_tall_subset_df = eval_runs_histories_tall_df[
            (eval_runs_histories_tall_df["attack_dataset"] == attack_dataset)
            & (eval_runs_histories_tall_df["eval_dataset"] == eval_dataset)
        ]

        if len(eval_runs_histories_tall_subset_df) == 0:
            print(
                f"No data for attack_dataset={attack_dataset} and eval_dataset={eval_dataset}."
            )
            continue

        plt.close()
        g = sns.relplot(
            data=eval_runs_histories_tall_subset_df,
            kind="line",
            x="optimizer_step_counter_epoch",
            y="Score",
            col="models_to_attack",
            col_order=sorted_unique_attacked_models,
            style="Attacked",
            style_order=[False, True],
            hue="Eval VLM",
            col_wrap=5,
            linewidth=3,
            aspect=0.75,
        )
        g.set_axis_labels("Gradient Step", "Harmful-Yet-Helpful")
        g.set(xlim=(0, 50000), ylim=(0.0, 1.0))
        g.set_titles(col_template="{col_name}")
        sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
        g.fig.suptitle(
            "Claude 3 Opus Scores of Transfer From Single VLM to New VLM",
            y=1.0,
            fontsize=60,
        )
        # Make space for the title.
        plt.subplots_adjust(top=0.9)
        src.plot.save_plot_with_multiple_extensions(
            plot_dir=learning_curves_eval_dataset_attack_dataset_results_dir,
            plot_title=f"score_vs_optimizer_step_by_attacked_split_models_to_attack",
        )
        # plt.show()


print("Finished notebooks/02_transfer_attack_prismatic_n=1!")
