#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import argparse
from pathlib import Path
import wandb

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import from existing scripts
import src.analyze
import src.globals
import src.plot

def load_similarity_data(metric, layer, num_images=50):
    """
    Load the similarity matrix from cached representations.
    Returns a dictionary with similarity scores between all model pairs.
    """
    # Define model combinations like in compare-vlv-representations.py
    # Map from the simple keys used in the similarity script to the full prism model names
    PRISM_MODEL_MAPPING = {
        "llama3+clip": "prism-llama3-instruct+8b+clip",
        "llama3+dinosiglip": "prism-llama3-instruct+8b+dinosiglip", 
        "llama3+siglip": "prism-llama3-instruct+8b+siglip",
        "llama2+clip": "prism-llama2-chat+7b+clip",
        "llama2+dinosiglip": "prism-llama2-chat+7b+dinosiglip",
        "llama2+siglip": "prism-llama2-chat+7b+siglip",
        "mistralv1+clip": "prism-mistral-instruct-v0.1+7b+clip",  # Note: this might not exist
        "mistralv1+dinosiglip": "prism-mistral-instruct-v0.1+7b+dinosiglip",  # Note: this might not exist
        "mistralv1+siglip": "prism-mistral-instruct-v0.1+7b+siglip",  # Note: this might not exist
        "mistralv2+clip": "prism-mistral-instruct-v0.2+7b+clip",
        "mistralv2+dinosiglip": "prism-mistral-instruct-v0.2+7b+dinosiglip",
        "mistralv2+siglip": "prism-mistral-instruct-v0.2+7b+siglip",
    }
    
    def get_cache_path(model_key, location, num_images):
        cache_dir = Path("results/cached_representations")
        return cache_dir / f"{model_key}_{location}_{num_images}.pkl"
    
    def load_cached_representations(model_key, location, num_images):
        cache_path = get_cache_path(model_key, location, num_images)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    # Load all cached representations
    all_representations = {}
    for simple_key, prism_key in PRISM_MODEL_MAPPING.items():
        cached_reps = load_cached_representations(simple_key, layer, num_images)
        if cached_reps is not None:
            all_representations[simple_key] = cached_reps
        else:
            print(f"Warning: No cached representations found for {simple_key}")
    
    if not all_representations:
        raise ValueError(f"No cached representations found for layer {layer}. Please run compare-vlm-representations.py first.")
    
    # Compute similarity matrix
    if metric == 'cosine':
        similarity_data = compute_cosine_similarity_matrix(all_representations)
    elif metric == 'cka':
        similarity_data = compute_centered_kernel_alignment_matrix(all_representations)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return similarity_data, list(all_representations.keys())

def compute_cosine_similarity_matrix(models_dict):
    """Compute cosine similarity matrix (from compare-vlm-representations.py)"""
    model_keys = list(models_dict.keys())
    n = len(model_keys)
    similarity_matrix = np.zeros((n, n))
    
    for i, (model_key_i, reps_i) in enumerate(models_dict.items()):
        for j, (model_key_j, reps_j) in enumerate(models_dict.items()):
            if reps_i.shape[1] != reps_j.shape[1]:
                similarity_matrix[i, j] = np.nan
                continue
            
            try:
                import torch
                # Normalize the representations
                reps_i_norm = reps_i / reps_i.norm(dim=1, keepdim=True)
                reps_j_norm = reps_j / reps_j.norm(dim=1, keepdim=True)
                
                # Compute cosine similarity for each pair
                similarities = torch.matmul(reps_i_norm, reps_j_norm.t())
                
                # Take the average over the diagonal (same input similarity)
                avg_similarity = torch.mean(torch.diag(similarities)).item()
                similarity_matrix[i, j] = avg_similarity
            except Exception as e:
                print(f"Error computing similarity between {model_key_i} and {model_key_j}: {e}")
                similarity_matrix[i, j] = np.nan
    
    return {
        'similarity_matrix': similarity_matrix,
        'model_keys': model_keys
    }

def compute_centered_kernel_alignment_matrix(models_dict):
    """Compute CKA matrix (from compare-vlm-representations.py)"""
    model_keys = list(models_dict.keys())
    n = len(model_keys)
    cka_matrix = np.zeros((n, n))
    
    def cka(X, Y):
        """Compute Centered Kernel Alignment between X and Y."""
        import torch
        # Center the data
        X_centered = X - X.mean(dim=0, keepdim=True)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute Gram matrices
        K = torch.matmul(X_centered, X_centered.t())
        L = torch.matmul(Y_centered, Y_centered.t())
        
        # Compute CKA
        numerator = torch.trace(torch.matmul(K, L))
        denominator = torch.sqrt(torch.trace(torch.matmul(K, K)) * torch.trace(torch.matmul(L, L)))
        
        return (numerator / denominator).item()
    
    for i, (model_key_i, reps_i) in enumerate(models_dict.items()):
        for j, (model_key_j, reps_j) in enumerate(models_dict.items()):
            if reps_i.shape[1] != reps_j.shape[1]:
                cka_matrix[i, j] = np.nan
                continue
            try:
                cka_value = cka(reps_i, reps_j)
                cka_matrix[i, j] = cka_value
            except Exception as e:
                print(f"Error computing CKA between {model_key_i} and {model_key_j}: {e}")
                cka_matrix[i, j] = np.nan
    
    return {
        'similarity_matrix': cka_matrix,
        'model_keys': model_keys
    }

def load_transfer_data():
    """
    Load transfer attack data (adapted from 02_transfer_attack_prismatic_n=1.py)
    """
    # Setup directories
    data_dir, results_dir = src.analyze.setup_notebook_dir(
        notebook_dir=os.path.dirname(os.path.abspath(__file__)),
        refresh=False,
    )
    
    # Use the same sweep IDs as in the original script
    sweep_ids = ["4cdnwkvi", "l4z6vrlq", "i3ng4vgj"]
    wandb_username = "ishagupta2000"
    
    # Download evaluation runs
    eval_runs_configs_df = src.analyze.download_wandb_project_runs_configs(
        wandb_project_path="universal-vlm-jailbreak-eval",
        data_dir=data_dir,
        sweep_ids=sweep_ids,
        refresh=False,
        finished_only=True,
        wandb_username=wandb_username,
        filetype="csv",
    )
    
    # Extract relevant columns
    eval_runs_configs_df = src.analyze.extract_key_value_from_df_col(
        df=eval_runs_configs_df,
        col_name="data",
        key_in_dict="dataset",
        new_col_name="eval_dataset",
    )
    
    eval_runs_configs_df.rename(
        columns={"run_id": "eval_run_id", "wandb_attack_run_id": "attack_run_id"},
        inplace=True,
    )
    
    # Map model names to nice strings
    eval_runs_configs_df["model_to_eval"] = eval_runs_configs_df["model_to_eval"].apply(
        src.analyze.map_string_set_of_models_to_nice_string
    )
    eval_runs_configs_df["models_to_attack"] = eval_runs_configs_df[
        "models_to_attack"
    ].apply(src.analyze.map_string_set_of_models_to_nice_string)
    
    # Download attack runs
    unique_attack_run_ids = eval_runs_configs_df["attack_run_id"].unique()
    attack_runs_configs_df = src.analyze.download_wandb_project_runs_configs_by_run_ids(
        wandb_project_path="universal-vlm-jailbreak",
        wandb_username=wandb_username,
        data_dir=data_dir,
        run_ids=unique_attack_run_ids,
        refresh=False,
        finished_only=True,
        filetype="csv",
    )
    
    attack_runs_configs_df.rename(columns={"run_id": "attack_run_id"}, inplace=True)
    
    # Merge attack data with eval data
    eval_runs_configs_df = eval_runs_configs_df.merge(
        right=attack_runs_configs_df[["attack_run_id"]],
        how="left",
        on="attack_run_id",
    )
    
    # Mark which runs are "attacked" (same model for attack and eval)
    eval_runs_configs_df["Attacked"] = eval_runs_configs_df.apply(
        lambda row: row["model_to_eval"] in row["models_to_attack"], axis=1
    )
    
    # Load run histories (actual scores)
    eval_runs_histories_df = src.analyze.download_wandb_project_runs_histories(
        wandb_project_path="universal-vlm-jailbreak-eval",
        wandb_username=wandb_username,
        data_dir=data_dir,
        sweep_ids=sweep_ids,
        refresh=False,
        wandb_run_history_samples=1000000,
        filetype="csv",
    )
    
    eval_runs_histories_df.rename(columns={"run_id": "eval_run_id"}, inplace=True)
    eval_runs_histories_df.drop(columns=["models_to_attack"], inplace=True, errors='ignore')
    
    # Merge with configs
    eval_runs_histories_df = eval_runs_histories_df.merge(
        right=eval_runs_configs_df[
            [
                "eval_run_id",
                "attack_run_id",
                "model_to_eval",
                "models_to_attack",
                "Attacked",
            ]
        ],
        how="inner",
        on="eval_run_id",
    )
    
    # Get final scores (last optimizer step for each run)
    final_scores_df = (
        eval_runs_histories_df.groupby("eval_run_id")
        .apply(lambda x: x.loc[x["optimizer_step_counter_epoch"].idxmax()])
        .reset_index(drop=True)
    )
    
    # Focus on Claude 3 Opus scores
    score_col = "loss/score_model=claude3opus"
    if score_col not in final_scores_df.columns:
        raise ValueError(f"Score column {score_col} not found in data")
    
    final_scores_df = final_scores_df[
        ["eval_run_id", "attack_run_id", "model_to_eval", "models_to_attack", 
         "Attacked", score_col]
    ].rename(columns={score_col: "final_score"})
    
    return final_scores_df

def map_model_names_to_keys(model_name):
    """
    Map the nice model names from transfer data to model keys used in similarity data.
    Based on the actual model names we observed in the debug output.
    """
    # Mapping from nice model names (used in transfer data) to simple keys (used in similarity data)
    name_to_key_mapping = {
        'Llama2 Chat 7B + CLIP': 'llama2+clip',
        'Llama2 Chat 7B + DINOv2/SigLIP': 'llama2+dinosiglip', 
        'Llama2 Chat 7B + SigLIP': 'llama2+siglip',
        'Llama3 Instr 8B + CLIP': 'llama3+clip',
        'Llama3 Instr 8B + DINOv2/SigLIP': 'llama3+dinosiglip',
        'Llama3 Instr 8B + SigLIP': 'llama3+siglip',
        'Mistral Instr v0.2 7B + CLIP': 'mistralv2+clip',
        'Mistral Instr v0.2 7B + DINOv2/SigLIP': 'mistralv2+dinosiglip',
        'Mistral Instr v0.2 7B + SigLIP': 'mistralv2+siglip',
    }
    
    return name_to_key_mapping.get(model_name, None)

def create_scatter_plot(attacked_model, similarity_data, transfer_data, metric, layer):
    """
    Create scatter plot for a specific attacked model.
    """
    # Get the model key for the attacked model
    attacked_model_key = map_model_names_to_keys(attacked_model)
    if attacked_model_key is None:
        print(f"Warning: Could not map attacked model '{attacked_model}' to model key")
        return
    
    # Find the index of attacked model in similarity matrix
    model_keys = similarity_data['model_keys']
    if attacked_model_key not in model_keys:
        print(f"Warning: Attacked model key '{attacked_model_key}' not found in similarity data")
        return
    
    attacked_idx = model_keys.index(attacked_model_key)
    
    # Filter transfer data for this attacked model
    model_transfer_data = transfer_data[
        transfer_data["models_to_attack"] == attacked_model
    ].copy()
    
    if len(model_transfer_data) == 0:
        print(f"Warning: No transfer data found for attacked model '{attacked_model}'")
        return
    
    # Average transfer scores across runs for each evaluated model
    avg_transfer_scores = model_transfer_data.groupby("model_to_eval")["final_score"].mean().reset_index()
    
    # Prepare data for plotting
    plot_data = []
    
    for _, row in avg_transfer_scores.iterrows():
        eval_model = row["model_to_eval"]
        eval_model_key = map_model_names_to_keys(eval_model)
        
        if eval_model_key is None or eval_model_key not in model_keys:
            print(f"Warning: Could not find similarity data for eval model '{eval_model}'")
            continue
        
        eval_idx = model_keys.index(eval_model_key)
        similarity_score = similarity_data['similarity_matrix'][attacked_idx, eval_idx]
        
        if not np.isnan(similarity_score):
            plot_data.append({
                'model': eval_model,
                'similarity': similarity_score,
                'transfer_success': row["final_score"],
                'is_self': eval_model == attacked_model
            })
    
    if not plot_data:
        print(f"Warning: No valid data points for attacked model '{attacked_model}'")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot non-self points
    non_self = plot_df[~plot_df['is_self']]
    if len(non_self) > 0:
        plt.scatter(non_self['similarity'], non_self['transfer_success'], 
                   alpha=0.7, s=100, label='Other models')
    
    # Plot self point (attacked model evaluated on itself)
    self_point = plot_df[plot_df['is_self']]
    if len(self_point) > 0:
        plt.scatter(self_point['similarity'], self_point['transfer_success'], 
                   alpha=0.9, s=150, color='red', marker='*', label='Self (attacked model)')
    
    # Add model labels
    for _, row in plot_df.iterrows():
        plt.annotate(row['model'], (row['similarity'], row['transfer_success']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    plt.xlabel(f'{metric.upper()} Similarity to {attacked_model}')
    plt.ylabel('Transfer Success (Final Score)')
    plt.title(f'Similarity vs Transfer Success\nAttacked Model: {attacked_model}\nLayer: {layer}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs("results/similarity_transfer_plots", exist_ok=True)
    safe_model_name = attacked_model.replace(' ', '_').replace('/', '_').replace('+', '_')
    filename = f"results/similarity_transfer_plots/{safe_model_name}_{metric}_{layer}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot for {attacked_model} to {filename}")
    print(f"Data points: {len(plot_df)}")
    if len(plot_df) > 1:
        corr = plot_df['similarity'].corr(plot_df['transfer_success'])
        print(f"Correlation: {corr:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Combine similarity metrics with attack transfer results.')
    parser.add_argument('--attacked_model', type=str, required=True,
                        help='Name of the attacked model (as it appears in transfer data)')
    parser.add_argument('--metric', type=str, choices=['cosine', 'cka'], default='cosine',
                        help='Similarity metric to use')
    parser.add_argument('--layer', type=str, 
                        choices=['vision_encoder', 'post_projector', 'llm_final_layer'],
                        default='llm_final_layer',
                        help='Layer to extract representations from')
    parser.add_argument('--num_images', type=int, default=50,
                        help='Number of images used for similarity computation')
    
    args = parser.parse_args()
    
    print(f"Loading similarity data for metric={args.metric}, layer={args.layer}...")
    try:
        similarity_data, model_keys = load_similarity_data(args.metric, args.layer, args.num_images)
        print(f"Loaded similarity data for {len(model_keys)} models")
    except Exception as e:
        print(f"Error loading similarity data: {e}")
        print("Please run compare-vlm-representations.py first to generate cached representations.")
        return
    
    print("Loading transfer attack data...")
    try:
        transfer_data = load_transfer_data()
        print(f"Loaded transfer data with {len(transfer_data)} entries")
        print(f"Available attacked models: {sorted(transfer_data['models_to_attack'].unique())}")
    except Exception as e:
        print(f"Error loading transfer data: {e}")
        return
    
    print(f"Creating scatter plot for attacked model: {args.attacked_model}")
    create_scatter_plot(args.attacked_model, similarity_data, transfer_data, args.metric, args.layer)

if __name__ == "__main__":
    main() 