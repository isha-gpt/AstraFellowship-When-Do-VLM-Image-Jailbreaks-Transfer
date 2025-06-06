from enum import Enum, auto
import numpy as np
import lightning
import torch
import torch.optim
import torchvision.transforms
from typing import Any, Dict, List, Optional, Tuple, Union
import wandb

from src.models.ensemble import VLMEnsemble
from src.models.evaluators import HarmBenchEvaluator, LlamaGuard2Evaluator
from src.utils import create_initial_image


class AttackType(Enum):
    UNCONSTRAINED = "unconstrained"
    PGD = "pgd"
    APGD = "apgd"


class VLMEnsembleAttackingSystem(lightning.LightningModule):
    def __init__(
        self,
        wandb_config: Dict[str, Any],
    ):
        super().__init__()
        self.wandb_config = wandb_config
        self.vlm_ensemble = VLMEnsemble(
            model_strs=wandb_config["models_to_attack"],
            model_generation_kwargs=wandb_config["model_generation_kwargs"],
            precision=wandb_config["lightning_kwargs"]["precision"],
        )

        # Load initial image plus prompt and target data.
        tensor_image: torch.Tensor = create_initial_image(
            image_kwargs=wandb_config["image_kwargs"],
            seed=wandb_config["seed"],
        )
        # print(f"tensor_image.shape: {tensor_image.shape}")
        # print(f"tensor_image: {tensor_image}")
       
        if wandb_config["opt_type"] == "full_vlm":
            self.tensor_image = torch.nn.Parameter(tensor_image, requires_grad=True)
            self.convert_tensor_to_pil_image = torchvision.transforms.ToPILImage()
            self.param_to_optimize = [self.tensor_image]
        elif wandb_config["opt_type"] == "lm_only":
            # ensure that only one model is being attacked (we can only attack one model at a time in latent space) and it is a prismatic model
            model_str = wandb_config["models_to_attack"].pop()
            assert len(wandb_config["models_to_attack"]) == 0 and model_str.startswith("prism-")
            vlm = self.vlm_ensemble.vlms_dict[model_str]
            transform_fn = vlm.images_transform_fn

           # transform the image 
            images = tensor_image.repeat(wandb_config["data"]["batch_size"], 1, 1, 1)
            transformed_images: Union[
                torch.Tensor, Dict[str, torch.Tensor]
            ] = transform_fn(images)

            # encode and project
            projected_patch_embeddings = vlm.model.encode_project_images(transformed_images, batch_size=wandb_config["data"]["batch_size"])
            autocast_dtype = vlm.model.llm_backbone.half_precision_dtype
            self.projected_patch_embeddings = torch.nn.Parameter(
                projected_patch_embeddings.to(dtype=autocast_dtype), requires_grad=True
            )
            self.param_to_optimize = [self.projected_patch_embeddings]

        else:
            raise ValueError(f"Invalid optimization type: {wandb_config['opt_type']}")
        self.optimizer_step_counter = 0
        
        # Store image embeddings for language-only optimization
        self.image_embeddings = None
        self.optimize_language_only = wandb_config.get("optimize_language_only", False)

    def to(self, *args, **kwargs):
        # Call parent's to() method
        super().to(*args, **kwargs)
        # Move mask to the same device as tensor_image
        self.mask = self.mask.to(self.tensor_image.device)
        self.original_image = self.original_image.to(self.tensor_image.device)
        return self

    def configure_optimizers(self) -> Dict:
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        # TODO: Maybe add SWA
        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.StochasticWeightAveraging.html#pytorch_lightning.callbacks.StochasticWeightAveraging
        optimization_kwargs = self.wandb_config["optimization"]
        if optimization_kwargs["optimizer"] == "adadelta":
            optimizer = torch.optim.Adadelta(
                self.param_to_optimize,
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
            )
        elif optimization_kwargs["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.param_to_optimize,
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                eps=optimization_kwargs[
                    "eps"
                ],  # https://stackoverflow.com/a/42420014/4570472
            )
        elif optimization_kwargs["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.param_to_optimize,
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                eps=optimization_kwargs[
                    "eps"
                ],  # https://stackoverflow.com/a/42420014/4570472
            )
        elif optimization_kwargs["optimizer"] == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.param_to_optimize,
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                momentum=optimization_kwargs["momentum"],
                eps=1e-4,
            )
        elif optimization_kwargs["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                self.param_to_optimize,
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                momentum=optimization_kwargs["momentum"],
            )
        else:
            # TODO: add adafactor https://pytorch-optimizer.readthedocs.io/en/latest/index.html
            raise NotImplementedError(f"{self.wandb_config['optimizer']}")

        optimizer_and_maybe_others_dict = {
            "optimizer": optimizer,
        }

        # if self.wandb_config["learning_rate_scheduler"] is None:
        #     pass
        # elif (
        #     self.wandb_config["learning_rate_scheduler"]
        #     == "cosine_annealing_warm_restarts"
        # ):
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #         optimizer=optimizer,
        #         T_0=2,
        #     )
        #     optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler
        #
        # elif (
        #     self.wandb_config["learning_rate_scheduler"]
        #     == "linear_warmup_cosine_annealing"
        # ):
        #     from flash.core.optimizers import LinearWarmupCosineAnnealingLR
        #
        #     scheduler = LinearWarmupCosineAnnealingLR(
        #         optimizer=optimizer,
        #         warmup_epochs=1,
        #         max_epochs=self.wandb_config["n_epochs"],
        #     )
        #
        #     optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler
        #
        # elif self.wandb_config["learning_rate_scheduler"] == "reduce_lr_on_plateau":
        #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         factor=0.95,
        #         optimizer=optimizer,
        #         patience=3,
        #     )
        #     optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler
        #     optimizer_and_maybe_others_dict["monitor"] = "train/loss=total_loss"
        # else:
        #     raise NotImplementedError(f"{self.wandb_config['learning_rate_scheduler']}")

        return optimizer_and_maybe_others_dict

    def training_step(
        self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        
        if self.wandb_config["opt_type"] == "full_vlm":
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#training_step
            losses_per_model: Dict[str, torch.Tensor] = self.vlm_ensemble.compute_loss(
                image=self.tensor_image,
                latent_image=None,
                text_data_by_model=batch,
            )
        elif self.wandb_config["opt_type"] == "lm_only":
            losses_per_model: Dict[str, torch.Tensor] = self.vlm_ensemble.compute_loss(
                image=None,
                latent_image=self.projected_patch_embeddings,
                text_data_by_model=batch,
            )
        else:
            raise ValueError(f"Invalid optimization type: {self.wandb_config['opt_type']}")
        for loss_str, loss_val in losses_per_model.items():
            self.log(
                f"loss/{loss_str}",
                loss_val.detach().item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

        self.log(
            "optimizer_step_counter",
            self.optimizer_step_counter,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return losses_per_model["avg"]

    def optimizer_step(self, *args, **kwargs):
        if (
            self.optimizer_step_counter
            % self.wandb_config["lightning_kwargs"]["log_image_every_n_steps"]
        ) == 0:
            opt_type = self.wandb_config.get("opt_type", "")
            log_key = f"jailbreak_step={self.optimizer_step_counter}"

            if opt_type == "lm_only":
                # 1. Convert tensor to CPU and save temporarily
                tensor_to_save = self.projected_patch_embeddings.detach().cpu()
                save_path = f"projected_patch_step={self.optimizer_step_counter}.pt"
                torch.save(tensor_to_save, save_path)

                print(f"projected_patch_embeddings first row: {self.projected_patch_embeddings[0][0]}")

                # 2. Create an artifact and log it
                artifact = wandb.Artifact(
                    name=f"proj_patch_step_{self.optimizer_step_counter}",  # unique name
                    type="tensor",
                )
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)

                # 3. Log just the norm to wandb for live charting
                wandb.log({
                    f"{log_key}/embed_norm": tensor_to_save.to(torch.float32).norm().item()
                })

            else:
                # Log adversarial image
                wandb.log({
                    log_key: wandb.Image(
                        self.convert_tensor_to_pil_image(
                            self.tensor_image[0].detach().to(torch.float32)
                        )
                    )
                })
                with torch.no_grad():
                    self.tensor_image.data = self.tensor_image.data.clamp(min=0.0, max=1.0)

        super().optimizer_step(*args, **kwargs)
        self.optimizer_step_counter += 1
        

class VLMEnsembleEvaluatingSystem(lightning.LightningModule):
    def __init__(
        self,
        wandb_config: Dict[str, Any],
        tensor_image: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.wandb_config = wandb_config
        self.vlm_ensemble = VLMEnsemble(
            model_strs=wandb_config["model_to_eval"],
            model_generation_kwargs=wandb_config["model_generation_kwargs"],
        )
        self.tensor_image = torch.nn.Parameter(tensor_image, requires_grad=False)
        self.wandb_additional_data = {}

    def test_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int):
        if self.tensor_image is None:
            raise ValueError("Image must be provided!")

        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#training_step
        losses_per_model: Dict[str, torch.Tensor] = self.vlm_ensemble.compute_loss(
            image=self.tensor_image,
            text_data_by_model=batch,
        )

        for loss_str, loss_val in losses_per_model.items():
            self.log(
                f"loss/{loss_str}",
                loss_val.detach().item(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        # Make sure the number of optimizer steps is simultaneously logged.
        self.log(
            "optimizer_step_counter",
            self.wandb_additional_data["optimizer_step_counter"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
