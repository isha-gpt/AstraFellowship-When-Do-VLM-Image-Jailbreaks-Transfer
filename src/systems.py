from enum import Enum, auto
import numpy as np
import lightning
import torch
import torch.optim
import torchvision.transforms
from typing import Any, Dict, List, Optional, Tuple
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
        
        # Store the original image for masked pixels
        self.original_image = tensor_image.clone()
        
        # Create mask for random patch
        patch_size = wandb_config["image_kwargs"]["patch_size"]
        image_size = wandb_config["image_kwargs"]["image_size"]
        
        # Initialize mask as all True (all pixels masked)
        mask = torch.ones_like(tensor_image, dtype=torch.bool)
        
        if patch_size < image_size:
            # Calculate random starting position for the patch
            max_start = image_size - patch_size
            start_h = torch.randint(0, max_start + 1, (1,)).item()
            start_w = torch.randint(0, max_start + 1, (1,)).item()
            
            # Set the patch region to False (unmasked)
            mask[..., start_h:start_h + patch_size, start_w:start_w + patch_size] = False
            
            print(f"\nPatch masking statistics:")
            print(f"- Image size: {image_size}x{image_size}")
            print(f"- Patch size: {patch_size}x{patch_size}")
            print(f"- Patch position: ({start_h}, {start_w})")
            print(f"- Total pixels: {mask.numel()}")
            print(f"- Masked (unchangeable) pixels: {mask.sum().item()}")
            print(f"- Unmasked (changeable) pixels: {(mask.numel() - mask.sum().item()) // 3}")
            print(f"- Percentage masked: {100 * (mask.sum().item() // 3) / (mask.numel() // 3):.1f}%")
        
        self.mask = mask
        self.tensor_image = torch.nn.Parameter(tensor_image, requires_grad=True)
        self.convert_tensor_to_pil_image = torchvision.transforms.ToPILImage()
        self.optimizer_step_counter = 0

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
                [self.tensor_image],
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
            )
        elif optimization_kwargs["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                [self.tensor_image],
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                eps=optimization_kwargs[
                    "eps"
                ],  # https://stackoverflow.com/a/42420014/4570472
            )
        elif optimization_kwargs["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                [self.tensor_image],
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                eps=optimization_kwargs[
                    "eps"
                ],  # https://stackoverflow.com/a/42420014/4570472
            )
        elif optimization_kwargs["optimizer"] == "rmsprop":
            optimizer = torch.optim.RMSprop(
                [self.tensor_image],
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                momentum=optimization_kwargs["momentum"],
                eps=1e-4,
            )
        elif optimization_kwargs["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                [self.tensor_image],
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

        # if batch_idx == 0:
        #     print(torch.cuda.memory_summary())
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass

        return losses_per_model["avg"]

    def optimizer_step(self, *args, **kwargs):
        # Store pre-optimization state for logging
        pre_opt_image = self.tensor_image.clone().detach()
        
        # Perform optimization step
        super().optimizer_step(*args, **kwargs)
        self.optimizer_step_counter += 1
        
        with torch.no_grad():
            # First clamp all values
            self.tensor_image.data = self.tensor_image.data.clamp(min=0.0, max=1.0)
            
            # Calculate changes before applying mask
            pixel_changes = (self.tensor_image.data - pre_opt_image).abs()
            max_change_masked = pixel_changes[self.mask].max().item()
            max_change_unmasked = pixel_changes[~self.mask].max().item()
            avg_change_masked = pixel_changes[self.mask].mean().item()
            avg_change_unmasked = pixel_changes[~self.mask].mean().item()
            
            # Then restore original pixels where mask is True
            self.tensor_image.data[self.mask] = self.original_image[self.mask]
            
            # Log statistics every N steps
            if self.optimizer_step_counter % self.wandb_config["lightning_kwargs"]["log_image_every_n_steps"] == 0:
                print(f"\nStep {self.optimizer_step_counter} pixel change statistics:")
                print(f"- Max change in masked pixels (before restore): {max_change_masked:.6f}")
                print(f"- Max change in unmasked pixels: {max_change_unmasked:.6f}")
                print(f"- Avg change in masked pixels (before restore): {avg_change_masked:.6f}")
                print(f"- Avg change in unmasked pixels: {avg_change_unmasked:.6f}")
                
                # Also log to wandb
                wandb.log({
                    f"jailbreak_image_step={self.optimizer_step_counter}": wandb.Image(
                        data_or_path=self.convert_tensor_to_pil_image(
                            self.tensor_image[0].detach().to(torch.float32)
                        ),
                    ),
                    "pixel_changes/max_change_masked": max_change_masked,
                    "pixel_changes/max_change_unmasked": max_change_unmasked,
                    "pixel_changes/avg_change_masked": avg_change_masked,
                    "pixel_changes/avg_change_unmasked": avg_change_unmasked,
                    "optimizer_step": self.optimizer_step_counter,
                })
                
                # Verify mask is still properly applied
                diff_from_original = (self.tensor_image.data - self.original_image).abs().max().item()
                if diff_from_original > 1e-6:
                    print(f"Warning: Maximum difference from original in masked pixels: {diff_from_original:.6f}")


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
