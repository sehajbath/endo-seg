"""Training utilities for Swin UNETR with MONAI-style helpers."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch import amp
from monai.data import decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction


@dataclass
class AverageMeter:
    name: str
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def create_optimizer_and_scheduler(model: torch.nn.Module, config: Dict) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get("epochs", 100),
        eta_min=config.get("min_lr", 1e-7),
    )
    return optimizer, scheduler


class DiceCEWithWeights(nn.Module):
    """Combine Dice loss with weighted cross entropy to stabilize rare classes.

    Args:
        class_weights: Weights for CrossEntropy loss per class
        use_generalized_dice: If True, use GeneralizedDiceLoss (better for class imbalance)
        dice_weight: Weight for Dice loss component (default 0.7)
        ce_weight: Weight for CE loss component (default 0.3)
    """

    def __init__(
        self,
        class_weights: Sequence[float] | None = None,
        use_generalized_dice: bool = False,
        dice_weight: float = 0.7,
        ce_weight: float = 0.3,
    ) -> None:
        super().__init__()

        # CRITICAL FIX: Remove squared_pred=True to preserve gradient signal for rare classes
        # For rare classes (endometriomas ~1% of voxels), squaring predictions (e.g., 0.2^2 = 0.04)
        # effectively removes gradient signal, preventing learning
        if use_generalized_dice:
            from monai.losses import GeneralizedDiceLoss
            self.dice = GeneralizedDiceLoss(
                to_onehot_y=True,
                softmax=True,
            )
        else:
            self.dice = DiceLoss(
                to_onehot_y=True,
                softmax=True,
                squared_pred=False,  # ✅ FIXED: Linear predictions
                smooth_nr=1e-5,
                smooth_dr=1e-5,
            )

        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.ce = nn.CrossEntropyLoss(weight=weight)

        # Weighted combination (Dice dominates, CE helps with class balance)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, labels)
        target = labels.squeeze(1) if labels.ndim == logits.ndim else labels
        ce_loss = self.ce(logits, target)

        # Weighted combination instead of simple sum
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


class FocalDiceCELoss(nn.Module):
    """Focal Dice + Focal CE loss for extreme class imbalance (e.g., endometrioma finetuning).

    Focal loss down-weights easy examples and focuses gradient on hard examples.
    Particularly effective for rare classes occupying <1% of voxels.

    Args:
        class_weights: Weights for FocalLoss per class
        gamma: Focusing parameter (default 2.0). Higher gamma = more focus on hard examples
        dice_weight: Weight for Dice Focal loss component (default 0.7)
        focal_weight: Weight for Focal CE loss component (default 0.3)
    """

    def __init__(
        self,
        class_weights: Sequence[float] | None = None,
        gamma: float = 2.0,
        dice_weight: float = 0.7,
        focal_weight: float = 0.3,
    ) -> None:
        super().__init__()

        from monai.losses import DiceFocalLoss, FocalLoss

        self.dice_focal = DiceFocalLoss(
            to_onehot_y=True,
            softmax=True,
            gamma=gamma,  # Focus on hard examples
        )

        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.focal = FocalLoss(
            to_onehot_y=True,
            gamma=gamma,
            weight=weight,
        )

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice_focal(logits, labels)
        target = labels.squeeze(1) if labels.ndim == logits.ndim else labels
        focal_loss = self.focal(logits, target)

        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


def build_loss_and_metrics(config: Dict) -> Tuple[nn.Module, DiceMetric, AsDiscrete, AsDiscrete]:
    """Build loss function and metrics based on configuration.

    Args:
        config: Training configuration dictionary

    Supported loss types (via config["loss_type"]):
        - "dice_ce" (default): DiceCEWithWeights
        - "focal": FocalDiceCELoss (for extreme class imbalance)

    Returns:
        Tuple of (loss_fn, dice_metric, post_pred, post_label)
    """
    class_weights = config.get("class_weights")
    loss_type = config.get("loss_type", "dice_ce")

    if loss_type == "focal":
        # Focal loss for finetuning on rare classes
        gamma = config.get("focal_gamma", 2.0)
        dice_weight = config.get("dice_weight", 0.7)
        focal_weight = config.get("focal_weight", 0.3)

        loss_fn = FocalDiceCELoss(
            class_weights=class_weights,
            gamma=gamma,
            dice_weight=dice_weight,
            focal_weight=focal_weight,
        )
    else:
        # Standard Dice + CE loss (for pretraining or general use)
        use_generalized_dice = config.get("use_generalized_dice", False)
        dice_weight = config.get("dice_weight", 0.7)
        ce_weight = config.get("ce_weight", 0.3)

        loss_fn = DiceCEWithWeights(
            class_weights=class_weights,
            use_generalized_dice=use_generalized_dice,
            dice_weight=dice_weight,
            ce_weight=ce_weight,
        )

    dice_metric = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )
    post_pred = AsDiscrete(argmax=True, to_onehot=config["num_classes"])
    post_label = AsDiscrete(to_onehot=config["num_classes"])
    return loss_fn, dice_metric, post_pred, post_label


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    grad_clip: float | None = None,
    mixed_precision: bool = True,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    model.train()
    meter = AverageMeter("train_loss")
    scaler = scaler or amp.GradScaler("cuda", enabled=mixed_precision)

    start = time.time()
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device).long()

        if labels.ndim == 4:
            labels = labels.unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=mixed_precision):
            logits = model(images)
            loss = loss_fn(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print(
                f"Skipping batch {step} at epoch {epoch} due to invalid loss ({loss.item()})",
                flush=True,
            )
            continue

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        meter.update(loss.item(), images.size(0))

        if step % 5 == 0:
            elapsed = time.time() - start
            print(
                f"Epoch [{epoch}/{max_epochs}] Step [{step}/{len(loader)}] "
                f"Loss: {meter.avg:.4f} Time: {elapsed:.1f}s",
                flush=True,
            )
            start = time.time()

    return meter.avg


def validate(
    model,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    dice_metric: DiceMetric,
    post_pred: AsDiscrete,
    post_label: AsDiscrete,
) -> Tuple[torch.Tensor, float]:
    model.eval()
    dice_metric.reset()

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            if labels.ndim == 4:
                labels = labels.unsqueeze(1)  # [B, 1, H, W, D]

            logits = model.infer_sliding_window(images)

            preds = decollate_batch(logits)
            ground_truth = decollate_batch(labels)

            preds = [post_pred(p) for p in preds]
            ground_truth = [post_label(g) for g in ground_truth]

            dice_metric(y_pred=preds, y=ground_truth)

            if step % 5 == 0 or step == len(loader):
                dice_val = dice_metric.aggregate()
                if isinstance(dice_val, tuple):
                    dice_val, _ = dice_val
                dice_val = dice_val.detach().cpu()
                print(
                    f"Val Epoch [{epoch}/{max_epochs}] Step [{step}/{len(loader)}] "
                    f"Dice: {dice_val.mean().item():.4f}",
                    flush=True,
                )

    dice_scores = dice_metric.aggregate()
    if isinstance(dice_scores, tuple):
        dice_scores, _ = dice_scores

    dice_scores = dice_scores.detach().cpu()
    mean_dice = dice_scores.mean().item()
    dice_metric.reset()
    return dice_scores, mean_dice


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    score: float,
    checkpoint_dir: str,
    filename: str,
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "score": score,
        },
        path,
    )
    return path


def train_loop(
    model,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict,
    device: torch.device,
) -> Dict[str, List[float]]:
    epochs = config.get("epochs", 100)
    grad_clip = config.get("grad_clip", 1.0)
    mixed_precision = config.get("mixed_precision", True)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    loss_fn, dice_metric, post_pred, post_label = build_loss_and_metrics(config)
    loss_fn = loss_fn.to(device)
    scaler = amp.GradScaler("cuda", enabled=mixed_precision)

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    best_score = float("-inf")
    history = {"train_loss": [], "val_dice": [], "val_epochs": [], "priority_dice": [], "val_class_dice": []}
    val_every = config.get("save_frequency", 5)
    patience = config.get("target_metric_patience")
    epochs_without_improve = 0

    priority_classes: Sequence[int] = config.get("priority_classes", [2, 3])
    if isinstance(priority_classes, int):
        priority_classes = [priority_classes]
    priority_classes = [c for c in priority_classes if 0 <= c < config["num_classes"]]
    if not priority_classes:
        priority_classes = list(range(config["num_classes"]))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epoch,
            epochs,
            grad_clip=grad_clip,
            mixed_precision=mixed_precision,
            scaler=scaler,
        )
        history["train_loss"].append(train_loss)

        scheduler.step()

        if epoch % val_every == 0 or epoch == epochs:
            dice_scores, mean_dice = validate(
                model,
                val_loader,
                device,
                epoch,
                epochs,
                dice_metric,
                post_pred,
                post_label,
            )
            history["val_dice"].append(mean_dice)
            history["val_epochs"].append(epoch)
            history["val_class_dice"].append(dice_scores.tolist())

            class_names = config.get("class_names")
            if class_names and len(class_names) == len(dice_scores):
                per_class = ", ".join(
                    f"{name}:{score:.3f}" for name, score in zip(class_names, dice_scores.tolist())
                )
                print(f"Per-class Dice @ epoch {epoch}: {per_class}", flush=True)

            selected = dice_scores[priority_classes].mean().item()
            history["priority_dice"].append(selected)
            print(
                f"Target-class Dice ({priority_classes}) at epoch {epoch}: {selected:.4f} (mean Dice {mean_dice:.4f})",
                flush=True,
            )

            latest_path = save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                selected,
                checkpoint_dir,
                "latest.pth",
            )
            print(f"Saved checkpoint: {latest_path}")

            if selected > best_score:
                best_score = selected
                epochs_without_improve = 0
                best_path = save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_score,
                    checkpoint_dir,
                    "best.pth",
                )
                print(
                    f"New best target Dice {best_score:.4f} (priority classes {priority_classes}). Saved: {best_path}",
                    flush=True,
                )
            else:
                epochs_without_improve += 1
                if patience is not None and epochs_without_improve >= patience:
                    print(
                        f"Early stopping triggered after {patience} validation checks without target metric improvement.",
                        flush=True,
                    )
                    break

    return history


__all__ = [
    "AverageMeter",
    "create_optimizer_and_scheduler",
    "build_loss_and_metrics",
    "train_one_epoch",
    "validate",
    "save_checkpoint",
    "train_loop",
]
