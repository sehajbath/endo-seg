"""Training utilities for Swin UNETR with MONAI-style helpers."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from monai.data import decollate_batch
from monai.losses import DiceCELoss
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


def build_loss_and_metrics(config: Dict) -> Tuple[DiceCELoss, DiceMetric, AsDiscrete, AsDiscrete]:
    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        batch=True,
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
    loss_fn: DiceCELoss,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    grad_clip: float | None = None,
    mixed_precision: bool = True,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    model.train()
    meter = AverageMeter("train_loss")
    scaler = scaler or torch.cuda.amp.GradScaler(enabled=mixed_precision)

    start = time.time()
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device).long() 

        # FIX: ensure labels have a channel dimension so shape matches logits
        if labels.ndim == 4:          # [B, H, W, D]
            labels = labels.unsqueeze(1)  # -> [B, 1, H, W, D]

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            logits = model(images)
            loss = loss_fn(logits, labels)

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
                dice_val = dice_metric.aggregate()  # may be tensor or (tensor, not_nans)
                if isinstance(dice_val, tuple):
                    dice_val, _ = dice_val
                print(
                    f"Val Epoch [{epoch}/{max_epochs}] Step [{step}/{len(loader)}] "
                    f"Dice: {dice_val.mean().item():.4f}",
                    flush=True,
                )

    dice_scores = dice_metric.aggregate()
    if isinstance(dice_scores, tuple):
        dice_scores, _ = dice_scores

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
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    best_score = 0.0
    history = {"train_loss": [], "val_dice": [], "val_epochs": []}
    val_every = config.get("save_frequency", 5)

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
            _, mean_dice = validate(
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

            latest_path = save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                mean_dice,
                checkpoint_dir,
                "latest.pth",
            )
            print(f"Saved checkpoint: {latest_path}")

            if mean_dice > best_score:
                best_score = mean_dice
                best_path = save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_score,
                    checkpoint_dir,
                    "best.pth",
                )
                print(f"New best Dice {best_score:.4f}. Saved: {best_path}")

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
