"""
Checkpoint saving and loading utilities.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints during training."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best: bool = True,
        monitor_metric: str = "val_dice",
        mode: str = "max",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.monitor_metric = monitor_metric
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.mode = mode

        self.checkpoints = []
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_checkpoint_path = None

        self._load_checkpoint_info()

        logger.info(
            "Checkpoint manager initialized at %s (max=%d, monitor=%s, mode=%s)",
            checkpoint_dir,
            max_checkpoints,
            monitor_metric,
            mode,
        )

    def _load_checkpoint_info(self) -> None:
        info_file = self.checkpoint_dir / "checkpoint_info.json"
        if info_file.exists():
            with open(info_file, "r") as f:
                info = json.load(f)
            self.checkpoints = info.get("checkpoints", [])
            self.best_score = info.get("best_score", self.best_score)
            self.best_checkpoint_path = info.get("best_checkpoint_path")

    def _save_checkpoint_info(self) -> None:
        info = {
            "checkpoints": self.checkpoints,
            "best_score": self.best_score,
            "best_checkpoint_path": self.best_checkpoint_path,
        }
        info_file = self.checkpoint_dir / "checkpoint_info.json"
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)

    def _is_better(self, score: float) -> bool:
        return score > self.best_score if self.mode == "max" else score < self.best_score

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        extra_info: Optional[Dict] = None,
    ) -> str:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if extra_info is not None:
            checkpoint["extra_info"] = extra_info

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch{epoch:04d}_{timestamp}.pth"
        checkpoint_path = self.checkpoint_dir / filename

        torch.save(checkpoint, checkpoint_path)
        logger.info("Saved checkpoint: %s", checkpoint_path)

        self.checkpoints.append(
            {
                "path": str(checkpoint_path),
                "epoch": epoch,
                "metrics": metrics,
                "timestamp": checkpoint["timestamp"],
            }
        )

        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint["path"]):
                os.remove(old_checkpoint["path"])
                logger.info("Removed old checkpoint: %s", old_checkpoint["path"])

        if self.save_best and self.monitor_metric in metrics:
            score = metrics[self.monitor_metric]
            if self._is_better(score):
                self.best_score = score
                best_path = self.checkpoint_dir / "best_model.pth"
                torch.save(checkpoint, best_path)
                self.best_checkpoint_path = str(best_path)
                logger.info("New best checkpoint (%.4f) saved to %s", score, best_path)

        self._save_checkpoint_info()
        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.best_checkpoint_path is None or not os.path.exists(self.best_checkpoint_path):
            logger.warning("No best checkpoint found")
            return None

        return self.load_checkpoint(
            self.best_checkpoint_path,
            model,
            optimizer,
            scheduler,
            device,
        )

    def get_latest_checkpoint(self) -> Optional[str]:
        return self.checkpoints[-1]["path"] if self.checkpoints else None

    def resume_from_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[Dict[str, Any]]:
        latest_path = self.get_latest_checkpoint()
        if latest_path is None:
            logger.info("No checkpoint found to resume from")
            return None

        return self.load_checkpoint(latest_path, model, optimizer, scheduler, device)


def save_model_only(
    model: torch.nn.Module,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save only model weights (for deployment/inference)."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }
    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, save_path)
    logger.info("Saved model weights to %s", save_path)


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> torch.nn.Module:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)

    logger.info("Loaded model weights from %s", checkpoint_path)
    return model


def create_checkpoint_from_pretrained(
    pretrained_path: str,
    save_path: str,
    model_name: str,
    additional_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Create a standardized checkpoint from pretrained weights."""
    pretrained = torch.load(pretrained_path)
    checkpoint = {
        "model_state_dict": (
            pretrained
            if not isinstance(pretrained, dict)
            else pretrained.get("model_state_dict", pretrained)
        ),
        "model_name": model_name,
        "source": "pretrained",
        "original_path": pretrained_path,
        "timestamp": datetime.now().isoformat(),
    }

    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, save_path)
    logger.info("Created checkpoint from pretrained weights at %s", save_path)


__all__ = [
    "CheckpointManager",
    "save_model_only",
    "load_model_weights",
    "create_checkpoint_from_pretrained",
]
