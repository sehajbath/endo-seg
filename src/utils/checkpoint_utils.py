"""
Checkpoint saving and loading utilities
"""
import torch
import os
from pathlib import Path
from typing import Dict, Optional, Any
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints during training"""

    def __init__(
            self,
            checkpoint_dir: str,
            max_checkpoints: int = 5,
            save_best: bool = True,
            monitor_metric: str = "val_dice",
            mode: str = "max"
    ):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Whether to save best checkpoint separately
            monitor_metric: Metric to monitor for best checkpoint
            mode: 'max' or 'min' for best checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.monitor_metric = monitor_metric
        self.mode = mode

        # Track checkpoints
        self.checkpoints = []
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.best_checkpoint_path = None

        # Load existing checkpoint info if available
        self._load_checkpoint_info()

        logger.info(f"Checkpoint manager initialized: {checkpoint_dir}")
        logger.info(f"  Max checkpoints: {max_checkpoints}")
        logger.info(f"  Monitor metric: {monitor_metric} ({mode})")

    def _load_checkpoint_info(self):
        """Load checkpoint information from file"""
        info_file = self.checkpoint_dir / "checkpoint_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                info = json.load(f)
                self.checkpoints = info.get('checkpoints', [])
                self.best_score = info.get('best_score', self.best_score)
                self.best_checkpoint_path = info.get('best_checkpoint_path')
                logger.info(f"Loaded checkpoint info from {info_file}")

    def _save_checkpoint_info(self):
        """Save checkpoint information to file"""
        info = {
            'checkpoints': self.checkpoints,
            'best_score': self.best_score,
            'best_checkpoint_path': self.best_checkpoint_path
        }

        info_file = self.checkpoint_dir / "checkpoint_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)

    def _is_better(self, score: float) -> bool:
        """Check if score is better than current best"""
        if self.mode == 'max':
            return score > self.best_score
        else:
            return score < self.best_score

    def save_checkpoint(
            self,
            model: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer],
            scheduler: Optional[Any],
            epoch: int,
            metrics: Dict[str, float],
            extra_info: Optional[Dict] = None
    ) -> str:
        """
        Save checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            metrics: Dictionary of metrics
            extra_info: Additional information to save

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if extra_info is not None:
            checkpoint['extra_info'] = extra_info

        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"checkpoint_epoch{epoch:04d}_{timestamp}.pth"
        checkpoint_path = self.checkpoint_dir / filename

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Update checkpoint list
        self.checkpoints.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': checkpoint['timestamp']
        })

        # Remove old checkpoints if exceeding max
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint['path']):
                os.remove(old_checkpoint['path'])
                logger.info(f"Removed old checkpoint: {old_checkpoint['path']}")

        # Check if this is the best checkpoint
        if self.save_best and self.monitor_metric in metrics:
            score = metrics[self.monitor_metric]
            if self._is_better(score):
                self.best_score = score
                best_path = self.checkpoint_dir / "best_model.pth"

                # Save best checkpoint
                torch.save(checkpoint, best_path)
                self.best_checkpoint_path = str(best_path)

                logger.info(f"New best checkpoint! {self.monitor_metric}: {score:.4f}")
                logger.info(f"Saved to: {best_path}")

        # Save checkpoint info
        self._save_checkpoint_info()

        return str(checkpoint_path)

    def load_checkpoint(
            self,
            checkpoint_path: str,
            model: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[Any] = None,
            device: Optional[torch.device] = None
    ) -> Dict:
        """
        Load checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load to

        Returns:
            Checkpoint dictionary
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})

        logger.info(f"Loaded checkpoint from epoch {epoch}")
        logger.info(f"Metrics: {metrics}")

        return checkpoint

    def load_best_checkpoint(
            self,
            model: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[Any] = None,
            device: Optional[torch.device] = None
    ) -> Optional[Dict]:
        """
        Load best checkpoint

        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load to

        Returns:
            Checkpoint dictionary or None if no best checkpoint exists
        """
        if self.best_checkpoint_path is None or not os.path.exists(self.best_checkpoint_path):
            logger.warning("No best checkpoint found")
            return None

        return self.load_checkpoint(
            self.best_checkpoint_path,
            model,
            optimizer,
            scheduler,
            device
        )

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]['path']

    def resume_from_latest(
            self,
            model: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[Any] = None,
            device: Optional[torch.device] = None
    ) -> Optional[Dict]:
        """
        Resume training from latest checkpoint

        Returns:
            Checkpoint dictionary or None if no checkpoint exists
        """
        latest_path = self.get_latest_checkpoint()
        if latest_path is None:
            logger.info("No checkpoint found to resume from")
            return None

        return self.load_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            device
        )


def save_model_only(
        model: torch.nn.Module,
        save_path: str,
        metadata: Optional[Dict] = None
):
    """
    Save only model weights (for deployment/inference)

    Args:
        model: Model to save
        save_path: Path to save model
        metadata: Optional metadata to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat()
    }

    if metadata is not None:
        checkpoint['metadata'] = metadata

    torch.save(checkpoint, save_path)
    logger.info(f"Saved model weights to: {save_path}")


def load_model_weights(
        model: torch.nn.Module,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        strict: bool = True
) -> torch.nn.Module:
    """
    Load model weights from checkpoint

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        device: Device to load to
        strict: Whether to strictly enforce weight matching

    Returns:
        Model with loaded weights
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load weights
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)

    logger.info(f"Loaded model weights from: {checkpoint_path}")

    return model


def create_checkpoint_from_pretrained(
        pretrained_path: str,
        save_path: str,
        model_name: str,
        additional_info: Optional[Dict] = None
):
    """
    Create a standardized checkpoint from pretrained weights

    Args:
        pretrained_path: Path to pretrained weights
        save_path: Path to save standardized checkpoint
        model_name: Name of the model
        additional_info: Additional information to include
    """
    # Load pretrained weights
    pretrained = torch.load(pretrained_path)

    # Create standardized checkpoint
    checkpoint = {
        'model_state_dict': pretrained if not isinstance(pretrained, dict) else pretrained.get('model_state_dict',
                                                                                               pretrained),
        'model_name': model_name,
        'source': 'pretrained',
        'original_path': pretrained_path,
        'timestamp': datetime.now().isoformat()
    }

    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, save_path)
    logger.info(f"Created checkpoint from pretrained: {save_path}")


if __name__ == "__main__":
    import torch.nn as nn

    logging.basicConfig(level=logging.INFO)

    # Create dummy model
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(
        checkpoint_dir="test_checkpoints",
        max_checkpoints=3,
        save_best=True,
        monitor_metric="val_dice",
        mode="max"
    )

    # Save some checkpoints
    for epoch in range(5):
        metrics = {
            'train_loss': 1.0 / (epoch + 1),
            'val_dice': 0.5 + epoch * 0.1
        }

        ckpt_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=epoch,
            metrics=metrics
        )

    print(f"\nBest checkpoint: {ckpt_manager.best_checkpoint_path}")
    print(f"Best score: {ckpt_manager.best_score}")

    # Clean up
    import shutil

    if os.path.exists("test_checkpoints"):
        shutil.rmtree("test_checkpoints")

    print("\n✓ Checkpoint utilities test successful!")