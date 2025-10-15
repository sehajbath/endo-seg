"""
Logging utilities for experiment tracking
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import coloredlogs
import json
from datetime import datetime


def setup_logging(
        log_dir: Optional[str] = None,
        log_level: str = "INFO",
        experiment_name: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        experiment_name: Name of experiment for log file

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler with colored output
    coloredlogs.install(
        level=log_level.upper(),
        logger=logger,
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler if log_dir provided
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if experiment_name:
            log_file = log_path / f"{experiment_name}_{timestamp}.log"
        else:
            log_file = log_path / f"experiment_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger


class ExperimentLogger:
    """Logger for experiment tracking with W&B and TensorBoard support"""

    def __init__(
            self,
            experiment_name: str,
            log_dir: str,
            use_wandb: bool = False,
            use_tensorboard: bool = True,
            config: Optional[Dict] = None
    ):
        """
        Initialize experiment logger

        Args:
            experiment_name: Name of experiment
            log_dir: Directory for logs
            use_wandb: Whether to use Weights & Biases
            use_tensorboard: Whether to use TensorBoard
            config: Configuration dictionary to log
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup standard logging
        self.logger = setup_logging(
            log_dir=str(self.log_dir),
            experiment_name=experiment_name
        )

        # Setup W&B
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(
                    project="endometriosis-uncertainty-seg",
                    name=experiment_name,
                    config=config
                )
                self.logger.info("Initialized Weights & Biases logging")
            except ImportError:
                self.logger.warning("wandb not installed, disabling W&B logging")
                self.use_wandb = False

        # Setup TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard"
                self.tb_writer = SummaryWriter(str(tb_dir))
                self.logger.info(f"Initialized TensorBoard logging: {tb_dir}")
            except ImportError:
                self.logger.warning("tensorboard not installed, disabling TB logging")
                self.use_tensorboard = False

        # Save config
        if config is not None:
            self.save_config(config)

        self.step = 0

    def save_config(self, config: Dict):
        """Save configuration to file"""
        config_file = self.log_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        self.logger.info(f"Saved config to {config_file}")

    def log_metrics(
            self,
            metrics: Dict[str, float],
            step: Optional[int] = None,
            prefix: str = ""
    ):
        """
        Log metrics to all configured backends

        Args:
            metrics: Dictionary of metric names and values
            step: Training step/epoch
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        if step is None:
            step = self.step
        else:
            self.step = step

        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Log to standard logger
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {metrics_str}")

        # Log to W&B
        if self.use_wandb:
            self.wandb.log(metrics, step=step)

        # Log to TensorBoard
        if self.use_tensorboard:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, step)

    def log_image(
            self,
            tag: str,
            image: Any,
            step: Optional[int] = None
    ):
        """
        Log image to TensorBoard and W&B

        Args:
            tag: Image tag/name
            image: Image tensor or array
            step: Training step
        """
        if step is None:
            step = self.step

        if self.use_tensorboard:
            self.tb_writer.add_image(tag, image, step)

        if self.use_wandb:
            self.wandb.log({tag: self.wandb.Image(image)}, step=step)

    def log_histogram(
            self,
            tag: str,
            values: Any,
            step: Optional[int] = None
    ):
        """Log histogram to TensorBoard"""
        if step is None:
            step = self.step

        if self.use_tensorboard:
            self.tb_writer.add_histogram(tag, values, step)

    def log_text(
            self,
            tag: str,
            text: str,
            step: Optional[int] = None
    ):
        """Log text to TensorBoard"""
        if step is None:
            step = self.step

        if self.use_tensorboard:
            self.tb_writer.add_text(tag, text, step)

    def save_checkpoint_info(self, checkpoint_path: str, metrics: Dict):
        """Save checkpoint information"""
        checkpoint_info = {
            'path': checkpoint_path,
            'step': self.step,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        info_file = self.log_dir / "checkpoints_info.json"

        # Load existing info
        if info_file.exists():
            with open(info_file, 'r') as f:
                all_info = json.load(f)
        else:
            all_info = []

        all_info.append(checkpoint_info)

        # Save updated info
        with open(info_file, 'w') as f:
            json.dump(all_info, f, indent=2)

        self.logger.info(f"Saved checkpoint info: {checkpoint_path}")

    def finish(self):
        """Close all logging backends"""
        if self.use_wandb:
            self.wandb.finish()

        if self.use_tensorboard:
            self.tb_writer.close()

        self.logger.info("Experiment logging finished")


class MetricsTracker:
    """Track and compute running metrics"""

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float], batch_size: int = 1):
        """
        Update metrics with new batch

        Args:
            metrics: Dictionary of metric values
            batch_size: Batch size for weighted averaging
        """
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0

            self.metrics[name] += value * batch_size
            self.counts[name] += batch_size

    def compute(self) -> Dict[str, float]:
        """
        Compute average metrics

        Returns:
            Dictionary of averaged metrics
        """
        avg_metrics = {}
        for name in self.metrics:
            avg_metrics[name] = self.metrics[name] / max(self.counts[name], 1)
        return avg_metrics

    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
        self.counts = {}

    def get_last(self, name: str) -> float:
        """Get last recorded value for a metric"""
        if name in self.metrics and self.counts[name] > 0:
            return self.metrics[name] / self.counts[name]
        return 0.0


def log_system_info(logger: logging.Logger):
    """Log system and environment information"""
    import platform
    import torch

    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    logger.info("=" * 60)


if __name__ == "__main__":
    # Example usage
    logger = ExperimentLogger(
        experiment_name="test_experiment",
        log_dir="experiments/experiment_logs",
        use_wandb=False,
        use_tensorboard=True,
        config={'lr': 0.001, 'batch_size': 2}
    )

    # Log some metrics
    logger.log_metrics({'loss': 0.5, 'dice': 0.75}, step=0, prefix='train/')
    logger.log_metrics({'loss': 0.3, 'dice': 0.85}, step=1, prefix='val/')

    # Finish logging
    logger.finish()

    print("Example logging completed successfully!")