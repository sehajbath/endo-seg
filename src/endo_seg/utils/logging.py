"""
Logging utilities for experiment tracking.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import coloredlogs


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    experiment_name: Optional[str] = None,
) -> logging.Logger:
    """Configure root logging with optional file output."""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    coloredlogs.install(
        level=log_level.upper(),
        logger=logger,
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = (
            log_path / f"{experiment_name}_{timestamp}.log"
            if experiment_name
            else log_path / f"experiment_{timestamp}.log"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info("Logging to file: %s", log_file)

    return logger


class ExperimentLogger:
    """Logger for experiment tracking with optional W&B and TensorBoard support."""

    def __init__(
        self,
        experiment_name: str,
        log_dir: str,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(log_dir=str(self.log_dir), experiment_name=experiment_name)

        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb

                self.wandb = wandb
                self.wandb.init(
                    project="endometriosis-uncertainty-seg",
                    name=experiment_name,
                    config=config,
                )
                self.logger.info("Initialized Weights & Biases logging")
            except ImportError:
                self.logger.warning("wandb not installed, disabling W&B logging")
                self.use_wandb = False

        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = self.log_dir / "tensorboard"
                self.tb_writer = SummaryWriter(str(tb_dir))
                self.logger.info("Initialized TensorBoard logging at %s", tb_dir)
            except ImportError:
                self.logger.warning("tensorboard not installed, disabling TB logging")
                self.use_tensorboard = False

        if config is not None:
            self.save_config(config)

        self.step = 0

    def save_config(self, config: Dict[str, Any]) -> None:
        config_file = self.log_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2, default=str)
        self.logger.info("Saved config to %s", config_file)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        step = self.step if step is None else step
        self.step = step

        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info("Step %d - %s", step, metrics_str)

        if self.use_wandb:
            self.wandb.log(metrics, step=step)

        if self.use_tensorboard:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, step)

    def log_image(self, tag: str, image: Any, step: Optional[int] = None) -> None:
        step = self.step if step is None else step

        if self.use_tensorboard:
            self.tb_writer.add_image(tag, image, step)

        if self.use_wandb:
            self.wandb.log({tag: self.wandb.Image(image)}, step=step)

    def log_histogram(self, tag: str, values: Any, step: Optional[int] = None) -> None:
        step = self.step if step is None else step
        if self.use_tensorboard:
            self.tb_writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        step = self.step if step is None else step
        if self.use_tensorboard:
            self.tb_writer.add_text(tag, text, step)

    def save_checkpoint_info(self, checkpoint_path: str, metrics: Dict[str, float]) -> None:
        checkpoint_info = {
            "path": checkpoint_path,
            "step": self.step,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        info_file = self.log_dir / "checkpoints_info.json"

        if info_file.exists():
            with open(info_file, "r") as f:
                all_info = json.load(f)
        else:
            all_info = []

        all_info.append(checkpoint_info)

        with open(info_file, "w") as f:
            json.dump(all_info, f, indent=2)

        self.logger.info("Saved checkpoint info: %s", checkpoint_path)

    def finish(self) -> None:
        if self.use_wandb:
            self.wandb.finish()

        if self.use_tensorboard:
            self.tb_writer.close()

        self.logger.info("Experiment logging finished")


class MetricsTracker:
    """Track and compute running metrics."""

    def __init__(self) -> None:
        self.metrics: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], batch_size: int = 1) -> None:
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            self.metrics[name] += value * batch_size
            self.counts[name] += batch_size

    def compute(self) -> Dict[str, float]:
        return {
            name: self.metrics[name] / max(self.counts[name], 1) for name in self.metrics
        }

    def reset(self) -> None:
        self.metrics = {}
        self.counts = {}

    def get_last(self, name: str) -> float:
        if name in self.metrics and self.counts[name] > 0:
            return self.metrics[name] / self.counts[name]
        return 0.0


def log_system_info(logger: logging.Logger) -> None:
    """Log system and environment information."""
    import platform

    import torch

    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)
    logger.info("Python version: %s", sys.version)
    logger.info("Platform: %s", platform.platform())
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())

    if torch.cuda.is_available():
        logger.info("CUDA version: %s", torch.version.cuda)
        logger.info("Number of GPUs: %d", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            logger.info("GPU %d: %s", i, torch.cuda.get_device_name(i))
            logger.info(
                "  Memory: %.2f GB",
                torch.cuda.get_device_properties(i).total_memory / 1e9,
            )

    logger.info("=" * 60)


__all__ = [
    "setup_logging",
    "ExperimentLogger",
    "MetricsTracker",
    "log_system_info",
]
