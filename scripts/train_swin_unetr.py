"""Command-line entrypoint for Swin UNETR training on UT-EndoMRI.

Usage example:

    python scripts/train_swin_unetr.py \
        --config configs/config.yaml \
        --run-name swin_unetr_stratified

The script will create (or reuse) a stratified patient-level split for the
specified dataset, build dataloaders with foreground-biased sampling, and run
the SwinUNETR training loop that prioritizes ovary/endometrioma Dice.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))

from endo_seg.config.loader import load_yaml_config
from endo_seg.data import get_dataloaders
from endo_seg.data.core.dataloader import compute_class_weights
from endo_seg.data.io.splits import (
    compute_patient_label_stats,
    create_data_splits,
    load_data_splits,
    log_split_summary,
    summarize_split_counts,
)
from endo_seg.models.swin_unetr_uncertainty import SwinUNETRWithUncertainty
from endo_seg.training.swin_unetr_trainer import train_loop


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_swin_unetr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swin UNETR on UT-EndoMRI")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--splits-file", type=str, default=None, help="Optional explicit split file")
    parser.add_argument("--run-name", type=str, default="swin_unetr_run")
    parser.add_argument("--no-stratified", action="store_true", help="Disable patient-level stratification")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    return parser.parse_args()


def _paper_split(seed: int) -> Dict[str, list]:
    train_val_ids = [f"D2-{i:03d}" for i in range(8)]
    test_ids = [f"D2-{i:03d}" for i in range(8, 38)]
    rng = np.random.default_rng(seed)
    rng.shuffle(train_val_ids)
    n_train = max(1, int(len(train_val_ids) * 0.8))
    train_ids = train_val_ids[:n_train]
    val_ids = train_val_ids[n_train:]
    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "dataset": "D2_TCPW",
        "seed": seed,
        "paper_split": True,
        "ratios": {
            "train": len(train_ids) / (len(train_ids) + len(val_ids) + len(test_ids)),
            "val": len(val_ids) / (len(train_ids) + len(val_ids) + len(test_ids)),
            "test": len(test_ids) / (len(train_ids) + len(val_ids) + len(test_ids)),
        },
    }


def ensure_splits(config: Dict, args: argparse.Namespace) -> Tuple[Dict, Path]:
    split_cfg = config.get("splits", {})
    dataset_name = split_cfg.get("dataset_name", "D2_TCPW")
    seed = split_cfg.get("seed", config.get("seed", 42))
    stratified = split_cfg.get("stratified", True) and not args.no_stratified
    use_paper = split_cfg.get("use_paper_split", False)

    sequence_cfg = config.get("sequences", {})
    enabled_sequences = [seq for seq, enabled in sequence_cfg.items() if enabled]
    primary_sequence = enabled_sequences[0] if enabled_sequences else None

    split_dir = Path(config["paths"].get("splits_dir", "data/splits"))
    split_dir.mkdir(parents=True, exist_ok=True)
    split_tag = "paper" if use_paper else ("strat" if stratified else "random")
    default_split_file = config["paths"].get("default_split_file")
    split_path = Path(
        args.splits_file
        or default_split_file
        or split_dir / f"{dataset_name}_{split_tag}_seed{seed}.json"
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if not split_path.exists():
        if use_paper and dataset_name == "D2_TCPW":
            splits = _paper_split(seed)
            with open(split_path, "w") as f:
                json.dump(splits, f, indent=2)
        else:
            splits = create_data_splits(
                data_root=config["paths"]["data_root"],
                output_file=str(split_path),
                dataset_name=dataset_name,
                train_ratio=split_cfg.get("train_ratio", 0.7),
                val_ratio=split_cfg.get("val_ratio", 0.15),
                test_ratio=split_cfg.get("test_ratio", 0.15),
                seed=seed,
                stratified=stratified,
                primary_sequence=primary_sequence,
            )
    else:
        splits = load_data_splits(str(split_path))

    if stratified and not splits.get("patient_stats"):
        patient_stats = compute_patient_label_stats(
            data_root=config["paths"]["data_root"],
            dataset_name=dataset_name,
        )
        splits["patient_stats"] = patient_stats
        splits["split_summary"] = {
            split_name: summarize_split_counts(splits.get(split_name, []), patient_stats)
            for split_name in ("train", "val", "test")
        }
        with open(split_path, "w") as f:
            json.dump(splits, f, indent=2)

    if splits.get("split_summary"):
        log_split_summary(splits["split_summary"])

    return splits, split_path


def build_model(config: Dict) -> SwinUNETRWithUncertainty:
    model_cfg = config.get("model", {})
    input_cfg = model_cfg.get("input", {})
    output_cfg = model_cfg.get("output", {})
    swin_cfg = model_cfg.get("swin_unetr", {})

    return SwinUNETRWithUncertainty(
        in_channels=input_cfg.get("in_channels", 1),
        out_channels=output_cfg.get("num_classes", 4),
        feature_size=swin_cfg.get("feature_size", 48),
        drop_rate=swin_cfg.get("drop_rate", 0.0),
        attn_drop_rate=swin_cfg.get("attn_drop_rate", 0.0),
        dropout_path_rate=swin_cfg.get("dropout_path_rate", 0.0),
        use_checkpoint=swin_cfg.get("use_checkpoint", True),
        roi_size=tuple(swin_cfg.get("img_size", [128, 128, 32])),
        sw_batch_size=swin_cfg.get("sw_batch_size", 2),
        infer_overlap=swin_cfg.get("infer_overlap", 0.5),
        spatial_dims=input_cfg.get("spatial_dims", 3),
        img_size=swin_cfg.get("img_size"),
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    splits, split_path = ensure_splits(config, args)
    logger.info("Using splits file: %s", split_path)

    data_root = config["paths"]["data_root"]
    dataset_name = config.get("splits", {}).get("dataset_name", "D2_TCPW")

    sequences = [seq for seq, enabled in config.get("sequences", {}).items() if enabled]
    structures = [struct for struct, enabled in config.get("structures", {}).items() if enabled]

    training_cfg = config.get("training", {}).copy()
    trainer_config = training_cfg.copy()
    trainer_config["num_classes"] = config.get("model", {}).get("output", {}).get("num_classes", 4)
    trainer_config["mixed_precision"] = config.get("mixed_precision", True)
    checkpoint_dir = Path(config["paths"].get("checkpoint_dir", "checkpoints")) / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer_config["checkpoint_dir"] = str(checkpoint_dir)
    trainer_config["priority_classes"] = training_cfg.get("priority_classes", [2, 3])
    trainer_config["target_metric_patience"] = training_cfg.get("early_stopping", {}).get("patience")
    trainer_config["save_frequency"] = config.get("logging", {}).get("save_frequency", 5)
    trainer_config["class_names"] = ["background", "uterus", "ovary", "endometrioma"]

    if training_cfg.get("use_class_weights", True):
        class_weights = compute_class_weights(
            data_root=data_root,
            subject_ids=splits["train"],
            sequences=sequences,
            structures=structures,
            dataset_name=dataset_name,
            num_classes=trainer_config["num_classes"],
        )
        trainer_config["class_weights"] = class_weights.tolist()
        logger.info("Class weights: %s", trainer_config["class_weights"])
    elif training_cfg.get("class_weights"):
        trainer_config["class_weights"] = training_cfg["class_weights"]

    dataloaders = get_dataloaders(
        data_root=data_root,
        splits=splits,
        config=config,
        dataset_name=dataset_name,
        num_workers=training_cfg.get("num_workers", 4),
    )

    model = build_model(config)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    model.to(device)

    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    history = train_loop(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        config=trainer_config,
        device=device,
    )

    logger.info("Training complete. Last recorded val Dice: %s", history["val_dice"][-1] if history["val_dice"] else "n/a")


if __name__ == "__main__":
    main()
