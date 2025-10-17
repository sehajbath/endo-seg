"""
Create train/validation/test splits for UT-EndoMRI dataset
"""
import logging
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))

from endo_seg.data.io.splits import create_data_splits

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for UT-EndoMRI"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/raw/UT-EndoMRI",
        help="Root directory of UT-EndoMRI dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/splits/split_info.json",
        help="Output file for splits"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="D2_TCPW",
        choices=["D1_MHS", "D2_TCPW"],
        help="Dataset to split"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio for training set"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Ratio for validation set"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Ratio for test set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_paper_split",
        action="store_true",
        help="Use the exact split from the paper for D2_TCPW"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Creating Data Splits")
    logger.info("=" * 60)

    if args.use_paper_split and args.dataset == "D2_TCPW":
        # Use the exact split from the RAovSeg paper
        logger.info("Using exact split from paper")

        # Training/validation subjects: D2-000 to D2-007
        # Testing subjects: D2-008 to D2-037
        train_val_ids = [f"D2-{i:03d}" for i in range(8)]
        test_ids = [f"D2-{i:03d}" for i in range(8, 38)]

        # Split train_val into train and val
        import numpy as np
        np.random.seed(args.seed)
        indices = np.random.permutation(len(train_val_ids))
        n_train = int(len(train_val_ids) * 0.8)  # 80% for train, 20% for val

        train_ids = [train_val_ids[i] for i in indices[:n_train]]
        val_ids = [train_val_ids[i] for i in indices[n_train:]]

        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids,
            'dataset': args.dataset,
            'seed': args.seed,
            'paper_split': True,
            'ratios': {
                'train': len(train_ids) / (len(train_ids) + len(val_ids) + len(test_ids)),
                'val': len(val_ids) / (len(train_ids) + len(val_ids) + len(test_ids)),
                'test': len(test_ids) / (len(train_ids) + len(val_ids) + len(test_ids))
            }
        }

        # Save splits
        import json
        import os
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(splits, f, indent=2)

        logger.info(f"Train: {len(splits['train'])} subjects")
        logger.info(f"Val: {len(splits['val'])} subjects")
        logger.info(f"Test: {len(splits['test'])} subjects")

    else:
        # Create random splits
        splits = create_data_splits(
            data_root=args.data_root,
            output_file=args.output,
            dataset_name=args.dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )

    logger.info("=" * 60)
    logger.info(f"Splits saved to: {args.output}")
    logger.info("=" * 60)

    # Print summary
    print("\nSplit Summary:")
    print(f"  Dataset: {splits['dataset']}")
    print(f"  Train: {len(splits['train'])} subjects ({splits['ratios']['train']:.1%})")
    print(f"  Val: {len(splits['val'])} subjects ({splits['ratios']['val']:.1%})")
    print(f"  Test: {len(splits['test'])} subjects ({splits['ratios']['test']:.1%})")
    print(f"  Random seed: {splits['seed']}")

    if splits.get('paper_split'):
        print("  Using paper's train/test split")

    print("\nTrain subjects:")
    print(f"  {', '.join(splits['train'][:5])}{'...' if len(splits['train']) > 5 else ''}")
    print("\nValidation subjects:")
    print(f"  {', '.join(splits['val'][:5])}{'...' if len(splits['val']) > 5 else ''}")
    print("\nTest subjects:")
    print(f"  {', '.join(splits['test'][:5])}{'...' if len(splits['test']) > 5 else ''}")


if __name__ == "__main__":
    main()
