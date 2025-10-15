"""
Exploratory Data Analysis for UT-EndoMRI dataset
"""
import argparse
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.utils import (
    load_nifti,
    get_dataset_statistics,
    parse_filename,
    get_subject_files
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_dataset_overview(data_root: str, dataset_name: str):
    """Print overview of dataset"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Dataset Overview: {dataset_name}")
    logger.info(f"{'=' * 60}")

    stats = get_dataset_statistics(data_root, dataset_name)

    print(f"\nNumber of subjects: {stats['num_subjects']}")
    print(f"MRI sequences: {', '.join(stats['sequences'])}")
    print(f"Structures annotated: {', '.join(stats['structures'])}")
    if stats['raters']:
        print(f"Number of raters: {len(stats['raters'])}")
        print(f"Rater IDs: {', '.join(stats['raters'])}")

    return stats


def analyze_image_properties(data_root: str, dataset_name: str):
    """Analyze image dimensions and spacing"""
    logger.info(f"\n{'=' * 60}")
    logger.info("Analyzing Image Properties")
    logger.info(f"{'=' * 60}")

    dataset_path = Path(data_root) / dataset_name

    shapes = []
    spacings = []
    intensities = []

    for subject_dir in sorted(dataset_path.iterdir()):
        if not subject_dir.is_dir():
            continue

        # Get first image file
        image_files = list(subject_dir.glob("*T2*.nii.gz"))
        if not image_files:
            continue

        # Filter out labels
        image_files = [f for f in image_files if '_r' not in f.name]
        if not image_files:
            continue

        try:
            data, img = load_nifti(str(image_files[0]))
            shapes.append(data.shape)

            # Get spacing from affine
            affine = img.affine
            spacing = np.abs(np.diag(affine)[:3])
            spacings.append(spacing)

            # Intensity statistics
            intensities.append({
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'p1': np.percentile(data, 1),
                'p99': np.percentile(data, 99)
            })

        except Exception as e:
            logger.warning(f"Error loading {image_files[0]}: {e}")
            continue

    # Print shape statistics
    shapes = np.array(shapes)
    spacings = np.array(spacings)

    print("\nImage Shape Statistics:")
    print(f"  Mean: {shapes.mean(axis=0)}")
    print(f"  Std: {shapes.std(axis=0)}")
    print(f"  Min: {shapes.min(axis=0)}")
    print(f"  Max: {shapes.max(axis=0)}")

    print("\nVoxel Spacing Statistics (mm):")
    print(f"  Mean: {spacings.mean(axis=0)}")
    print(f"  Std: {spacings.std(axis=0)}")
    print(f"  Min: {spacings.min(axis=0)}")
    print(f"  Max: {spacings.max(axis=0)}")

    print("\nIntensity Statistics:")
    mean_vals = [x['mean'] for x in intensities]
    std_vals = [x['std'] for x in intensities]
    min_vals = [x['min'] for x in intensities]
    max_vals = [x['max'] for x in intensities]

    print(f"  Mean intensity: {np.mean(mean_vals):.2f} ± {np.std(mean_vals):.2f}")
    print(f"  Std intensity: {np.mean(std_vals):.2f} ± {np.std(std_vals):.2f}")
    print(f"  Min intensity: {np.mean(min_vals):.2f} ± {np.std(min_vals):.2f}")
    print(f"  Max intensity: {np.mean(max_vals):.2f} ± {np.std(max_vals):.2f}")

    return {
        'shapes': shapes,
        'spacings': spacings,
        'intensities': intensities
    }


def analyze_label_properties(data_root: str, dataset_name: str):
    """Analyze label statistics"""
    logger.info(f"\n{'=' * 60}")
    logger.info("Analyzing Label Properties")
    logger.info(f"{'=' * 60}")

    dataset_path = Path(data_root) / dataset_name

    structure_volumes = defaultdict(list)
    structure_counts = defaultdict(int)

    for subject_dir in sorted(dataset_path.iterdir()):
        if not subject_dir.is_dir():
            continue

        files = get_subject_files(subject_dir)

        for label_file in files['labels']:
            info = parse_filename(label_file.name)
            struct_type = info['type']

            try:
                data, img = load_nifti(str(label_file))

                # Calculate volume
                affine = img.affine
                spacing = np.abs(np.diag(affine)[:3])
                voxel_volume = np.prod(spacing)

                num_voxels = np.sum(data > 0)
                volume_cc = (num_voxels * voxel_volume) / 1000  # Convert to cc

                if volume_cc > 0:  # Only count non-empty labels
                    structure_volumes[struct_type].append(volume_cc)
                    structure_counts[struct_type] += 1

            except Exception as e:
                logger.warning(f"Error loading {label_file}: {e}")
                continue

    # Print statistics
    print("\nStructure Frequency:")
    for struct, count in sorted(structure_counts.items()):
        print(f"  {struct}: {count} annotations")

    print("\nStructure Volume Statistics (cc):")
    for struct, volumes in sorted(structure_volumes.items()):
        volumes = np.array(volumes)
        print(f"\n  {struct}:")
        print(f"    Mean: {volumes.mean():.2f} ± {volumes.std():.2f}")
        print(f"    Median: {np.median(volumes):.2f}")
        print(f"    Min: {volumes.min():.2f}")
        print(f"    Max: {volumes.max():.2f}")
        print(f"    25th percentile: {np.percentile(volumes, 25):.2f}")
        print(f"    75th percentile: {np.percentile(volumes, 75):.2f}")

    return structure_volumes


def analyze_missing_data(data_root: str, dataset_name: str):
    """Analyze missing sequences and labels"""
    logger.info(f"\n{'=' * 60}")
    logger.info("Analyzing Missing Data")
    logger.info(f"{'=' * 60}")

    dataset_path = Path(data_root) / dataset_name

    missing_sequences = defaultdict(int)
    missing_labels = defaultdict(int)
    total_subjects = 0

    for subject_dir in sorted(dataset_path.iterdir()):
        if not subject_dir.is_dir():
            continue

        total_subjects += 1
        files = get_subject_files(subject_dir)

        # Check for each sequence type
        for seq in ['T1', 'T1FS', 'T2', 'T2FS']:
            has_seq = any(seq in f.name for f in files['images'])
            if not has_seq:
                missing_sequences[seq] += 1

        # Check for each structure type
        for struct in ['ut', 'ov', 'em', 'cy', 'cds']:
            has_label = any(struct in f.name for f in files['labels'])
            if not has_label:
                missing_labels[struct] += 1

    print(f"\nTotal subjects analyzed: {total_subjects}")

    print("\nMissing Sequences:")
    for seq, count in sorted(missing_sequences.items()):
        pct = (count / total_subjects) * 100
        print(f"  {seq}: {count} subjects ({pct:.1f}%)")

    print("\nMissing Labels:")
    for struct, count in sorted(missing_labels.items()):
        pct = (count / total_subjects) * 100
        print(f"  {struct}: {count} subjects ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Explore UT-EndoMRI dataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/raw/UT-EndoMRI",
        help="Root directory of UT-EndoMRI dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["D1_MHS", "D2_TCPW", "both"],
        help="Dataset to analyze"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/eda_results",
        help="Directory to save analysis results"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    if args.dataset == "both":
        datasets = ["D1_MHS", "D2_TCPW"]
    else:
        datasets = [args.dataset]

    results = {}

    for dataset in datasets:
        print(f"\n\n{'#' * 70}")
        print(f"# Analyzing {dataset}")
        print(f"{'#' * 70}")

        # Run analyses
        overview = analyze_dataset_overview(args.data_root, dataset)
        image_props = analyze_image_properties(args.data_root, dataset)
        label_props = analyze_label_properties(args.data_root, dataset)
        analyze_missing_data(args.data_root, dataset)

        results[dataset] = {
            'overview': overview,
            'image_properties': {
                'shapes': image_props['shapes'].tolist(),
                'spacings': image_props['spacings'].tolist()
            },
            'label_volumes': {
                k: v for k, v in label_props.items()
            }
        }

    # Save results
    output_file = output_dir / "dataset_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n\nAnalysis results saved to: {output_file}")
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()