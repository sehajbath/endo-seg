# ============================================
# src/__init__.py
# ============================================
"""
Endometriosis Uncertainty Segmentation Package
"""
__version__ = "0.1.0"

# ============================================
# src/data/__init__.py
# ============================================
"""Data loading and preprocessing modules"""
from .data.dataset import EndoMRIDataset, EndoMRIMultiRaterDataset
from .data.preprocessing import MRIPreprocessor
from .data.dataloader import get_dataloaders, compute_class_weights
from .data.augmentation import (
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomElasticDeformation,
    RandomGamma,
    RandomGaussianNoise,
    get_train_transforms
)
from .data.utils import (
    load_nifti,
    save_nifti,
    parse_filename,
    get_subject_files,
    get_dataset_statistics,
    create_data_splits,
    load_data_splits,
    EndoMRIDataInfo,
    merge_structure_labels,
    canonicalize_structure_list
)

__all__ = [
    'EndoMRIDataset',
    'EndoMRIMultiRaterDataset',
    'MRIPreprocessor',
    'get_dataloaders',
    'compute_class_weights',
    'RandomFlip',
    'RandomRotation',
    'RandomTranslation',
    'RandomElasticDeformation',
    'RandomGamma',
    'RandomGaussianNoise',
    'get_train_transforms',
    'load_nifti',
    'save_nifti',
    'parse_filename',
    'get_subject_files',
    'merge_structure_labels',
    'canonicalize_structure_list',
    'get_dataset_statistics',
    'create_data_splits',
    'load_data_splits',
    'EndoMRIDataInfo'
]

# ============================================
# src/models/__init__.py
# ============================================
"""Model architectures and utilities"""
from .models.metrics import (
    dice_coefficient,
    iou_score,
    hausdorff_distance_95,
    surface_dice,
    SegmentationMetrics,
    compute_confusion_matrix
)

__all__ = [
    'dice_coefficient',
    'iou_score',
    'hausdorff_distance_95',
    'surface_dice',
    'SegmentationMetrics',
    'compute_confusion_matrix'
]

# ============================================
# src/training/__init__.py
# ============================================
"""Training utilities"""
# Will be populated in Phase 2

__all__ = []

# ============================================
# src/inference/__init__.py
# ============================================
"""Inference utilities"""
# Will be populated in Phase 4

__all__ = []

# ============================================
# src/visualization/__init__.py
# ============================================
"""Visualization utilities"""
# Will be populated as needed

__all__ = []

# ============================================
# src/utils/__init__.py
# ============================================
"""General utilities"""
from .utils.logging_utils import (
    setup_logging,
    ExperimentLogger,
    MetricsTracker,
    log_system_info
)
from .utils.checkpoint_utils import (
    CheckpointManager,
    save_model_only,
    load_model_weights
)
from .utils.config_utils import (
    load_yaml_config,
    load_json_config,
    merge_configs,
    save_config,
    load_all_configs,
    ConfigManager,
    TrainingConfig,
    ModelConfig,
    validate_config
)

__all__ = [
    'setup_logging',
    'ExperimentLogger',
    'MetricsTracker',
    'log_system_info',
    'CheckpointManager',
    'save_model_only',
    'load_model_weights',
    'load_yaml_config',
    'load_json_config',
    'merge_configs',
    'save_config',
    'load_all_configs',
    'ConfigManager',
    'TrainingConfig',
    'ModelConfig',
    'validate_config'
]
