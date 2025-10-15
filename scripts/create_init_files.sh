#!/bin/bash

# Script to create all __init__.py files for the project

echo "Creating __init__.py files..."

# Main src/__init__.py
cat > src/__init__.py << 'EOF'
"""
Endometriosis Uncertainty Segmentation Package
"""
__version__ = "0.1.0"
EOF

# src/data/__init__.py
cat > src/data/__init__.py << 'EOF'
"""Data loading and preprocessing modules"""
from .dataset import EndoMRIDataset, EndoMRIMultiRaterDataset
from .preprocessing import MRIPreprocessor
from .dataloader import get_dataloaders, compute_class_weights
from .augmentation import get_train_transforms
from .utils import (
    load_nifti,
    save_nifti,
    get_dataset_statistics,
    create_data_splits,
    load_data_splits,
    EndoMRIDataInfo
)

__all__ = [
    'EndoMRIDataset',
    'EndoMRIMultiRaterDataset',
    'MRIPreprocessor',
    'get_dataloaders',
    'compute_class_weights',
    'get_train_transforms',
    'load_nifti',
    'save_nifti',
    'get_dataset_statistics',
    'create_data_splits',
    'load_data_splits',
    'EndoMRIDataInfo'
]
EOF

# src/models/__init__.py
cat > src/models/__init__.py << 'EOF'
"""Model architectures and utilities"""
from .metrics import (
    dice_coefficient,
    iou_score,
    SegmentationMetrics
)

__all__ = [
    'dice_coefficient',
    'iou_score',
    'SegmentationMetrics'
]
EOF

# src/training/__init__.py
cat > src/training/__init__.py << 'EOF'
"""Training utilities"""
# Will be populated in Phase 2
__all__ = []
EOF

# src/inference/__init__.py
cat > src/inference/__init__.py << 'EOF'
"""Inference utilities"""
# Will be populated in Phase 4
__all__ = []
EOF

# src/visualization/__init__.py
cat > src/visualization/__init__.py << 'EOF'
"""Visualization utilities"""
# Will be populated as needed
__all__ = []
EOF

# src/utils/__init__.py
cat > src/utils/__init__.py << 'EOF'
"""General utilities"""
from .logging_utils import (
    setup_logging,
    ExperimentLogger,
    MetricsTracker
)
from .checkpoint_utils import (
    CheckpointManager,
    save_model_only,
    load_model_weights
)
from .config_utils import (
    load_yaml_config,
    ConfigManager,
    TrainingConfig,
    ModelConfig
)

__all__ = [
    'setup_logging',
    'ExperimentLogger',
    'MetricsTracker',
    'CheckpointManager',
    'save_model_only',
    'load_model_weights',
    'load_yaml_config',
    'ConfigManager',
    'TrainingConfig',
    'ModelConfig'
]
EOF

# configs/__init__.py
touch configs/__init__.py

# tests/__init__.py
touch tests/__init__.py

echo "✓ All __init__.py files created successfully!"