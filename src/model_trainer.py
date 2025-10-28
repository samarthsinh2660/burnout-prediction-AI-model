"""
Model Trainer Module
Unified interface for all model training functionality
Provides backward compatibility by importing all classes
"""

# Import custom loss functions
from .loss_functions import focal_loss_objective, weighted_logloss_objective

# Import single model trainer
from .single_model_trainer import BurnoutPredictor

# Import ensemble trainer
from .ensemble_trainer import HybridEnsemble

# Export all classes for backward compatibility
__all__ = [
    'focal_loss_objective',
    'weighted_logloss_objective',
    'BurnoutPredictor',
    'HybridEnsemble'
]
