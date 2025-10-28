"""
Custom Loss Functions
Contains focal loss and weighted log loss implementations for XGBoost
"""

import numpy as np


def focal_loss_objective(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Custom Focal Loss for handling class imbalance
    Focuses more on hard-to-classify examples

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)

    Returns:
        Gradient and hessian for XGBoost
    """
    # Convert predictions to probabilities
    p = 1.0 / (1.0 + np.exp(-y_pred))

    # Calculate focal loss components
    grad = np.where(y_true == 1,
                    alpha * (p - 1) * (1 - p) ** (gamma - 1) * (gamma * p * np.log(p) + p - 1),
                    -(1 - alpha) * p * (1 - p) ** (gamma - 1) * (gamma * (1 - p) * np.log(1 - p) + p))

    hess = np.where(y_true == 1,
                    alpha * (1 - p) ** (gamma - 2) * (p * (gamma * (2 * p - 1) - p) + (1 - p) ** 2),
                    (1 - alpha) * p * (1 - p) ** (gamma - 2) * ((1 - p) * (gamma * (1 - 2 * p) + p) + p ** 2))

    return grad, hess


def weighted_logloss_objective(y_true, y_pred, pos_weight=2.0):
    """
    Weighted Log Loss - gives more importance to minority class (burnout)

    Args:
        y_true: True labels
        y_pred: Predicted logits
        pos_weight: Weight for positive class

    Returns:
        Gradient and hessian for XGBoost
    """
    # Convert to probabilities
    p = 1.0 / (1.0 + np.exp(-y_pred))

    # Calculate weighted gradients
    grad = np.where(y_true == 1,
                    pos_weight * (p - 1),
                    p)

    hess = np.where(y_true == 1,
                    pos_weight * p * (1 - p),
                    p * (1 - p))

    return grad, hess
