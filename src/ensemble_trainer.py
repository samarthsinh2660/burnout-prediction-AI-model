"""
Ensemble Trainer
Contains the HybridEnsemble class for training multi-model ensembles
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix)
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import joblib
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config.config import SMOTE_PARAMS


class HybridEnsemble:
    """
    Hybrid Ensemble Model combining XGBoost, LightGBM, and CatBoost
    Uses weighted voting for final predictions
    """

    def __init__(self):
        self.models = {}
        self.ensemble = None
        self.feature_names = None
        self.is_trained = False
        self.training_history = {
            'xgboost': {'train_loss': [], 'val_loss': []},
            'lightgbm': {'train_loss': [], 'val_loss': []},
            'catboost': {'train_loss': [], 'val_loss': []}
        }
        self.model_weights = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all three models: XGBoost, LightGBM, CatBoost
        """
        print("\n" + "="*60)
        print("TRAINING HYBRID ENSEMBLE (3 MODELS)")
        print("="*60)

        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()

        # Handle class imbalance with SMOTE
        print("\nBalancing classes with SMOTE...")
        smote = SMOTE(**SMOTE_PARAMS)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        print(f"  Original samples: {len(y_train)}")
        print(f"  After SMOTE: {len(y_train_balanced)}")
        print(f"  Class distribution after SMOTE:")
        print(f"    No Burnout (0): {sum(y_train_balanced == 0)}")
        print(f"    Burnout (1): {sum(y_train_balanced == 1)}")

        # Calculate class weight
        scale_pos_weight = len(y_train_balanced[y_train_balanced == 0]) / len(y_train_balanced[y_train_balanced == 1])

        # Prepare evaluation set
        eval_set_train = [(X_train_balanced, y_train_balanced)]
        if X_val is not None and y_val is not None:
            eval_set_train.append((X_val, y_val))
            print(f"\n  Validation set: {len(X_val)} samples")

        # ========================================
        # MODEL 1: XGBoost (Conservative - Prevent Overfitting)
        # ========================================
        print("\n[1/3] Training XGBoost (Conservative Strategy)...")
        print("  Strategy: Shallower trees, more iterations, higher regularization")
        xgb_params = {
            'n_estimators': 400,             # More trees
            'learning_rate': 0.03,           # Slower learning
            'max_depth': 6,                  # Shallower trees
            'min_child_weight': 5,           # More conservative splits
            'subsample': 0.75,               # Lower subsample ratio
            'colsample_bytree': 0.75,
            'reg_alpha': 0.5,                # Higher L1 regularization
            'reg_lambda': 1.5,               # Higher L2 regularization
            'scale_pos_weight': scale_pos_weight * 1.5,
            'random_state': 42,
            'eval_metric': 'logloss'
        }

        self.models['xgboost'] = XGBClassifier(**xgb_params)
        self.models['xgboost'].fit(
            X_train_balanced, y_train_balanced,
            eval_set=eval_set_train,
            verbose=False
        )

        # Store XGBoost history
        if X_val is not None:
            results = self.models['xgboost'].evals_result()
            if 'validation_0' in results:
                self.training_history['xgboost']['train_loss'] = results['validation_0'].get('logloss', [])
            if 'validation_1' in results:
                self.training_history['xgboost']['val_loss'] = results['validation_1'].get('logloss', [])

        print("  ✓ XGBoost training complete")

        # ========================================
        # MODEL 2: LightGBM (Aggressive - Capture Complex Patterns)
        # ========================================
        print("\n[2/3] Training LightGBM (Aggressive Strategy)...")
        print("  Strategy: Deeper trees, more leaves, capture complex interactions")
        lgb_params = {
            'n_estimators': 300,
            'learning_rate': 0.05,           # Normal learning rate
            'max_depth': 9,                  # Deeper trees
            'num_leaves': 127,               # More leaves (complex patterns)
            'min_child_samples': 10,         # Less conservative
            'subsample': 0.85,               # Higher subsample
            'colsample_bytree': 0.85,
            'reg_alpha': 0.1,                # Lower L1 regularization
            'reg_lambda': 0.5,               # Lower L2 regularization
            'scale_pos_weight': scale_pos_weight * 1.5,
            'random_state': 43,              # Different seed
            'verbosity': -1,
            'min_split_gain': 0.0,           # Allow more splits
            'min_child_weight': 0.001
        }

        self.models['lightgbm'] = LGBMClassifier(**lgb_params)
        self.models['lightgbm'].fit(
            X_train_balanced, y_train_balanced,
            eval_set=eval_set_train
        )

        # Store LightGBM history
        if X_val is not None:
            results = self.models['lightgbm'].evals_result_
            if 'training' in results:
                self.training_history['lightgbm']['train_loss'] = results['training'].get('binary_logloss', [])
            if 'valid_1' in results:
                self.training_history['lightgbm']['val_loss'] = results['valid_1'].get('binary_logloss', [])

        print("  ✓ LightGBM training complete")

        # ========================================
        # MODEL 3: CatBoost (Balanced - Different Algorithm with Randomness)
        # ========================================
        print("\n[3/3] Training CatBoost (Balanced Strategy)...")
        print("  Strategy: Medium depth, strong regularization, bootstrap randomness")
        cat_params = {
            'iterations': 350,               # More iterations
            'learning_rate': 0.04,           # Slightly slower
            'depth': 7,                      # Balanced depth
            'l2_leaf_reg': 5,                # Strong L2 regularization
            'bagging_temperature': 0.8,      # Add randomness via bagging
            'random_strength': 0.5,          # More randomness in split selection
            'border_count': 128,             # More feature borders
            'scale_pos_weight': scale_pos_weight * 1.5,
            'random_state': 44,              # Different seed
            'verbose': False,
            'bootstrap_type': 'Bayesian',     # Different bootstrap method
            'train_dir': None                 # Prevent creating info folder
        }

        self.models['catboost'] = CatBoostClassifier(**cat_params)
        self.models['catboost'].fit(
            X_train_balanced, y_train_balanced,
            eval_set=(X_val, y_val) if X_val is not None else None,
            verbose=False
        )

        print("  ✓ CatBoost training complete")

        # ========================================
        # CREATE WEIGHTED SOFT VOTING ENSEMBLE
        # ========================================
        print("\n[4/4] Creating Weighted Soft Voting Ensemble...")
        print("  Using soft voting: combining probability predictions from all models")

        # Calculate weights based on validation performance
        if X_val is not None and y_val is not None:
            val_scores = {}
            val_f1_scores = {}

            for name, model in self.models.items():
                val_pred = model.predict(X_val)
                val_scores[name] = accuracy_score(y_val, val_pred)
                val_f1 = precision_recall_fscore_support(y_val, val_pred, average='binary')[2]
                val_f1_scores[name] = val_f1
                print(f"  {name.upper():12} - Validation Accuracy: {val_scores[name]*100:.2f}%, F1: {val_f1:.4f}")

            # Calculate weights based on both accuracy and F1 (F1 is more important for imbalanced data)
            combined_scores = {name: (val_scores[name] * 0.4 + val_f1_scores[name] * 0.6)
                             for name in val_scores.keys()}
            total_score = sum(combined_scores.values())
            self.model_weights = {name: score/total_score for name, score in combined_scores.items()}

            print(f"\n  Soft Voting Weights (based on Accuracy + F1):")
            for name, weight in self.model_weights.items():
                print(f"    {name.upper():12}: {weight:.4f}")

            # Show diversity metrics
            acc_diff = max(val_scores.values()) - min(val_scores.values())
            print(f"\n  Model Diversity: {acc_diff*100:.2f}% accuracy spread")
            if acc_diff > 0.003:
                print(f"  ✓ Good diversity! Models make different predictions")
            else:
                print(f"  ⚠ Low diversity - models are similar")
        else:
            # Equal weights if no validation set
            self.model_weights = {'xgboost': 0.33, 'lightgbm': 0.33, 'catboost': 0.34}

        # Create soft voting ensemble
        self.ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft',  # Use soft voting (probability-based)
            weights=list(self.model_weights.values())
        )

        # Fit ensemble (quick since models already trained)
        self.ensemble.fit(X_train_balanced, y_train_balanced)

        self.is_trained = True

        print("\n✓ Hybrid Ensemble training complete!")
        print("="*60)

        return self.ensemble

    def predict(self, X):
        """Get ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble is not trained yet!")
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        """Get ensemble prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Ensemble is not trained yet!")
        return self.ensemble.predict_proba(X)

    def predict_individual(self, X):
        """Get predictions from each individual model"""
        if not self.is_trained:
            raise ValueError("Ensemble is not trained yet!")

        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        return predictions

    def check_accuracy(self, X_test, y_test):
        """Evaluate ensemble and individual models"""
        print("\n" + "="*60)
        print("ENSEMBLE EVALUATION")
        print("="*60)

        # Ensemble predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n🎯 ENSEMBLE PERFORMANCE:")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"  [[TN={cm[0][0]}, FP={cm[0][1]}]")
        print(f"   [FN={cm[1][0]}, TP={cm[1][1]}]]")

        # Individual model performance
        print(f"\n📊 INDIVIDUAL MODEL PERFORMANCE:")
        individual_preds = self.predict_individual(X_test)
        individual_results = {}

        for name, pred in individual_preds.items():
            acc = accuracy_score(y_test, pred)
            f1_ind = precision_recall_fscore_support(y_test, pred, average='binary')[2]
            individual_results[name] = {'accuracy': acc, 'f1': f1_ind}
            print(f"  {name.upper():12} - Accuracy: {acc*100:.2f}%, F1: {f1_ind:.4f}")

        # Show best individual model
        best_individual = max(individual_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n  🏆 Best Individual Model: {best_individual[0].upper()} ({best_individual[1]['accuracy']*100:.2f}%)")

        print("="*60)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

    def plot_training_history(self, save_path='./models/ensemble_training_history.png'):
        """Plot training history for all 3 models"""
        if not self.training_history['xgboost']['train_loss']:
            print("No training history available")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = {'train': '#2E86AB', 'val': '#A23B72'}

        model_names = ['xgboost', 'lightgbm', 'catboost']
        titles = ['XGBoost', 'LightGBM', 'CatBoost']

        for idx, (model_name, title) in enumerate(zip(model_names, titles)):
            ax = axes[idx]
            history = self.training_history[model_name]
            has_labels = False

            if history['train_loss']:
                iterations = range(1, len(history['train_loss']) + 1)
                ax.plot(iterations, history['train_loss'],
                       label='Training Loss', linewidth=2, color=colors['train'], alpha=0.8)
                has_labels = True

            if history['val_loss']:
                iterations = range(1, len(history['val_loss']) + 1)
                ax.plot(iterations, history['val_loss'],
                       label='Validation Loss', linewidth=2, color=colors['val'], alpha=0.8)

                # Mark best iteration
                best_iter = np.argmin(history['val_loss']) + 1
                best_loss = min(history['val_loss'])
                ax.axvline(x=best_iter, color='green', linestyle='--',
                          linewidth=1.5, alpha=0.5)
                ax.scatter([best_iter], [best_loss], color='green', s=80,
                          zorder=5, marker='*')
                has_labels = True

            ax.set_xlabel('Iteration', fontsize=10, fontweight='bold')
            ax.set_ylabel('Log Loss', fontsize=10, fontweight='bold')
            ax.set_title(f'{title} Learning Curve', fontsize=12, fontweight='bold')
            
            # Only add legend if there are labeled artists
            if has_labels:
                ax.legend(fontsize=9)
            
            ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nEnsemble training history saved: {save_path}")
        plt.close()

    def save_model(self, filename='./models/hybrid_ensemble.pkl'):
        """Save ensemble and individual models"""
        if not self.is_trained:
            raise ValueError("Ensemble is not trained yet!")

        # Save individual models separately
        individual_models = {}
        for name, model in self.models.items():
            model_filename = f'./models/{name}_model.pkl'
            joblib.dump(model, model_filename)
            individual_models[name] = model_filename
            print(f"✓ {name.upper()} model saved: {model_filename}")

        # Save ensemble
        model_data = {
            'models': self.models,
            'ensemble': self.ensemble,
            'feature_names': self.feature_names,
            'model_weights': self.model_weights,
            'individual_files': individual_models
        }
        joblib.dump(model_data, filename)
        print(f"✓ Hybrid Ensemble saved: {filename}")

        return individual_models

    def load_model(self, filename='./models/hybrid_ensemble.pkl'):
        """Load ensemble model"""
        model_data = joblib.load(filename)
        self.models = model_data['models']
        self.ensemble = model_data['ensemble']
        self.feature_names = model_data['feature_names']
        self.model_weights = model_data['model_weights']
        self.is_trained = True
        print(f"\n✓ Hybrid Ensemble loaded from '{filename}'")
