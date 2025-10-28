"""
Single Model Trainer
Contains the BurnoutPredictor class for training individual XGBoost models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_fscore_support, log_loss)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config.config import MODEL_PARAMS, SMOTE_PARAMS


class BurnoutPredictor:
    """Unified burnout prediction model with prediction capabilities"""

    def __init__(self, approach='merged'):
        self.approach = approach
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        self.is_trained = False
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

    def train(self, X_train, y_train, X_val=None, y_val=None, use_custom_loss=True):
        """
        TRAIN THE MODEL with validation set and custom loss function

        Args:
            X_train: Training features (numpy array or pandas DataFrame)
            y_train: Training labels (0 or 1)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            use_custom_loss: Whether to use custom focal loss (default: True)

        Returns:
            Trained model
        """
        print("\n" + "="*60)
        print("TRAINING MODEL WITH VALIDATION")
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

        # Calculate class weights for better F1
        scale_pos_weight = len(y_train_balanced[y_train_balanced == 0]) / len(y_train_balanced[y_train_balanced == 1])
        print(f"  Scale positive weight: {scale_pos_weight:.2f}")

        # Prepare model configuration
        model_config = MODEL_PARAMS.copy()
        model_config['scale_pos_weight'] = scale_pos_weight

        # Note: Custom loss functions are complex with XGBoost
        # Using standard binary:logistic with class weights for better compatibility
        if use_custom_loss:
            print("\nUsing Weighted XGBoost with enhanced class balancing...")
            # Increase scale_pos_weight for better minority class handling
            model_config['scale_pos_weight'] = scale_pos_weight * 1.5
        else:
            print("\nUsing Standard XGBoost...")

        # Train XGBoost with validation
        print("\nTraining XGBoost classifier...")
        self.model = XGBClassifier(**model_config)

        # Prepare evaluation set
        eval_set = [(X_train_balanced, y_train_balanced)]

        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            print(f"  Validation set: {len(X_val)} samples")

        # Train with validation monitoring
        if X_val is not None:
            self.model.fit(
                X_train_balanced, y_train_balanced,
                eval_set=eval_set,
                verbose=False
            )

            # Store training history
            results = self.model.evals_result()
            if 'validation_0' in results:
                self.training_history['train_loss'] = results['validation_0'].get('logloss', [])
            if 'validation_1' in results:
                self.training_history['val_loss'] = results['validation_1'].get('logloss', [])

            print(f"\n  Training completed: {self.model.n_estimators} iterations")
            if self.training_history['val_loss']:
                final_val_loss = self.training_history['val_loss'][-1]
                print(f"  Final validation loss: {final_val_loss:.4f}")
        else:
            self.model.fit(X_train_balanced, y_train_balanced)

        self.is_trained = True

        print("✓ Model training complete!")
        print("="*60)

        return self.model

    def plot_training_history(self, save_path='./models/training_history.png'):
        """Plot training and validation loss curves"""
        if not self.training_history['train_loss']:
            print("No training history available")
            return

        plt.figure(figsize=(10, 6))

        # Plot loss curves
        iterations = range(1, len(self.training_history['train_loss']) + 1)
        plt.plot(iterations, self.training_history['train_loss'],
                label='Training Loss', linewidth=2, color='#2E86AB', alpha=0.8)
        has_labels = True

        if self.training_history['val_loss']:
            plt.plot(iterations, self.training_history['val_loss'],
                    label='Validation Loss', linewidth=2, color='#A23B72', alpha=0.8)

            # Mark the best iteration (lowest validation loss)
            best_iter = np.argmin(self.training_history['val_loss']) + 1
            best_loss = min(self.training_history['val_loss'])
            plt.axvline(x=best_iter, color='green', linestyle='--',
                       linewidth=1.5, alpha=0.6, label=f'Best Iteration ({best_iter})')
            plt.scatter([best_iter], [best_loss], color='green', s=100,
                       zorder=5, marker='*', edgecolors='darkgreen', linewidth=1.5)

        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('Log Loss', fontsize=12, fontweight='bold')
        plt.title('Training History - Model Learning Curve', fontsize=14, fontweight='bold', pad=15)
        
        # Only add legend if there are labeled artists
        if has_labels:
            plt.legend(loc='best', fontsize=10, framealpha=0.9)
        
        plt.grid(True, alpha=0.3, linestyle='--')

        # Add statistics text box
        if self.training_history['val_loss']:
            stats_text = f'Final Train Loss: {self.training_history["train_loss"][-1]:.4f}\n'
            stats_text += f'Final Val Loss: {self.training_history["val_loss"][-1]:.4f}\n'
            stats_text += f'Best Val Loss: {best_loss:.4f}'
            plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved: {save_path}")
        plt.close()

    def predict(self, X):
        """
        GET PREDICTIONS FROM THE MODEL
        Use this to predict burnout for new data

        Args:
            X: Features to predict (numpy array or pandas DataFrame)

        Returns:
            predictions: Array of predictions (0 = No Burnout, 1 = Burnout)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet! Call train() first.")

        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X):
        """
        GET PREDICTION PROBABILITIES
        Returns probability of burnout (0 to 1)

        Args:
            X: Features to predict

        Returns:
            probabilities: Array of probabilities for burnout
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet! Call train() first.")

        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of class 1 (burnout)
        return probabilities

    def predict_with_threshold(self, X, threshold=0.4):
        """Predict with custom threshold to improve F1"""
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)
        return predictions

    def predict_single(self, features_dict):
        """
        PREDICT FOR A SINGLE PERSON

        Args:
            features_dict: Dictionary with feature values
            Example: {
                'stress_level': 7.5,
                'workload': 8.0,
                'sleep_quality': 4.0,
                'screen_time': 10.0,
                ...
            }

        Returns:
            prediction: 0 (No Burnout) or 1 (Burnout)
            probability: Probability of burnout (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet! Call train() first.")

        # Create DataFrame from dictionary
        df = pd.DataFrame([features_dict])

        # Ensure all required features are present
        if self.feature_names:
            for feat in self.feature_names:
                if feat not in df.columns:
                    df[feat] = 0  # Default value for missing features
            df = df[self.feature_names]  # Ensure correct order

        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0, 1]

        return prediction, probability

    def check_accuracy(self, X_test, y_test):
        """
        CHECK MODEL ACCURACY
        This shows how well your model performs

        Args:
            X_test: Test features
            y_test: True labels for test data

        Returns:
            Dictionary with accuracy metrics
        """
        print("\n" + "="*60)
        print("CHECKING MODEL ACCURACY")
        print("="*60)

        # Get predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )

        # Display results
        print(f"\n📊 OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📈 ROC-AUC Score: {roc_auc:.4f}")
        print(f"🎯 Precision: {precision:.4f}")
        print(f"🔍 Recall: {recall:.4f}")
        print(f"⚖️  F1-Score: {f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n📋 CONFUSION MATRIX:")
        print("                  Predicted")
        print("                No Burnout  Burnout")
        print(f"Actual No Burnout    {cm[0,0]:6d}    {cm[0,1]:6d}")
        print(f"Actual Burnout       {cm[1,0]:6d}    {cm[1,1]:6d}")

        print(f"\n✓ True Negatives (Correctly predicted No Burnout): {cm[0,0]}")
        print(f"✗ False Positives (Incorrectly predicted Burnout): {cm[0,1]}")
        print(f"✗ False Negatives (Missed Burnout cases): {cm[1,0]}")
        print(f"✓ True Positives (Correctly predicted Burnout): {cm[1,1]}")

        # Detailed classification report
        print("\n📝 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred,
                                   target_names=['No Burnout', 'Burnout']))

        print("="*60)

        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

    def save_model(self, filename='burnout_model.pkl'):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filename)
        print(f"\n✓ Model saved to '{filename}'")

    def load_model(self, filename='burnout_model.pkl'):
        """Load trained model from file"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        print(f"\n✓ Model loaded from '{filename}'")
