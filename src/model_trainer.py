"""
Model Trainer Module
Handles model training, prediction, and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, precision_recall_fscore_support)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import sys
from pathlib import Path

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
        
    def train(self, X_train, y_train):
        """
        TRAIN THE MODEL
        This is the main training method - call this to train your model!
        
        Args:
            X_train: Training features (numpy array or pandas DataFrame)
            y_train: Training labels (0 or 1)
        
        Returns:
            Trained model
        """
        print("\n" + "="*60)
        print("TRAINING MODEL")
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
        
        # Train XGBoost
        print("\nTraining XGBoost classifier...")
        model_config = MODEL_PARAMS.copy()
        model_config['scale_pos_weight'] = scale_pos_weight
        
        self.model = XGBClassifier(**model_config)
        
        self.model.fit(X_train_balanced, y_train_balanced)
        self.is_trained = True
        
        print("✓ Model training complete!")
        print("="*60)
        
        return self.model
    
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
