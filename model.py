"""
Burnout Prediction Model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
import warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class BurnoutDatasetMerger:
    """Handles merging of multiple burnout datasets with feature alignment"""
    
    def __init__(self):
        self.feature_maps = {
            'employee_burnout': {
                'stress_level': 'Mental Fatigue Score',
                'workload': 'Resource Allocation',
                'work_setup': 'WFH Setup Available',
                'company_type': 'Company Type',
                'target_col': 'Burn Rate',
                'target_transform': lambda x: (x > 0.5).astype(int)
            },
            'hackerearth_burnout': {
                'stress_level': 'Mental Fatigue Score',
                'workload': 'Resource Allocation',
                'work_setup': 'WFH Setup Available',
                'target_col': 'Burn Rate',
                'target_transform': lambda x: (x > 0.5).astype(int)
            },
            'workplace_mental': {
                'stress_level': 'StressLevel',
                'work_setup': 'RemoteWork',
                'support': 'HasMentalHealthSupport',
                'target_col': 'BurnoutRisk',
                'target_transform': lambda x: x.astype(int)
            },
            'student_resilience': {
                'stress_level': 'Anxiety_Score',
                'workload': 'Stress_Level',
                'sleep_quality': 'Sleep_Hours',
                'support': 'Sentiment_Score',
                'target_col': 'Mental_Health_Status',
                'target_transform': lambda x: (x >= 2).astype(int)
            },
            'student_stress': {
                'stress_level': 'anxiety_level',
                'workload': 'study_load',
                'sleep_quality': 'sleep_quality',
                'support': 'social_support',
                'target_col': 'stress_level',
                'target_transform': lambda x: (x >= 1).astype(int)
            },
            'tech_usage': {
                'stress_level': 'stress_level',
                'screen_time': 'screen_time_hours',
                'social_media': 'social_media_platforms_used',
                'sleep_quality': 'sleep_hours',
                'target_col': 'mood_score',
                'target_transform': lambda x: (x < 5).astype(int)
            },
            'screen_time': {
                'stress_level': 'Stress_Level',
                'screen_time': 'Screen_Time_Hours',
                'social_media': 'Social_Media_Usage_Hours',
                'sleep_quality': 'Sleep_Hours',
                'target_col': 'Mental_Health_Status',
                'target_transform': lambda x: x.isin(['Poor', 'Fair']).astype(int)
            },
            'healthcare_mental': {
                'stress_level': 'Stress Level',
                'support': 'Access to EAPs',
                'target_col': 'Burnout Frequency',
                'target_transform': lambda x: (x == 'Often').astype(int)
            },
            'additional_train': {
                'stress_level': 'Mental Fatigue Score',
                'workload': 'Resource Allocation',
                'work_setup': 'WFH Setup Available_Yes',
                'company_type': 'Company Type_Service',
                'target_col': 'Burn Rate',
                'target_transform': lambda x: (x > 0.5).astype(int)
            }
        }
        
        self.unified_datasets = {}
        self.combined_data = None
        
    def load_datasets(self, file_paths):
        """Load all datasets from file paths dictionary"""
        datasets = {}
        print("Loading datasets...")
        for name, path in file_paths.items():
            try:
                if name == 'additional_train':
                    # Special handling for additional processed data
                    X_path = './dataset/X_train_processed.csv'
                    y_path = './dataset/y_train_processed.csv'
                    X_df = pd.read_csv(X_path)
                    y_df = pd.read_csv(y_path)
                    df = X_df.copy()
                    df['Burn Rate'] = y_df['Burn Rate']
                    datasets[name] = df
                    print(f"✓ Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns (merged from X and y)")
                else:
                    df = pd.read_csv(path)
                    datasets[name] = df
                    print(f"✓ Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"✗ Error loading {name}: {e}")
        return datasets
    
    def create_unified_features(self, df, dataset_name):
        """Map dataset-specific features to common schema"""
        unified_df = pd.DataFrame()
        
        if dataset_name not in self.feature_maps:
            print(f"Warning: No mapping for {dataset_name}")
            return None
        
        feature_map = self.feature_maps[dataset_name]
        
        # Map common features
        common_features = ['stress_level', 'workload', 'sleep_quality', 
                          'support', 'screen_time', 'social_media', 'work_setup']
        
        for feat in common_features:
            if feat in feature_map and feature_map[feat] in df.columns:
                unified_df[feat] = df[feature_map[feat]]
            else:
                unified_df[feat] = np.nan
        
        # Handle target variable
        if 'target_col' in feature_map and feature_map['target_col'] in df.columns:
            target_transform = feature_map.get('target_transform', lambda x: x)
            unified_df['burnout_target'] = target_transform(df[feature_map['target_col']])
        
        # Add dataset source as feature
        unified_df['dataset_source'] = dataset_name
        
        # Convert categorical to numeric
        if 'work_setup' in unified_df.columns:
            unified_df['work_setup'] = unified_df['work_setup'].map({'Yes': 1, 'No': 0, 'Hybrid': 0.5}).fillna(0)
        
        if 'support' in unified_df.columns and unified_df['support'].dtype == 'object':
            unified_df['support'] = unified_df['support'].map({'Yes': 1, 'No': 0}).fillna(0)
        
        if dataset_name == 'screen_time' and 'stress_level' in unified_df.columns:
            unified_df['stress_level'] = unified_df['stress_level'].map({'Low': 3, 'Medium': 6, 'High': 9}).fillna(6)
        
        return unified_df
    
    def merge_datasets(self, datasets):
        """Merge all datasets into unified format"""
        print("\nCreating unified feature schema...")
        unified_list = []
        
        for name, df in datasets.items():
            unified = self.create_unified_features(df, name)
            if unified is not None:
                self.unified_datasets[name] = unified
                unified_list.append(unified)
                print(f"✓ Unified {name}: {unified.shape[0]} samples")
        
        # Concatenate all datasets
        print("\nMerging datasets...")
        self.combined_data = pd.concat(unified_list, ignore_index=True)
        
        print(f"\n✓ Combined dataset created: {self.combined_data.shape[0]} total samples")
        print(f"  Features: {self.combined_data.shape[1]}")
        print(f"  Missing values:\n{self.combined_data.isnull().sum()}")
        
        return self.combined_data
    
    def preprocess_combined_data(self):
        """Preprocess the combined dataset"""
        print("\nPreprocessing combined data...")
        
        # Fill missing values with median for numerical features
        numeric_cols = self.combined_data.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop('burnout_target', errors='ignore')
        
        for col in numeric_cols:
            if self.combined_data[col].isnull().any():
                median_val = self.combined_data[col].median()
                self.combined_data[col].fillna(median_val, inplace=True)
        
        # Feature engineering
        print("Creating interaction features...")
        
        # Stress-sleep interaction
        if 'stress_level' in self.combined_data.columns and 'sleep_quality' in self.combined_data.columns:
            self.combined_data['stress_sleep_ratio'] = (
                self.combined_data['stress_level'] / (self.combined_data['sleep_quality'] + 1)
            )
        
        # Digital overload
        if 'screen_time' in self.combined_data.columns and 'social_media' in self.combined_data.columns:
            self.combined_data['digital_overload'] = (
                self.combined_data['screen_time'] * self.combined_data['social_media']
            ).fillna(0)
        
        # Workload stress
        if 'workload' in self.combined_data.columns and 'stress_level' in self.combined_data.columns:
            self.combined_data['workload_stress'] = (
                self.combined_data['workload'] * self.combined_data['stress_level']
            ).fillna(0)
        
        # One-hot encode dataset source
        self.combined_data = pd.get_dummies(
            self.combined_data, 
            columns=['dataset_source'], 
            drop_first=False
        )
        
        print("✓ Preprocessing complete")
        print(f"  Final shape: {self.combined_data.shape}")
        
        return self.combined_data

    def handle_missing_values(self):
        """
        IMPROVED: Fill missing values instead of dropping columns
        Creates missing indicators and uses smart filling strategies
        """
        print("\n" + "="*70)
        print("STEP 1: HANDLING MISSING VALUES (IMPROVED)")
        print("="*70)
        
        print("\n📊 Missing Values Before Cleaning:")
        missing_summary = self.combined_data.isnull().sum()
        missing_pct = (missing_summary / len(self.combined_data)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_summary.index,
            'Missing_Count': missing_summary.values,
            'Missing_Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        )
        
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
        
        # ✨ NEW STRATEGY: DON'T DROP - FILL SMARTLY!
        print("\n💡 Strategy: Filling ALL columns (not dropping any)")
        
        # For numeric columns with high missing rates
        numeric_cols = self.combined_data.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop('burnout_target', errors='ignore')
        
        for col in numeric_cols:
            if self.combined_data[col].isnull().any():
                missing_pct = (self.combined_data[col].isnull().sum() / len(self.combined_data)) * 100
                
                if missing_pct > 50:
                    # High missing rate: Fill with median AND create missing indicator
                    print(f"\n  📍 {col} ({missing_pct:.1f}% missing):")
                    
                    # Create binary indicator for "was missing"
                    indicator_col = f'{col}_was_missing'
                    self.combined_data[indicator_col] = self.combined_data[col].isnull().astype(int)
                    print(f"     ✓ Created indicator: {indicator_col}")
                    
                    # Fill with median grouped by burnout status (smarter!)
                    if 'burnout_target' in self.combined_data.columns:
                        self.combined_data[col] = self.combined_data.groupby('burnout_target')[col].transform(
                            lambda x: x.fillna(x.median())
                        )
                        print(f"     ✓ Filled with group-specific median")
                    else:
                        median_val = self.combined_data[col].median()
                        self.combined_data[col].fillna(median_val, inplace=True)
                        print(f"     ✓ Filled with overall median: {median_val:.2f}")
                else:
                    # Low missing rate: Simple median fill
                    median_val = self.combined_data[col].median()
                    missing_count = self.combined_data[col].isnull().sum()
                    self.combined_data[col].fillna(median_val, inplace=True)
                    print(f"  ✓ {col}: filled {missing_count} values with {median_val:.2f}")
        
        # Categorical columns
        categorical_cols = self.combined_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.combined_data[col].isnull().any():
                mode_val = self.combined_data[col].mode()[0] if len(self.combined_data[col].mode()) > 0 else 'Unknown'
                missing_count = self.combined_data[col].isnull().sum()
                self.combined_data[col].fillna(mode_val, inplace=True)
                print(f"  ✓ {col}: filled {missing_count} values with '{mode_val}'")
        
        # Target variable
        print("\n🎯 Handling target variable...")
        if 'burnout_target' in self.combined_data.columns:
            missing_target = self.combined_data['burnout_target'].isnull().sum()
            if missing_target > 0:
                print(f"  ⚠️  Removing {missing_target} rows with missing target")
                self.combined_data.dropna(subset=['burnout_target'], inplace=True)
            else:
                print("  ✓ No missing target values")
        
        # Final check
        print("\n✅ Missing Values After Cleaning:")
        final_missing = self.combined_data.isnull().sum().sum()
        print(f"  Total missing values: {final_missing}")
        
        if final_missing == 0:
            print("  ✓ All missing values handled!")
        
        print(f"\n📊 Final dataset shape: {self.combined_data.shape}")
        print(f"📊 Features gained from missing indicators: {len([c for c in self.combined_data.columns if '_was_missing' in c])}")
        print("="*70)
        
        return self.combined_data

    def analyze_correlations(self, save_plot=True):
        """
        STEP 2: Correlation analysis and visualization
        Call this AFTER handling missing values
        """
        print("\n" + "="*70)
        print("STEP 2: CORRELATION ANALYSIS")
        print("="*70)
        
        # Select only numeric columns
        numeric_data = self.combined_data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            print("⚠️  Not enough numeric columns for correlation analysis")
            return None
        
        print(f"\n📊 Analyzing {numeric_data.shape[1]} numeric features...")
        
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        
        # Find correlations with target variable
        if 'burnout_target' in correlation_matrix.columns:
            print("\n🎯 TOP CORRELATIONS WITH BURNOUT:")
            target_corr = correlation_matrix['burnout_target'].sort_values(ascending=False)
            target_corr = target_corr[target_corr.index != 'burnout_target']
            
            print("\nPositive correlations (increase burnout risk):")
            positive = target_corr[target_corr > 0].head(10)
            for feat, corr in positive.items():
                print(f"  {feat:30s}: {corr:+.4f}")
            
            print("\nNegative correlations (decrease burnout risk):")
            negative = target_corr[target_corr < 0].head(10)
            for feat, corr in negative.items():
                print(f"  {feat:30s}: {corr:+.4f}")
        
        # Find highly correlated feature pairs (multicollinearity)
        print("\n⚠️  HIGHLY CORRELATED FEATURES (potential multicollinearity):")
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append({
                        'Feature 1': correlation_matrix.columns[i],
                        'Feature 2': correlation_matrix.columns[j],
                        'Correlation': correlation_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            for pair in high_corr_pairs:
                print(f"  {pair['Feature 1']} <-> {pair['Feature 2']}: {pair['Correlation']:.4f}")
            print(f"\n  💡 Consider removing one feature from each pair")
        else:
            print("  ✓ No highly correlated feature pairs found")
        
        # Create correlation heatmap
        print("\n📈 Creating correlation heatmap...")
        
        # Full correlation matrix
        plt.figure(figsize=(16, 14))
        
        # Mask for upper triangle (optional - makes it cleaner)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=False,  # Set to True if you want numbers on the heatmap
            cmap='RdBu_r',  # Red for positive, Blue for negative
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1
        )
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('correlation_matrix_full.png', dpi=300, bbox_inches='tight')
            print("  ✓ Saved: correlation_matrix_full.png")
        
        plt.show()
        
        # Create focused heatmap - only top features correlated with target
        if 'burnout_target' in correlation_matrix.columns:
            print("\n📈 Creating focused correlation heatmap (top features)...")
            
            # Get top 15 features most correlated with target
            target_corr = correlation_matrix['burnout_target'].abs().sort_values(ascending=False)
            top_features = target_corr.head(16).index.tolist()  # 15 + target itself
            
            # Create subset correlation matrix
            focused_corr = correlation_matrix.loc[top_features, top_features]
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                focused_corr,
                annot=True,  # Show correlation values
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1
            )
            
            plt.title('Top Features - Correlation with Burnout Target', 
                      fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            if save_plot:
                plt.savefig('correlation_matrix_focused.png', dpi=300, bbox_inches='tight')
                print("  ✓ Saved: correlation_matrix_focused.png")
            
            plt.show()
        
        print("\n" + "="*70)
        
        return correlation_matrix


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
        smote = SMOTE(random_state=42)
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
        self.model = XGBClassifier(
            learning_rate=0.05,
            max_depth=7,
            n_estimators=300,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,  # Add class weights
            random_state=42,
            eval_metric='logloss'
        )
        
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


def test_single_prediction(model):
    """
    EXAMPLE: How to predict for a single person
    """
    print("\n" + "="*60)
    print("TESTING SINGLE PREDICTION")
    print("="*60)
    
    # Example person 1: High burnout risk
    print("\n👤 Person 1: High Risk Profile")
    person1 = {
        'stress_level': 8.5,
        'workload': 9.0,
        'sleep_quality': 3.0,
        'support': 2.0,
        'screen_time': 12.0,
        'social_media': 6.0,
        'work_setup': 0
    }
    
    # Add all other features as zeros (they'll be filled by the model)
    for col in model.feature_names:
        if col not in person1:
            person1[col] = 0
    
    pred1, prob1 = model.predict_single(person1)
    print(f"  Prediction: {'🔴 BURNOUT' if pred1 == 1 else '🟢 NO BURNOUT'}")
    print(f"  Burnout Probability: {prob1*100:.2f}%")
    
    # Example person 2: Low burnout risk
    print("\n👤 Person 2: Low Risk Profile")
    person2 = {
        'stress_level': 3.0,
        'workload': 4.0,
        'sleep_quality': 8.0,
        'support': 8.0,
        'screen_time': 4.0,
        'social_media': 2.0,
        'work_setup': 1
    }
    
    for col in model.feature_names:
        if col not in person2:
            person2[col] = 0
    
    pred2, prob2 = model.predict_single(person2)
    print(f"  Prediction: {'🔴 BURNOUT' if pred2 == 1 else '🟢 NO BURNOUT'}")
    print(f"  Burnout Probability: {prob2*100:.2f}%")


def main():
    """
    MAIN FUNCTION - UPDATED WITH CORRECT ORDER
    """
    print("="*70)
    print("BURNOUT PREDICTION MODEL - COMPLETE PIPELINE")
    print("="*70)
    
    # ========================
    # STEP 0: LOAD DATASETS
    # ========================
    print("\n🔵 STEP 0: LOADING DATASETS")
    file_paths = {
        'employee_burnout': './dataset/train.csv',
        'hackerearth_burnout': './dataset/train.csv',  # duplicate for now
        'workplace_mental': './dataset/mental_health_workplace_survey.csv',
        'student_resilience': './dataset/mental_health_dataset.csv',
        'student_stress': './dataset/StressLevelDataset.csv',
        'tech_usage': './dataset/digital_habits_vs_mental_health.csv',
        'screen_time': './dataset/mental_health_and_technology_usage_2024.csv',
        'healthcare_mental': './dataset/Healthcare Workforce Mental Health Dataset.csv',
        'additional_train': './dataset/X_train_processed.csv'  # Will be merged with y
    }
    
    merger = BurnoutDatasetMerger()
    datasets = merger.load_datasets(file_paths)
    
    if len(datasets) == 0:
        print("\n❌ Error: No datasets loaded. Please check file paths.")
        return None, None
    
    # ========================
    # MERGE DATASETS
    # ========================
    print("\n🔵 MERGING DATASETS")
    combined_data = merger.merge_datasets(datasets)
    
    # ========================
    # 🎯 STEP 1: HANDLE MISSING VALUES (YOUR FIRST STEP!)
    # ========================
    print("\n🔵 STEP 1: CLEANING DATA")
    combined_data = merger.handle_missing_values()
    
    # ========================
    # 🎯 STEP 2: CORRELATION ANALYSIS (YOUR SECOND STEP!)
    # ========================
    print("\n🔵 STEP 2: ANALYZING CORRELATIONS")
    correlation_matrix = merger.analyze_correlations(save_plot=True)
    
    # ========================
    # STEP 3: FEATURE ENGINEERING
    # ========================
    print("\n🔵 STEP 3: FEATURE ENGINEERING")
    combined_data = merger.preprocess_combined_data()
    
    # ========================
    # STEP 4: PREPARE DATA
    # ========================
    print("\n🔵 STEP 4: PREPARING DATA FOR TRAINING")
    
    combined_data = combined_data.dropna(subset=['burnout_target'])
    
    X = combined_data.drop('burnout_target', axis=1)
    y = combined_data['burnout_target']
    
    print(f"\n✓ Total samples: {len(X)}")
    print(f"✓ Number of features: {X.shape[1]}")
    print(f"✓ Burnout cases: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"✓ No burnout cases: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # ========================
    # STEP 5: NORMALIZATION (SCALING)
    # ========================
    print("\n🔵 STEP 5: NORMALIZING FEATURES")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Features scaled using StandardScaler")
    print(f"✓ Training set: {X_train_scaled.shape[0]} samples")
    print(f"✓ Test set: {X_test_scaled.shape[0]} samples")
    
    # ========================
    # STEP 6: TRAIN MODEL
    # ========================
    print("\n🔵 STEP 6: TRAINING MODEL")
    predictor = BurnoutPredictor()
    predictor.feature_names = X.columns.tolist()
    predictor.train(X_train_scaled, y_train)
    
    # ========================
    # STEP 7: EVALUATE
    # ========================
    print("\n🔵 STEP 7: EVALUATING MODEL")
    results = predictor.check_accuracy(X_test_scaled, y_test)
    
    # ========================
    # STEP 8: TEST PREDICTIONS
    # ========================
    print("\n🔵 STEP 8: TESTING PREDICTIONS")
    test_single_prediction(predictor)
    
    # ========================
    # STEP 9: SAVE MODEL
    # ========================
    print("\n🔵 STEP 9: SAVING MODEL")
    predictor.save_model('burnout_combined_model.pkl')
    
    # ========================
    # FINAL SUMMARY
    # ========================
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE - SUMMARY")
    print("="*70)
    print(f"\n📊 Model Performance:")
    print(f"  Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"  F1-Score:  {results['f1_score']:.4f}")
    print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
    
    print(f"\n📁 Files Created:")
    print(f"  ✓ correlation_matrix_full.png")
    print(f"  ✓ correlation_matrix_focused.png")
    print(f"  ✓ burnout_combined_model.pkl")
    
    print("\n" + "="*70)
    
    return predictor, results, correlation_matrix


if __name__ == "__main__":
    # Run the complete pipeline
    model, results, corr_matrix = main()
    
    # Now you can use the model:
    # predictions = model.predict(new_data)
    # results = model.check_accuracy(test_data, test_labels)