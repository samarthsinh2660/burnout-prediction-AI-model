"""
Main Pipeline Module
Orchestrates the complete burnout prediction pipeline
"""

import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processor import DataProcessor
from src.feature_analyzer import FeatureAnalyzer
from src.model_trainer import BurnoutPredictor
from config.config import DATASET_PATHS, PROCESSING_PARAMS, OUTPUT_PATHS


def test_single_prediction(model):
    """
    EXAMPLE: How to predict for a single person
    """
    print("\n" + "="*60)
    print("TESTING SINGLE PREDICTION")
    print("="*60)
    
    # Example person 1: High burnout risk
    print("\nPerson 1: High Risk Profile")
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
    print(f"  Prediction: {'BURNOUT' if pred1 == 1 else 'NO BURNOUT'}")
    print(f"  Burnout Probability: {prob1*100:.2f}%")
    
    # Example person 2: Low burnout risk
    print("\nPerson 2: Low Risk Profile")
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
    print(f"  Prediction: {'BURNOUT' if pred2 == 1 else 'NO BURNOUT'}")
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
    print("\nSTEP 0: LOADING DATASETS")
    
    processor = DataProcessor()
    datasets = processor.load_datasets(DATASET_PATHS)
    
    if len(datasets) == 0:
        print("\nError: No datasets loaded. Please check file paths.")
        return None, None, None
    
    # ========================
    # MERGE DATASETS
    # ========================
    print("\nMERGING DATASETS")
    combined_data = processor.merge_datasets(datasets)
    
    # ========================
    # STEP 1: HANDLE MISSING VALUES
    # ========================
    print("\nSTEP 1: CLEANING DATA")
    combined_data = processor.handle_missing_values()
    
    # ========================
    # STEP 2: CORRELATION ANALYSIS
    # ========================
    print("\nSTEP 2: ANALYZING CORRELATIONS")
    analyzer = FeatureAnalyzer(combined_data)
    correlation_matrix = analyzer.analyze_correlations(save_plot=True)
    
    # ========================
    # STEP 3: FEATURE ENGINEERING
    # ========================
    print("\nSTEP 3: FEATURE ENGINEERING")
    combined_data = processor.preprocess_combined_data()
    
    # ========================
    # STEP 4: PREPARE DATA
    # ========================
    print("\nSTEP 4: PREPARING DATA FOR TRAINING")
    
    combined_data = combined_data.dropna(subset=['burnout_target'])
    
    X = combined_data.drop('burnout_target', axis=1)
    y = combined_data['burnout_target']
    
    print(f"\n✓ Total samples: {len(X)}")
    print(f"✓ Number of features: {X.shape[1]}")
    print(f"✓ Burnout cases: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"✓ No burnout cases: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=PROCESSING_PARAMS['test_size'], 
        stratify=y, 
        random_state=PROCESSING_PARAMS['random_state']
    )
    
    # ========================
    # STEP 5: NORMALIZATION (SCALING)
    # ========================
    print("\nSTEP 5: NORMALIZING FEATURES")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Features scaled using StandardScaler")
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    # ========================
    # STEP 6: TRAIN MODEL
    # ========================
    print("\nSTEP 6: TRAINING MODEL")
    predictor = BurnoutPredictor()
    predictor.feature_names = X.columns.tolist()
    predictor.train(X_train_scaled, y_train)
    
    # ========================
    # STEP 7: EVALUATE
    # ========================
    print("\nSTEP 7: EVALUATING MODEL")
    results = predictor.check_accuracy(X_test_scaled, y_test)
    
    # ========================
    # STEP 8: TEST PREDICTIONS
    # ========================
    print("\nSTEP 8: TESTING PREDICTIONS")
    test_single_prediction(predictor)
    
    # ========================
    # STEP 9: SAVE MODEL
    # ========================
    print("\nSTEP 9: SAVING MODEL")
    predictor.save_model(OUTPUT_PATHS['model'])
    
    # ========================
    # FINAL SUMMARY
    # ========================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*70)
    print(f"\nModel Performance:")
    print(f"  Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"  F1-Score:  {results['f1_score']:.4f}")
    print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
    
    print(f"\nFiles Created:")
    print(f"  {OUTPUT_PATHS['correlation_full']}")
    print(f"  {OUTPUT_PATHS['correlation_focused']}")
    print(f"  {OUTPUT_PATHS['model']}")
    
    print("\n" + "="*70)
    
    return predictor, results, correlation_matrix


if __name__ == "__main__":
    model, results, corr_matrix = main()
