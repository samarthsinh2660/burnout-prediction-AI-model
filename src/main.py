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
from src.model_trainer import BurnoutPredictor, HybridEnsemble
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
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Burnout cases: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"No burnout cases: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    
    # Split into train, validation, and test sets
    print(f"\nSplitting data into Train/Validation/Test...")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=PROCESSING_PARAMS['test_size'], 
        stratify=y, 
        random_state=PROCESSING_PARAMS['random_state']
    )
    
    # Second split: separate validation from training
    val_size_adjusted = PROCESSING_PARAMS['validation_size'] / (1 - PROCESSING_PARAMS['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=PROCESSING_PARAMS['random_state']
    )
    
    # ========================
    # STEP 5: DATA PREPARATION (NO SCALING FOR TREE MODELS)
    # ========================
    print("\nSTEP 5: PREPARING DATA FOR TREE-BASED MODELS")
    print("Note: Tree-based models (XGBoost, LightGBM, CatBoost) don't need feature scaling")
    print("Using raw features to preserve natural importance signals and increase model diversity")
    
    # Use unscaled data for better ensemble diversity
    X_train_processed = X_train
    X_val_processed = X_val
    X_test_processed = X_test
    
    print(f"\nTraining set: {X_train_processed.shape[0]} samples ({X_train_processed.shape[0]/len(X)*100:.1f}%)")
    print(f"Validation set: {X_val_processed.shape[0]} samples ({X_val_processed.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set: {X_test_processed.shape[0]} samples ({X_test_processed.shape[0]/len(X)*100:.1f}%)")
    
    # ========================
    # STEP 6: TRAIN SINGLE MODEL (XGBoost)
    # ========================
    print("\nSTEP 6: TRAINING SINGLE MODEL (XGBoost) WITH VALIDATION")
    predictor = BurnoutPredictor()
    predictor.feature_names = X.columns.tolist()
    predictor.train(X_train_processed, y_train, X_val_processed, y_val, use_custom_loss=True)
    
    # Plot training history
    predictor.plot_training_history()
    
    # ========================
    # STEP 7: EVALUATE SINGLE MODEL ON VALIDATION SET
    # ========================
    print("\nSTEP 7: EVALUATING SINGLE MODEL ON VALIDATION SET")
    val_results = predictor.check_accuracy(X_val_processed, y_val)
    
    # ========================
    # STEP 8: EVALUATE SINGLE MODEL ON TEST SET
    # ========================
    print("\nSTEP 8: EVALUATING SINGLE MODEL ON TEST SET")
    test_results = predictor.check_accuracy(X_test_processed, y_test)
    
    # ========================
    # STEP 9: TRAIN DIVERSE HYBRID ENSEMBLE (XGBoost + LightGBM + CatBoost)
    # ========================
    print("\n" + "="*70)
    print("STEP 9: TRAINING DIVERSE HYBRID ENSEMBLE")
    print("="*70)
    print("Using diverse hyperparameters and feature sets for maximum ensemble benefit")
    
    ensemble = HybridEnsemble()
    ensemble.feature_names = X.columns.tolist()
    ensemble.train(X_train_processed, y_train, X_val_processed, y_val)
    
    # Plot ensemble training history
    ensemble.plot_training_history()
    
    # ========================
    # STEP 10: EVALUATE ENSEMBLE ON VALIDATION SET
    # ========================
    print("\nSTEP 10: EVALUATING ENSEMBLE ON VALIDATION SET")
    ensemble_val_results = ensemble.check_accuracy(X_val_processed, y_val)
    
    # ========================
    # STEP 11: EVALUATE ENSEMBLE ON TEST SET
    # ========================
    print("\nSTEP 11: EVALUATING ENSEMBLE ON TEST SET")
    ensemble_test_results = ensemble.check_accuracy(X_test_processed, y_test)
    
    # ========================
    # STEP 12: TEST PREDICTIONS
    # ========================
    print("\nSTEP 12: TESTING PREDICTIONS")
    test_single_prediction(predictor)
    
    # ========================
    # STEP 13: SAVE MODELS
    # ========================
    print("\nSTEP 13: SAVING MODELS")
    
    # Save single XGBoost model
    predictor.save_model('./models/single_model_xgboost.pkl')
    
    # Save ensemble and individual models
    ensemble.save_model()
    
    print(f"\n✓ Single XGBoost model saved: ./models/single_model_xgboost.pkl")
    print(f"✓ Hybrid ensemble saved: ./models/hybrid_ensemble.pkl")
    print(f"✓ Individual models also saved within ensemble")
    
    print(f"\n{'='*70}")
    print(f"SINGLE MODEL (XGBoost) - Test Set:")
    print(f"{'='*70}")
    print(f"  Accuracy:  {test_results['accuracy']*100:.2f}%")
    print(f"  F1-Score:  {test_results['f1_score']:.4f}")
    print(f"  ROC-AUC:   {test_results['roc_auc']:.4f}")
    print(f"  Recall:    {test_results['recall']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"HYBRID ENSEMBLE (XGBoost + LightGBM + CatBoost) - Test Set:")
    print(f"{'='*70}")
    print(f"  Accuracy:  {ensemble_test_results['accuracy']*100:.2f}%")
    print(f"  F1-Score:  {ensemble_test_results['f1_score']:.4f}")
    print(f"  ROC-AUC:   {ensemble_test_results['roc_auc']:.4f}")
    print(f"  Recall:    {ensemble_test_results['recall']:.4f}")
    print(f"  Precision: {ensemble_test_results['precision']:.4f}")
    
    # Calculate improvements
    acc_improvement = (ensemble_test_results['accuracy'] - test_results['accuracy']) * 100
    f1_improvement = ensemble_test_results['f1_score'] - test_results['f1_score']
    
    print(f"\n{'='*70}")
    print(f"IMPROVEMENT WITH ENSEMBLE:")
    print(f"{'='*70}")
    print(f"  Accuracy Gain:  {acc_improvement:+.2f}%")
    print(f"  F1-Score Gain:  {f1_improvement:+.4f}")
    
    if ensemble_test_results['accuracy'] > test_results['accuracy']:
        print(f"\n  ✓ Ensemble performs BETTER!")
        best_model = "HYBRID ENSEMBLE"
    else:
        print(f"\n  Single model performs better (ensemble might be overfitting)")
        best_model = "SINGLE MODEL (XGBoost)"
    
    print(f"\n{'='*70}")
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"{'='*70}")
    
    print(f"\n{'='*70}")
    print(f"VALIDATION SET RESULTS (For Comparison):")
    print(f"{'='*70}")
    print(f"Single Model Val Accuracy: {val_results['accuracy']*100:.2f}%")
    print(f"Ensemble Val Accuracy:     {ensemble_val_results['accuracy']*100:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"FILES CREATED:")
    print(f"{'='*70}")
    print(f"  ./models/single_model_xgboost.pkl          - Single XGBoost model")
    print(f"  ./models/xgboost_model.pkl                 - XGBoost from ensemble")
    print(f"  ./models/lightgbm_model.pkl                - LightGBM from ensemble")
    print(f"  ./models/catboost_model.pkl                - CatBoost from ensemble")
    print(f"  ./models/hybrid_ensemble.pkl               - Hybrid ensemble")
    print(f"  ./models/correlation_matrix_full.png       - Full correlation heatmap")
    print(f"  ./models/correlation_matrix_focused.png    - Top features correlation")
    print(f"  ./models/training_history.png              - Single model training curves")
    print(f"  ./models/ensemble_training_history.png     - All 3 models training curves")
    
    print("\n" + "="*70)
    
    return {
        'single_model': predictor,
        'ensemble': ensemble,
        'results': {
            'single': {'validation': val_results, 'test': test_results},
            'ensemble': {'validation': ensemble_val_results, 'test': ensemble_test_results}
        },
        'correlation_matrix': correlation_matrix,
        'best_model': best_model
    }


if __name__ == "__main__":
    output = main()
