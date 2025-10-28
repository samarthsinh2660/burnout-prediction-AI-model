# Burnout Prediction Research Project

**Advanced Machine Learning Model for Burnout Risk Assessment**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 🎯 Model Performance Overview

### Individual Model Performance (Test Set - 26,717 samples)

| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|-------|----------|----------|-----------|--------|---------|
| **XGBoost** (Conservative) | 96.18% | 0.9032 | 82.99% | 99.06% | 0.9960 |
| **LightGBM** (Aggressive) | 96.20% | 0.9035 | 83.02% | 99.06% | 0.9961 |
| **CatBoost** (Balanced) | 96.12% | 0.9019 | 82.76% | 99.06% | 0.9960 |

### Hybrid Ensemble Performance
| Metric | Ensemble | Single Best | Improvement |
|--------|----------|-------------|-------------|
| **Accuracy** | **96.12%** | 96.20% | -0.08% |
| **F1-Score** | **0.9018** | 0.9035 | -0.17% |
| **ROC-AUC** | **0.9960** | 0.9961 | -0.01% |

**🏆 Best Performing Model: LightGBM (96.20% accuracy)**

---

## 📋 Table of Contents

- [Research Overview](#-research-overview)
- [Dataset Information](#-dataset-information)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technical Details](#-technical-details)
- [Results & Analysis](#-results--analysis)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

---

## 🔬 Research Overview

### Problem Statement

Burnout is a critical occupational health issue affecting millions of workers worldwide, leading to decreased productivity, mental health issues, and increased healthcare costs. Early detection and prevention of burnout is essential for maintaining workforce well-being and organizational performance.

### Research Objectives

1. **Develop an accurate predictive model** for burnout risk assessment
2. **Compare multiple machine learning algorithms** for optimal performance
3. **Implement ensemble methods** to improve prediction reliability
4. **Create interpretable features** for actionable insights
5. **Ensure model robustness** through comprehensive evaluation

### Methodology

This research employs a multi-faceted approach combining:
- **Multi-dataset integration** from diverse sources
- **Advanced feature engineering** for meaningful representations
- **Ensemble learning** with diverse algorithms
- **Rigorous validation** using train/validation/test splits
- **Comprehensive evaluation** metrics beyond accuracy

---

## 📊 Dataset Information

### Data Sources

The model was trained on **9 diverse datasets** providing comprehensive coverage of burnout risk factors:

| Dataset | Samples | Source | Focus Area |
|---------|---------|--------|------------|
| **Employee Burnout** | 22,750 | Corporate HR | Work-life balance |
| **HackerEarth Burnout** | 22,750 | Tech platform | Developer stress |
| **Workplace Mental Health** | 3,000 | Corporate survey | Office environment |
| **Student Resilience** | 500 | Academic study | Student stress |
| **Student Stress** | 1,100 | University survey | Academic pressure |
| **Tech Usage** | 100,000 | App analytics | Digital habits |
| **Screen Time** | 10,000 | Device tracking | Usage patterns |
| **Healthcare Mental** | 5,000 | Medical survey | Healthcare workers |
| **Additional Train** | 13,013 | Mixed sources | General population |

### Total Dataset: **178,113 samples**

### Class Distribution
- **Burnout Cases**: 32,038 (18.0%)
- **No Burnout Cases**: 146,075 (82.0%)
- **Class Imbalance Ratio**: 1:4.5

### Feature Engineering

#### Base Features (7 core features)
- **stress_level**: Self-reported stress (0-10 scale)
- **workload**: Work/study intensity (0-10 scale)
- **sleep_quality**: Sleep satisfaction (0-10 scale)
- **support**: Social support level (0-10 scale)
- **screen_time**: Daily screen usage (hours)
- **social_media**: Daily social media usage (hours)
- **work_setup**: Work environment (Office/WFH/Hybrid)

#### Engineered Features (14 additional)
- **Missing value indicators**: `workload_was_missing`, `support_was_missing`
- **Interaction features**: `stress_sleep_ratio`, `digital_overload`
- **Composite scores**: `work_life_balance`, `mental_health_index`
- **Categorical encodings**: One-hot encoded dataset sources

#### Final Feature Set: **21 features**

### Data Preprocessing

1. **Missing Value Handling**
   - Group-specific median imputation
   - Missing value indicators for categorical features
   - Zero imputation for numerical features

2. **Class Balancing**
   - SMOTE (Synthetic Minority Oversampling Technique)
   - Class weight adjustment in loss functions
   - Scale positive weight optimization

3. **Feature Scaling**
   - **No scaling applied** (optimized for tree-based models)
   - Preserves natural feature importance signals

---

## 🏗 Model Architecture

### Individual Models

#### 1. XGBoost (Conservative Strategy)
```python
Parameters:
- n_estimators: 400 (more trees)
- learning_rate: 0.03 (slower learning)
- max_depth: 6 (shallower trees)
- min_child_weight: 5 (conservative splits)
- reg_alpha: 0.5 (L1 regularization)
- reg_lambda: 1.5 (L2 regularization)
Strategy: Prevent overfitting through regularization
```

#### 2. LightGBM (Aggressive Strategy)
```python
Parameters:
- n_estimators: 300
- learning_rate: 0.05 (normal learning)
- max_depth: 9 (deeper trees)
- num_leaves: 127 (more leaves)
- min_child_samples: 10 (less conservative)
- subsample: 0.85 (row sampling)
- colsample_bytree: 0.85 (feature sampling)
Strategy: Capture complex patterns through depth
```

#### 3. CatBoost (Balanced Strategy)
```python
Parameters:
- iterations: 350
- learning_rate: 0.04
- depth: 7
- l2_leaf_reg: 5 (regularization)
- bagging_temperature: 0.8 (randomness)
- random_strength: 0.5 (split randomization)
- bootstrap_type: 'Bayesian'
Strategy: Different algorithm with built-in randomization
```

### Ensemble Architecture

#### Soft Voting Ensemble
- **Voting Method**: Soft voting (probability-based)
- **Weight Calculation**: 40% accuracy + 60% F1-score
- **Model Diversity**: Different hyperparameters and algorithms
- **Final Prediction**: Weighted average of probabilities

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended

### Quick Install
```bash
# Clone repository
git clone <repository-url>
cd burnout-prediction

# Install dependencies
pip install -r requirements.txt
```

### Required Packages
```
pandas==2.0.0          # Data manipulation
numpy==1.24.0          # Numerical computing
scikit-learn==1.3.0    # ML algorithms
xgboost==2.0.0         # Gradient boosting
lightgbm==4.0.0        # Fast GBM
catboost==1.2.0        # Categorical GBM
imbalanced-learn==0.11.0 # SMOTE
matplotlib==3.7.0      # Plotting
seaborn==0.12.0        # Visualization
joblib==1.3.0          # Model serialization
```

---

## 💻 Usage

### Training the Complete Pipeline
```bash
python run_pipeline.py
```

This executes the full research pipeline:
1. Loads and merges 9 datasets
2. Performs data cleaning and feature engineering
3. Trains XGBoost single model
4. Trains hybrid ensemble (XGBoost + LightGBM + CatBoost)
5. Evaluates both on validation and test sets
6. Saves models and generates visualizations

### Interactive Prediction
```bash
python interact.py
```

Provides:
- Real-time burnout risk assessment
- Sample prediction demonstrations
- Model comparison interface

### Using Individual Components
```python
from src.data_processor import DataProcessor
from src.model_trainer import BurnoutPredictor, HybridEnsemble

# Load and process data
processor = DataProcessor()
datasets = processor.load_datasets()
combined_data = processor.merge_datasets(datasets)

# Train single model
single_model = BurnoutPredictor()
single_model.train(X_train, y_train, X_val, y_val)

# Train ensemble
ensemble = HybridEnsemble()
ensemble.train(X_train, y_train, X_val, y_val)

# Make predictions
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)
```

---

## 🔧 Technical Details

### Training Configuration

#### Data Split Strategy
- **Training Set**: 70% (124,679 samples)
- **Validation Set**: 15% (26,717 samples)
- **Test Set**: 15% (26,717 samples)
- **Stratified splitting** to maintain class distribution

#### Evaluation Metrics
- **Accuracy**: Overall prediction accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Hyperparameter Optimization

Each model was tuned for optimal performance:
- **Grid search** on learning rates and tree depths
- **Cross-validation** on training set
- **Validation monitoring** to prevent overfitting
- **Early stopping** with patience of 50 iterations

### Model Interpretability

#### Feature Importance Analysis
- **XGBoost**: Gain-based feature importance
- **LightGBM**: Split-based feature importance
- **CatBoost**: Prediction value change

#### Key Findings
1. **workload** (55.68% correlation with burnout)
2. **stress_level** (10.74% correlation)
3. **sleep_quality** (negative correlation)
4. **support_was_missing** (predictive of burnout risk)

---

## 📈 Results & Analysis

### Performance Comparison

#### Test Set Results (26,717 samples)

| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|-------|----------|----------|-----------|--------|---------|
| XGBoost | 96.18% | 0.9032 | 82.99% | 99.06% | 0.9960 |
| LightGBM | 96.20% | 0.9035 | 83.02% | 99.06% | 0.9961 |
| CatBoost | 96.12% | 0.9019 | 82.76% | 99.06% | 0.9960 |
| **Ensemble** | 96.12% | 0.9018 | 82.76% | 99.06% | 0.9960 |

### Confusion Matrix Analysis

#### XGBoost (Best Individual Model)
```
                  Predicted
                No Burnout  Burnout
Actual No Burnout     20935       976
Actual Burnout           45      4761

- True Negatives: 20,935 (78.7%)
- False Positives: 976 (3.7%)
- False Negatives: 45 (0.2%)
- True Positives: 4,761 (17.8%)
```

### Model Strengths & Weaknesses

#### XGBoost (Conservative)
- ✅ **Best generalization** - consistent performance
- ✅ **Robust to overfitting** - regularization helps
- ✅ **High recall** - catches most burnout cases
- ⚠️ Slightly lower precision than aggressive models

#### LightGBM (Aggressive)
- ✅ **Highest accuracy** - best overall performance
- ✅ **Good balance** - precision vs recall trade-off
- ✅ **Fast training** - efficient algorithm
- ⚠️ May overfit on noisy data

#### CatBoost (Balanced)
- ✅ **Stable performance** - consistent results
- ✅ **Handles categorical** - built-in categorical support
- ✅ **Different algorithm** - algorithmic diversity
- ⚠️ Slightly slower training

#### Hybrid Ensemble
- ✅ **Robust predictions** - combines model strengths
- ✅ **Uncertainty estimation** - probability-based voting
- ✅ **Bias reduction** - multiple perspectives
- ⚠️ Computational overhead
- ⚠️ No significant improvement in this case

### Research Insights

1. **Tree-based models excel** at burnout prediction
2. **Feature engineering is crucial** - interaction features improve performance
3. **SMOTE + class weights** effectively handle class imbalance
4. **Ensemble methods** provide robustness but may not always improve accuracy
5. **Conservative models** generalize better than aggressive ones

### Limitations & Future Work

#### Current Limitations
- **Dataset bias**: Primarily tech and corporate samples
- **Self-reported data**: Subject to reporting bias
- **Static predictions**: No temporal modeling
- **Limited features**: Focus on work-related factors

#### Future Improvements
- **Temporal modeling**: Include time-series burnout progression
- **Additional features**: Include physiological data (HRV, sleep tracking)
- **Multi-modal learning**: Combine self-reports with behavioral data
- **Explainable AI**: Implement SHAP values for individual predictions
- **Real-time monitoring**: Continuous risk assessment systems

---

## 📁 Project Structure

```
burnout-prediction-research/
├── 📁 config/                       # Configuration files
│   ├── __init__.py
│   └── config.py                    # Model & data configuration
├── 📁 src/                          # Source code (modular)
│   ├── __init__.py
│   ├── data_processor.py            # Data loading & preprocessing
│   ├── feature_analyzer.py          # Correlation & feature analysis
│   ├── loss_functions.py            # Custom loss functions
│   ├── single_model_trainer.py      # XGBoost model trainer
│   ├── ensemble_trainer.py          # Hybrid ensemble trainer
│   ├── model_trainer.py             # Import hub (backward compatibility)
│   └── main.py                      # Pipeline orchestration
├── 📁 utils/                        # Utilities
│   ├── __init__.py
│   └── logger.py                    # Structured logging
├── 📁 dataset/                      # Research datasets
│   ├── employee_burnout.csv
│   ├── hackerearth_burnout.csv
│   ├── workplace_mental.csv
│   ├── student_resilience.csv
│   ├── student_stress.csv
│   ├── tech_usage.csv
│   ├── screen_time.csv
│   ├── healthcare_mental.csv
│   └── additional_train.csv
├── 📁 models/                       # Trained models & outputs
│   ├── single_model_xgboost.pkl     # Best single model
│   ├── xgboost_model.pkl            # Ensemble XGBoost
│   ├── lightgbm_model.pkl           # Ensemble LightGBM
│   ├── catboost_model.pkl           # Ensemble CatBoost
│   ├── hybrid_ensemble.pkl          # Combined ensemble
│   ├── training_history.png         # Single model curves
│   ├── ensemble_training_history.png # All model curves
│   ├── correlation_matrix_full.png  # Full correlations
│   └── correlation_matrix_focused.png # Key correlations
├── 📄 run_pipeline.py               # Main execution script
├── 📄 interact.py                   # Interactive interface
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # This research documentation
└── 📄 STRUCTURE.md                  # Code architecture details
```

---

## 🤝 Contributing

### Research Collaboration
We welcome contributions to improve burnout prediction research:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/new-approach`)
3. **Implement** your changes
4. **Test** thoroughly on validation set
5. **Submit** a pull request with detailed description

### Areas for Contribution
- **New datasets** for broader population coverage
- **Alternative algorithms** (neural networks, SVMs)
- **Feature engineering** techniques
- **Interpretability methods** (SHAP, LIME)
- **Real-world validation** studies

### Code Standards
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation

---

## 📜 License

This research project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact & Citation

**Research Lead**: Burnout Prediction Research Team

**Repository**: [GitHub Link]

**Contact**: Open an issue for questions or collaborations

### Citation
```
@research{burnout-prediction-2024,
  title={Multi-Model Ensemble Approach for Burnout Risk Prediction},
  author={Research Team},
  year={2024},
  publisher={GitHub Repository}
}
```

---

**🔬 This research demonstrates that machine learning can effectively predict burnout risk with 96%+ accuracy using multi-dataset integration and ensemble methods.**
