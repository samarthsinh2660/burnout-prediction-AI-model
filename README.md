# Burnout Prediction Model

A comprehensive machine learning model for predicting burnout risk using multiple datasets and advanced preprocessing techniques.

## 🚀 Features

### Core Capabilities
- **Multi-Dataset Integration**: Combines 9 different datasets for robust training
- **Advanced Preprocessing**: Missing value handling, correlation analysis, feature engineering
- **Class Balancing**: SMOTE + class weights for improved F1 scores
- **Interactive Predictions**: Real-time burnout risk assessment
- **Visualization**: Correlation heatmaps and feature analysis

### Technical Features
- **Algorithm**: XGBoost with class weighting
- **Preprocessing**: Standard scaling, one-hot encoding, feature engineering
- **Evaluation**: Accuracy, F1-score, ROC-AUC, confusion matrix
- **Data Sources**: Employee, workplace, student, healthcare, and tech usage data

## 📊 Model Features

The model uses 7 key features for prediction:
- **stress_level**: Current stress levels (0-10)
- **workload**: Work/study intensity (0-10)
- **sleep_quality**: Sleep quality rating (0-10)
- **support**: Social support level (0-10)
- **screen_time**: Daily screen usage (hours)
- **social_media**: Daily social media usage (hours)
- **work_setup**: Work environment (0=Office, 1=WFM, 0.5=Hybrid)

## 🛠 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies
```bash
# Clone or navigate to project directory
cd path/to/burnout-prediction

# Install all requirements
pip install -r requirements.txt
```

### Required Packages
- pandas - Data manipulation
- numpy - Numerical computing
- scikit-learn - ML algorithms
- xgboost - Gradient boosting
- imbalanced-learn - SMOTE
- matplotlib - Plotting
- seaborn - Visualization
- joblib - Model serialization

## 🚀 Usage

### Train the Model
```bash
python model.py
```
This will:
1. Load and merge datasets
2. Handle missing values
3. Analyze correlations (saves heatmaps)
4. Engineer features
5. Train XGBoost model
6. Evaluate performance
7. Save trained model

### Interactive Predictions
```bash
python interact.py
```
Choose option 1 for personalized burnout assessment or option 2 for sample predictions.

## 📈 Model Performance

- **Accuracy**: 96.43%
- **F1-Score**: 0.9087
- **ROC-AUC**: 0.9959
- **Recall**: 98.72% (Excellent for healthcare screening)
- **Precision**: 84.18%
- **Dataset Size**: 178,113+ samples (9 datasets combined)
- **Features**: 21 (including 2 missing indicators)

## 📁 Project Structure

```
burnout-prediction/
├── model.py              # Main training script
├── interact.py           # Interactive prediction interface
├── requirements.txt      # Python dependencies
├── dataset/              # Training data files
│   ├── train.csv
│   ├── mental_health_*.csv
│   └── ...
└── burnout_combined_model.pkl  # Trained model (generated)
```

## 🎯 Pipeline Overview

1. **Data Loading** - Load multiple CSV datasets
2. **Data Cleaning** - Handle missing values systematically
3. **Correlation Analysis** - Identify feature relationships
4. **Feature Engineering** - Create interaction features
5. **Preprocessing** - Scaling and encoding
6. **Model Training** - XGBoost with class balancing
7. **Evaluation** - Comprehensive metrics
8. **Deployment** - Save model for predictions

## 🔧 Advanced Usage

### Custom Threshold Prediction
```python
from model import BurnoutPredictor
model = BurnoutPredictor()
model.load_model('burnout_combined_model.pkl')

# Predict with custom threshold for better F1
predictions = model.predict_with_threshold(X_test, threshold=0.4)
```

### Batch Predictions
```python
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

## 📊 Output Files

After training, the following files are created:
- `burnout_combined_model.pkl` - Trained model
- `correlation_matrix_full.png` - Full feature correlation heatmap
- `correlation_matrix_focused.png` - Top features correlation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source. Feel free to use and modify.

## 📧 Contact

For questions or improvements, please open an issue in the repository.
