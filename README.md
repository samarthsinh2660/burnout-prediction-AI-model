# Burnout Prediction Model

A comprehensive machine learning model for predicting burnout risk using multiple datasets and advanced preprocessing techniques.

## 🚀 Features

### Core Capabilities
- **Multi-Dataset Integration**: Combines 9 different datasets for robust training
- **Advanced Preprocessing**: Missing value handling, correlation analysis, feature engineering
- **Class Balancing**: SMOTE + class weights for improved F1 scores
- **Interactive Predictions**: Real-time burnout risk assessment
- **Visualization**: Correlation heatmaps and feature analysis
- **Modular Architecture**: Professional code structure for maintainability

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

**New Modular Pipeline (Recommended):**
```bash
python run_pipeline.py
```

**Legacy Monolithic Script:**
```bash
python model.py
```

Both methods produce identical results. The modular version offers:
- Better code organization
- Easier maintenance
- Reusable components
- Professional structure

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
├── config/                       # Configuration
│   ├── __init__.py
│   └── config.py                 # Centralized settings
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_processor.py         # Data loading & preprocessing
│   ├── feature_analyzer.py       # Correlation analysis
│   ├── model_trainer.py          # Model training & prediction
│   └── main.py                   # Pipeline orchestration
├── utils/                        # Utilities
│   ├── __init__.py
│   └── logger.py                 # Structured logging
├── dataset/                      # Training data files
│   ├── train.csv
│   ├── mental_health_*.csv
│   └── ...
├── models/                       # Saved models
├── run_pipeline.py               # Main execution script
├── model.py                      # Legacy script (still functional)
├── interact.py                   # Interactive interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🎯 Pipeline Overview

The modular pipeline follows these steps:

1. **Data Loading** - Load multiple CSV datasets
2. **Data Cleaning** - Handle missing values systematically with smart indicators
3. **Correlation Analysis** - Identify feature relationships and multicollinearity
4. **Feature Engineering** - Create interaction features (stress-sleep ratio, digital overload, etc.)
5. **Preprocessing** - Scaling and encoding
6. **Model Training** - XGBoost with SMOTE class balancing
7. **Evaluation** - Comprehensive metrics and confusion matrix
8. **Deployment** - Save model for predictions

## 🔧 Advanced Usage

### Using Individual Modules
```python
from src.data_processor import DataProcessor
from src.model_trainer import BurnoutPredictor

# Load and process data
processor = DataProcessor()
datasets = processor.load_datasets(file_paths)
data = processor.merge_datasets(datasets)

# Train model
model = BurnoutPredictor()
model.train(X_train, y_train)
```

### Custom Threshold Prediction
```python
from src.model_trainer import BurnoutPredictor

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

### Modify Configuration
Edit `config/config.py` to change:
- Model hyperparameters
- Dataset paths
- Processing parameters
- Feature mappings

```python
# Example: Change model parameters
MODEL_PARAMS = {
    'learning_rate': 0.05,  # Adjust this
    'max_depth': 7,         # Or this
    'n_estimators': 300,    # Or this
}
```

## 📊 Output Files

After training, the following files are created in the `models/` folder:
- `models/burnout_combined_model.pkl` - Trained model
- `models/correlation_matrix_full.png` - Full feature correlation heatmap
- `models/correlation_matrix_focused.png` - Top features correlation

## 🎉 Key Improvements

### Modular Architecture Benefits:
- **Separation of Concerns**: Each module has a single responsibility
- **Reusability**: Import and use components independently
- **Maintainability**: Easy to locate and fix issues
- **Scalability**: Add features without breaking existing code
- **Testability**: Test modules in isolation
- **Configuration**: All settings centralized in `config/config.py`

### Technical Improvements:
- Smart missing value handling with indicators
- Group-specific median imputation
- Class weight optimization for F1-score
- Comprehensive correlation analysis
- Future-proof pandas operations (no warnings)

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
