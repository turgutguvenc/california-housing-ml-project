# california-housing-ml-project
Machine Learning project predicting California housing prices using regression models - Course Project

# California Housing Price Prediction
kaggle work: 
# Project Overview
This project implements supervised learning algorithms to predict median house values in California using socio-economic and geographic features from the 1990 census data.

## Dataset
- **Source**: [California Housing Prices Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- **Size**: 20,640 observations, 10 features
- **Target**: median_house_value
- **Problem Type**: Regression

## Key Features
- Comprehensive exploratory data analysis
- Feature engineering and data preprocessing
- Comparison of 12+ regression algorithms
- Hyperparameter tuning and ensemble methods
- Model interpretation and feature importance analysis

## Results Summary
- **Best Model**: Voting Regressor (Ensemble Method: CatBoost, LightGBM, xgbOOST) 
- **Test RMSE**: 43284.92
- **R² Score**: 0.857

## Files Description
- `notebooks/final_analysis.ipynb`: Complete analysis notebook
- `src/data_prep.py`: Custom data preprocessing functions
- `src/eda.py`: Exploratory data analysis utilities

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main analysis: `jupyter notebook notebooks/final_analysis.ipynb`
