#!/bin/bash
conda activate 215a
# Train Autoencoder
python autoencoder/hyperparameter_search.py
# Create EDA Plots
jupyter nbconvert --to notebook --execute --inplace EDA/eda.ipynb

# Logistic Regression
jupyter nbconvert --to notebook --execute --inplace models/logistic_regression/logistic_regression.ipynb
# Random Forest
python models/randomforest/trainhyper_full.py
python models/randomforest/trainhyper_reduc.py
jupyter nbconvert --to notebook --execute --inplace models/randomforest/randomforest.ipynb
# Weighted Ensembles
python models/ensemble/data_preprocessing.py
python models/ensemble/fit_model.py
python models/ensemble/data_preprocessing.py --use_only_engineered_features
python models/ensemble/fit_model.py --use_only_engineered_features
python models/ensemble/fit_stability_analysis_models.py 1
python models/ensemble/fit_stability_analysis_models.py 2
jupyter nbconvert --to notebook --execute --inplace models/ensemble/assess_model.ipynb
jupyter nbconvert --to notebook --execute --inplace models/ensemble/stability_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace models/ensemble/error_analysis.ipynb
# Convolutional Neural Network
python models/cnn/hyperparameter_search.py
jupyter nbconvert --to notebook --execute --inplace models/cnn/assess_model.ipynb

# Model Comparison
jupyter nbconvert --to notebook --execute --inplace models/model_comparison.ipynb
