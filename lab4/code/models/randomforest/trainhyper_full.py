# import packages and load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import os
from pyreadr import read_r
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import joblib

data_path = f"../../../data/image_data"
image_1 = pd.read_csv(f"{data_path}/image1.txt", delim_whitespace=True, header=None)
image_2 = pd.read_csv(f"{data_path}/image2.txt", delim_whitespace=True, header=None)
image_3 = pd.read_csv(f"{data_path}/image3.txt", delim_whitespace=True, header=None)

column_names = ['y_coor', 'x_coor', 'expert_label', 'NDAI', 'SD', 'CORR', 'Radiance_angle_DF','Radiance_angle_CF','Radiance_angle_BF','Radiance_angle_AF', 'Radiance_angle_AN'] 
image_1.columns = column_names
image_2.columns = column_names
image_3.columns = column_names


# remove when expert label is zero (remove uncertain data)
train_on1 = image_1[image_1['expert_label'] != 0]

# select all features except the expert_label column from dataset and call it x
#drop expert_label and x_coor, y_coor
#Don't use X and Y coordinates as features 
x_train1 = train_on1.drop(columns=['expert_label', 'x_coor', 'y_coor']) # explanatory variables
y_train1 = train_on1['expert_label'] # response variable we want to predict

# below are datasets we should use for caculating prediction accuracy using image 2
validate_on2 = image_2[image_2['expert_label'] != 0] # remove uncertain data

#drop expert_label and x_coor, y_coor
x_validate2 = validate_on2.drop(columns=['expert_label', 'x_coor', 'y_coor']) # explanatory variables
y_validate2 = validate_on2['expert_label'] # response variable we want to predict

# Try to tune the hyperparameters using gridsearch and cross validation (train on 1 and validate on 2)
rf = RandomForestClassifier(random_state=42)
param_grid = {
'n_estimators': [100, 200, 300], # number of trees in the forest, default=100. usually more trees better performance
'max_depth': [None, 10, 20, 30], # the maximum depth of the tree, default=none. may lead to overfitting if the depth too large
'min_samples_split': [2, 5, 10], # the minimum number of samples required to split an internal node, default=2 (the node will split as long as it has at least 2 samples). high values prevent overfitting
'min_samples_leaf': [1, 2, 4], # the minimum number of samples required to be at a leaf node, default=1. large values prevent overfitting
'criterion': ['gini', 'entropy'], # the function to measure the quality of a split, default=gini. 
'bootstrap': [True, False] # whether bootstrap samples are used when building trees, default=True
}

# Create a validation fold
X = pd.concat([x_train1, x_validate2], ignore_index=True)
y = pd.concat([y_train1, y_validate2], ignore_index=True)

# Create test_fold array for both splits simultaneously
test_fold = np.ones(len(X))  # Initialize with 1s (won't be used in CV)
test_fold[:len(x_train1)] = 0  # Elements with 0 are treated as validation data in split 1.
test_fold[len(x_train1):] = 1  # Elements with 1 are treated as validation data in split 2.
#should not use "-1" index because it will always be training. 
# With 0, 1 index, cv in Gridsearchcv will consider two scenarios: 0 as training and 1 as validation, and 1 as training and 0 as validation

# Create PredefinedSplit object
ps = PredefinedSplit(test_fold)

# Define multiple scoring metrics
scoring = {
    'roc_auc': 'roc_auc',        # Area Under the ROC Curve
    'accuracy': 'accuracy'      # Accuracy score
}

# Use GridSearchCV with the combined predefined split
grid_search = GridSearchCV(estimator=rf, 
                           param_grid=param_grid, 
                           scoring=scoring, 
                           cv=ps, 
                           refit='roc_auc', # Specify which metric to use for choosing the best model
                           n_jobs=-1)

# Fit the model with combined data
grid_search.fit(X, y)

# Access and print the results
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv('grid_search_results_full_model.csv', index=False)
best_roc_auc = grid_search.cv_results_['mean_test_roc_auc'].max()
best_accuracy = grid_search.cv_results_['mean_test_accuracy'].max()
best_params=grid_search.best_params_
# Create a dictionary to hold the best parameters and score
results_parameter = {
   'best_roc_auc': [best_roc_auc],
   'best_accuracy': [best_accuracy],
   'best_params': [best_params]
}

results_parameter = pd.DataFrame(results_parameter)
results_parameter.to_csv('results_parameter_full.csv', index=False)

# Save the best model
best_rf = grid_search.best_estimator_
joblib.dump(best_rf, 'best_random_forest_model_full.pkl')
