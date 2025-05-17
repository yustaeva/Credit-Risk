#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 20:26:09 2023

@author: juliana
"""



''' 
Machine Learning Guild - Knowledge Sharing

Decision Tree Classifier

Exercise
'''


# Import Python APIs
import numpy as np
import os
import pandas as pd
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.tree import plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score



'''
Step 1: Load the Breast Cancer dataset from SciKit Learn

The Breast Cancer dataset object includes data on breast cancer, features,
target, as well as additional information like feature names and target categories
'''

# Load the Breast Cancer dataset
dataset = load_breast_cancer(as_frame = True)

# Load only the features and target from the Breast Cancer dataset
data_all = dataset.frame

# Check Target categories
target_categories = dataset.target_names

'''
The Breast Cancer dataset contains 569 records. There are 30 features with
different measurements such as mean radius, mean area and mean texture of the
infected tissue. The target variable is the diagnosis of the patient. There
are two categories:

Category 0 - Malignant
Category 1 - Benign

All features are numerical and therefore no additional data processing
is required.
'''



'''
Step 2: Create a Train Test split

In Classification problems we should always split the available data in
Train and Test samples.
The Train sample is used to develop a model.
The Test sample is used to measure the model's performance on previously
unseen data.
The final model should perform equally well on the Train and Test data.
'''

# Save only predictive features in a dataframe
features = data_all.drop(['target'], axis = 1)

# Save only the target variable in a dataframe
target = data_all[['target']].copy()

# Create Train and Test samples
X_train, X_test, Y_train, Y_test = train_test_split(features, 
                                                    target, 
                                                    test_size=0.33, 
                                                    random_state = 23)

'''
Now we have created 4 dataframes:
X_train - Features of the Train sample
Y_train - Target of the Train sample
X_test - Features of the Test sample
Y_test - Target of the Test sample

We have split our data into a Train and Test samples that have the ratio:
Train sample - 67% of the total data
Test sample - 33% of the total data
'''



'''
Step 3: Model Development

In this step we will create a Gradient Boost model.

We will use the following model parameters:
n_estimators - the number of decision trees in the Gradient Boost model
max_depth - the maximum number of splits in the Gradient Boost model before a Leaf Node is met
min_samples_leaf - the minimum number of elements required in each Leaf Node. This hyperparameter could be provided as a number or as percentage.
max_leaf_nodes - the maximum number of Leaf Nodes in the Gradient Boost model
'''


''' Exercise 1 - Manual Tuning '''

# Hyperparameters of the model - Manual Tuning
hyperparameters = {'n_estimators' : 10,
                   'max_depth' : 5,
                   'min_samples_leaf' : 0.05,
                   'max_leaf_nodes' : 10}

# Create a Gradient Boost model
gradient_boost_model = GradientBoostingClassifier(**hyperparameters)

# Fit the Gradient Boost model to the Train sample
gradient_boost_model.fit(X_train, Y_train)

# Create a list of model feature names
model_features = gradient_boost_model.feature_names_in_.tolist()

# Plot the Gradient Boost model - 1st Tree
tree_1 = gradient_boost_model.estimators_[1, 0]

plot_tree(decision_tree = tree_1,
          feature_names = model_features)




''' 
Exercise 2 - Grid Search Optimization

Grid Search is an exhaustive search optimization technique that can be
used for Hyperparameter tuning and finding the best set of hyperparameters
for the Gradient Boost model. In this example we will pass a list of values
for the model hyperparameters in the hyperparameters dictionary.
The Grid search algorithm will try out all combinations and return the best one.

The best set of hyperparameters is the set that results in highest accuracy.
'''

# Hyperparameters of the model - Grid Search Tuning
hyperparameters = {'n_estimators' : [10, 20, 25, 30, 50],
                   'max_depth' : [5, 10, 15],
                   'min_samples_leaf' : [0.05, 0.10],
                   'max_leaf_nodes' : [10, 15, 20]}

# Create a Gradient Boost model
study_model = GradientBoostingClassifier()

# Grid Search Optimization
grid_search = GridSearchCV(study_model, hyperparameters)

# Fit Grid Search to Training data
grid_search.fit(X_train, Y_train)

# Save results in Pandas dataframe
grid_search_results = pd.DataFrame(grid_search.cv_results_)

# Get best set of hyperparameters
best_tuning = grid_search.best_params_

# Create a final model
gradient_boost_model = GradientBoostingClassifier(**best_tuning)

# Fit the Gradient Boost model to the Train sample
gradient_boost_model.fit(X_train, Y_train)

# Create a list of model feature names
model_features = gradient_boost_model.feature_names_in_.tolist()

# Plot the Gradient Boost model - 1st Tree
tree_1 = gradient_boost_model.estimators_[1, 0]

plot_tree(decision_tree = tree_1,
          feature_names = model_features)





''' 
Exercise 3 - Random Search Optimization

Random Search is an randomized search optimization technique that can be
used for Hyperparameter tuning and finding the best set of hyperparameters
for the Gradient Boost model. In this example we will pass a list of values
for the model hyperparameters in the hyperparameters dictionary.
The Random Search algorithm will try a number of combinations and return the best one.
The number of randomly selected combinations is selected by the user with the n_iter parameter.

The best set of hyperparameters is the set that results in highest accuracy.
'''

# Hyperparameters of the model - Grid Search Tuning
hyperparameters = {'n_estimators' : [10, 20, 25, 30, 50],
                   'max_depth' : [5, 10, 15],
                   'min_samples_leaf' : [0.05, 0.10],
                   'max_leaf_nodes' : [10, 15, 20]}

# Create a Gradient Boost model
study_model = GradientBoostingClassifier()

# Randomized Search Optimization
random_search = RandomizedSearchCV(study_model, hyperparameters, n_iter = 10)

# Fit Randomized Search to Training data
random_search.fit(X_train, Y_train)

# Save results in Pandas dataframe
random_search_results = pd.DataFrame(random_search.cv_results_)

# Get best set of hyperparameters
best_tuning = random_search.best_params_

# Create a final model
gradient_boost_model = GradientBoostingClassifier(**best_tuning)

# Fit the Gradient Boost model to the Train sample
gradient_boost_model.fit(X_train, Y_train)

# Create a list of model feature names
model_features = gradient_boost_model.feature_names_in_.tolist()

# Plot the Gradient Boost model - 1st Tree
tree_1 = gradient_boost_model.estimators_[1, 0]

plot_tree(decision_tree = tree_1,
          feature_names = model_features)




'''
Step 4: Model Performance

Measure the Gradient Boost model's performance on the Test data.
'''

# Predict target variable on Test sample
Y_predicted = gradient_boost_model.predict(X_test)

# Create a Confusion Matrix for the Test sample
confusion_matrix = confusion_matrix(Y_test, Y_predicted)

# Calculate model Accuracy on Test sample
print(accuracy_score(Y_test, Y_predicted))



'''
Step 5: Model Validation

In Classification problems we validate a model by measuring it's
performance on the Train and Test samples. A valid model would perform
equally well on the Train and Test samples. A situation whereby a model
performs worse on the Test sample than on the Train sample is referred to
as Overfitting.
'''

# Predict target variable on Train sample
train_prediction = gradient_boost_model.predict(X_train)

# Predict target variable on Test sample
test_prediction = gradient_boost_model.predict(X_test)

# Print the Accuracy Score on the Train sample
print("The Accuracy Score on the Train sample is: {score}".format(score = accuracy_score(Y_train, train_prediction)))

# Print the Accuracy Score on the Test sample
print("The Accuracy Score on the Test sample is: {score}".format(score = accuracy_score(Y_test, test_prediction)))



'''
Step 6: Feature Importance

SHAP values are a universal algorithm for computing feature importance in any 
Machine Learning algorithm by calculating the marginal contribution of each model 
feature to the final model outcome.

In this example we will see how to derive feature importance with SHAP.
'''

# SHAP - Create Tree Explainer
tree_explainer = shap.TreeExplainer(gradient_boost_model, 
                                    X_train)

# SHAP - Values
shap_values = tree_explainer.shap_values(X_train,
                                         Y_train)

# Plot SHAP Feature Contribution
shap.summary_plot(shap_values, 
                  X_train,
                  gradient_boost_model.feature_names_in_)

# Good Luck!