#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:06:59 2023

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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
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

In this step we will create a Decision Tree Classifier model.

We will use the following model parameters:
criterion - the statistical metric to quantify information gain of the splits
max_depth - the maximum number of splits in the Decision Tree model before a Leaf Node is met
min_samples_leaf - the minimum number of elements required in each Leaf Node. This hyperparameter could be provided as a number or as percentage.
max_leaf_nodes - the maximum number of Leaf Nodes in the Decision Tree
'''

# Hyperparameters of the model
hyperparameters = {'criterion' : 'gini',
                   'max_depth' : 5,
                   'min_samples_leaf' : 0.05,
                   'max_leaf_nodes' : 10}

# Create a Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier(**hyperparameters)

# Fit the Decision Tree Classifier to the Train sample
decision_tree_model.fit(X_train, Y_train)

# Create a list of model feature names
model_features = decision_tree_model.feature_names_in_.tolist()

# Plot the Decision Tree Classifier
plot_tree(decision_tree = decision_tree_model,
          feature_names = model_features)



'''
Step 4: Model Performance

Measure the Decision Tree Classifier's performance on the Test data.
'''

# Predict target variable on Test sample
Y_predicted = decision_tree_model.predict(X_test)

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
train_prediction = decision_tree_model.predict(X_train)

# Predict target variable on Test sample
test_prediction = decision_tree_model.predict(X_test)

# Print the Accuracy Score on the Train sample
print("The Accuracy Score on the Train sample is: {score}".format(score = accuracy_score(Y_train, train_prediction)))

# Print the Accuracy Score on the Test sample
print("The Accuracy Score on the Test sample is: {score}".format(score = accuracy_score(Y_test, test_prediction)))



'''
Step 6: Feature Importance

There are multiple ways to measure feature importance. In SciKit Learn
feature importance in Decision Trees is measured by calculating the Gini importance.

An alternative approach to Gini Impurity are SHAP values. SHAP values are a universal
algorithm for computing feature importance in any Machine Learning algorithm by calculating
the marginal contribution of each model feature to the final model outcome.

In this example we will see how to derive feature importance with SciKit Learn and SHAP.
'''

# Measure feature importance with SciKit Learn
feature_importance = pd.DataFrame({'Feature' : decision_tree_model.feature_names_in_,
                                   'Gini_Importance' : decision_tree_model.feature_importances_})

# Plot Feature Importance - Gini Importance
feature_importance.sort_values(by = 'Gini_Importance', ascending = False).plot(kind = 'bar')



# SHAP - Create Tree Explainer
tree_explainer = shap.TreeExplainer(decision_tree_model, 
                                    X_train)

# SHAP - Values
shap_values = tree_explainer.shap_values(X_train,
                                         Y_train)

# Plot SHAP Feature Contribution
shap.summary_plot(shap_values, 
                  X_train,
                  decision_tree_model.feature_names_in_)

# Good Luck!