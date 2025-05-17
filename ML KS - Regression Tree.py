#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 16:25:08 2023

@author: juliana
"""



# Decision Tree: Regression

# Example with Bank Case Study data



# Preliminaries
data_path = r"/home/juliana/Документи/Experian - ML Study/Data"

data_file = r"BankCaseStudyData.csv"

reports_path = r"/home/juliana/Документи/Experian - ML Study/Reports"



# Import Python APIs
import numpy as np
import os
import pandas as pd
import graphviz
import lightgbm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from lightgbm import LGBMRegressor



# Read Data
data_all = pd.read_csv(os.path.join(data_path, data_file),
                       keep_default_na = False,
                       encoding = 'utf8',
                       low_memory = False)



# Derive columns
columns = data_all.columns.to_list()



# Derive Modelling variables
model_features = data_all[['Loan_Amount',
                           'Time_at_Address',
                           'Time_in_Employment',
                           'Time_with_Bank',
                           'SP_Number_Of_Searches_L6M',
                           'SP_Number_of_CCJs']].copy()

model_target = data_all[['Gross_Annual_Income']].copy()


# Create Train-Test split
x_train, x_test,y_train, y_test = train_test_split(model_features,
                                                   model_target,
                                                   test_size = 0.2,
                                                   random_state = 23)



# Model development
final_model = LGBMRegressor(boosting_type='gbdt',
                            num_leaves = 5,
                            max_depth = 5,
                            n_estimators = 2,
                            min_child_samples = 500)



# Train Final Model
final_model.fit(x_train, y_train)



# Evaluate Final Model
final_model.score(x_train, y_train)

final_model.score(x_test, y_test)


# Calculate Score on total data
data_all_features = data_all[final_model.feature_name_]

data_all['ML Income'] = final_model.predict(data_all_features, raw_score = True)


# Export subsample of data
data_all.to_csv(os.path.join(data_path, r"Decision Tree: Regressor GAI.csv"))


# Visualize Final Model
graph = lightgbm.create_tree_digraph(final_model, tree_index = 0)

graph.render(os.path.join(reports_path, r"LGBM - Regressor- Tree 0"), format = 'png')


# Good Luck!