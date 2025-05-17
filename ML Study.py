#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 13:40:13 2022

@author: juliana
"""


# Experian - Bank Case Study

# Study Model

# Gradient Boosting Decision Trees




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
from lightgbm import LGBMClassifier




# Read Data
data_all = pd.read_csv(os.path.join(data_path, data_file),
                       keep_default_na = False,
                       encoding = 'utf8',
                       low_memory = False)



# Derive columns
columns = data_all.columns.to_list()



# Derive Goods and Bads
print(data_all['GB_Flag'].value_counts())

data_good_bad = data_all[(data_all['GB_Flag'] == 'Good') |
                         (data_all['GB_Flag'] == 'Bad')].copy()



# Create a Target variable
data_good_bad['Target'] = np.where(data_good_bad['GB_Flag'] == 'Good', 'G','B')

print(data_good_bad['Target'].value_counts())



# Derive Modelling variables
model_features = data_good_bad[['Gross_Annual_Income',
                                'Loan_Amount',
                                'Time_at_Address',
                                'Time_in_Employment',
                                'Time_with_Bank',
                                'SP_Number_Of_Searches_L6M',
                                'SP_Number_of_CCJs']].copy()

model_target = data_good_bad[['Target']].copy()

# print(model_features.columns.dtype)

# print(model_features['SP_Number_of_CCJs'].value_counts())


# Create Train-Test split
x_train, x_test,y_train, y_test = train_test_split(model_features,
                                                   model_target,
                                                   test_size = 0.2,
                                                   random_state = 23)


# Model development
final_model = LGBMClassifier(boosting_type='gbdt',
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

data_all['ML Score'] = final_model.predict(data_all_features, raw_score = True)



# Visualize Final Model
graph = lightgbm.create_tree_digraph(final_model, tree_index = 1)

graph.render(os.path.join(reports_path, r"LGBM - Tree 2"), format = 'png')


# Good Luck!