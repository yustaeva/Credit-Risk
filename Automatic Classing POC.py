#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 19:51:52 2023

@author: juliana
"""


# Automatic Classing - POC




# Preliminaries

data_path = r"/home/juliana/Документи/Experian - ML Study/Data"

data_file = r"BankCaseStudyData.csv"

reports_path = r"/home/juliana/Документи/Experian - ML Study/Reports"




# Import Python APIs
import numpy as np
import os
import pandas as pd
import lightgbm
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
model_feature = data_good_bad[['Time_with_Bank']].copy()

model_target = data_good_bad[['Target']].copy()



# Create Train-Test split
x_train, x_test,y_train, y_test = train_test_split(model_feature,
                                                   model_target,
                                                   test_size = 0.2,
                                                   random_state = 23)


# Model development
final_model = LGBMClassifier(boosting_type='gbdt',
                             num_leaves = 7,
                             max_depth = None,
                             n_estimators = 1,
                             min_child_samples = 1000)


# Train Final Model
final_model.fit(x_train, y_train)


# Visualize Final Model
graph = lightgbm.create_tree_digraph(final_model, tree_index = 0)

graph.render(os.path.join(reports_path, r"Automatic Classing - 3"), format = 'png')




# Good Luck!