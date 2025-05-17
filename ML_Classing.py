#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 19:03:56 2023

@author: juliana
"""





# Preliminaries
data_path = r"/home/juliana/Документи/Experian - ML Study/Data"

data_file = r"BankCaseStudyData.csv"

reports_path = r"/home/juliana/Документи/Experian - ML Study/Reports"




# Import Python APIs
import numpy as np
import os
import pandas as pd
from sklearn.tree import plot_tree, _tree
from sklearn.ensemble import GradientBoostingClassifier



# Read Data
data_all = pd.read_csv(os.path.join(data_path, data_file),
                       keep_default_na = False,
                       encoding = 'utf8',
                       low_memory = False)



# Target
gb_flag_dict = {
    'Good' : 'G',
    'Bad' : 'B'
    }

data_all['GB_Flag'] = data_all['GB_Flag'].map(gb_flag_dict)



# Good Bad sample
data_gb = data_all[data_all['GB_Flag'].isin(['G','B'])].copy()



# Features and target
feature = data_gb['Bureau_Score'].to_numpy().reshape(-1, 1)

target = data_gb['GB_Flag']


# ML Model
model = GradientBoostingClassifier(n_estimators = 10,
                                   min_samples_leaf = 0.05)


model.fit(feature,target)

thresholds = np.zeros([0])

for i in range(model.n_estimators_):
    
    tree_1 = model.estimators_[i, 0]
    
    thresholds = np.append(thresholds,tree_1.tree_.threshold)

thresholds = thresholds[thresholds != _tree.TREE_UNDEFINED]

thresholds = np.unique(thresholds)


# Good Luck!