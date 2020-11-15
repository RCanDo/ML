#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: CatBoost Quick Start
subtitle:
version: 1.0
type: tutorial/examples
keywords: [CatBoost, trees, boosting]
description: |
    Most basic examples of CatBoost usage.
remarks:
todo:
sources:
    - title: CatBoost Quick Start
      link: https://catboost.ai/docs/concepts/python-quickstart.html
file:
    usage:
        interactive: True   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    name: catboost-01-quick_start.py
    path: D:/Works/ML/Boosting/
    date: 2020-10-25
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.com
              - arek@staart.pl
"""
#%%

#%%
cd ~/Works/ML/Boosting/

#%%
import numpy as np
#import pandas as pd

#%% CatBoostClassifier

from catboost import CatBoostClassifier, Pool

# initialize data
train_data = np.random.randint(0, 100, size=(100, 10))

train_labels = np.random.randint(0, 2, size=(100))

test_data = catboost_pool = Pool(train_data, train_labels)                     #???

model = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)
# train the model
model.fit(train_data, train_labels)
# make the prediction using the resulting model
preds_class = model.predict(test_data)
preds_proba = model.predict_proba(test_data)
print("class = ", preds_class)
print("proba = ", preds_proba)


#%% CatBoostRegressor

from catboost import Pool, CatBoostRegressor
# initialize data
train_data = np.random.randint(0, 100, size=(100, 10))

train_labels = np.random.randint(0, 1000, size=(100))

test_data = np.random.randint(0, 100, size=(50, 10))

# initialize Pool
train_pool = Pool(train_data, train_labels, cat_features=[0,2,5])

test_pool = Pool(test_data, cat_features=[0,2,5]) 

# specify the training parameters 
model = CatBoostRegressor(iterations=2, 
                          depth=2, 
                          learning_rate=1, 
                          loss_function='RMSE')
#train the model
model.fit(train_pool)
# make the prediction using the resulting model
preds = model.predict(test_pool)
print(preds)


#%% CatBoost
# Datasets can be read from input files. 
# For example, the Pool class offers this functionality.

import numpy as np
from catboost import CatBoost, Pool

# read the dataset

train_data = np.random.randint(0, 100, size=(100, 10))

train_labels = np.random.randint(0, 2, size=(100))

test_data = np.random.randint(0, 100, size=(50, 10))
                                
train_pool = Pool(train_data, train_labels)

test_pool = Pool(test_data) 
# specify training parameters via map

param = {'iterations':5}
model = CatBoost(param)
#train the model
model.fit(train_pool) 
# make the prediction using the resulting model
preds_class = model.predict(test_pool, prediction_type='Class')   
    #! IndexError: index 0 is out of bounds for axis 0 with size 0
model.predict(test_data, prediction_type='Class') 
    #! IndexError: index 0 is out of bounds for axis 0 with size 0
preds_proba = model.predict(test_pool, prediction_type='Probability')
preds_raw_vals = model.predict(test_pool, prediction_type='RawFormulaVal')
model.predict(test_pool)   # the same
model.predict(test_data)   # the same

print("Class", preds_class)
print("Proba", preds_proba)
print("Raw", preds_raw_vals)




#%%
