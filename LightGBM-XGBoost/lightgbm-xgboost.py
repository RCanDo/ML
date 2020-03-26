#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: LightGB vs XGBoost
subtitle:
version: 1.0
type: tutorial/example
keywords: [LightGB, XGBoost, trees, boosting]
description: |
    Classification via LightGB vs XGBoost algorithms.
remarks:
    - work interactively (eg Spyder)
    - install lightgb, xgboost
todo:
    - ...
sources:
    - title: Which algorithm takes the crown: Light GBM vs XGBOOST?
      link: https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
      date: 2017-06-12
      authors:
          - fullname: Pranjal Khandelwal
      usage: |
          code and explanation taken from here;
          not only copy
    - title: Complete Machine Learning Guide to Parameter Tuning in Gradient Boosting (GBM) in Python
      link: https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
      author:
          - fullname: Aarshay Jain
      date: 2016-01-21
    - title: Complete Guide to Parameter Tuning in XGBoost with codes in Python
      link: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
      author:
          - fullname: Aarshay Jain
      date: 2016-03-01
    - title: XGBoost - Python API
      link: https://xgboost.readthedocs.io/en/latest/python/python_api.html
    - title: LightGBM - Python API
      link: https://lightgbm.readthedocs.io/en/latest/Python-API.html
file:
    usage:
        interactive: True   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    name: lightgbm-xgboost.py
    path: D:/Works/ML/LightGBM-XGBoost/
    date: 2020-03-16
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.com
              - arek@staart.pl
"""

#%%
cd D:/Works/ML/LightGBM-XGBoost/

#%%
import numpy as np
import pandas as pd

#%%
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

pd.set_option('display.large_repr', 'truncate')
pd.set_option('display.max_columns', 0)

#%%
# conda install lightgbm
import lightgbm as lgb
# conda install py-xgboost
import xgboost as xgb

#%%
data_raw = pd.read_csv('D:/Data/UCI-MLRepo/Adult/adult.data.csv', header=None, sep=", ")
# see 'D:/Data/UCI-MLRepo/Adult/adult.names.txt'
columns = [
'age', # continuous.
'workclass', # Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
'fnlwgt', # continuous.
'education', # Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
'education_num', # continuous.
'marital_status', # Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
'occupation', # Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
'relationship', # Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
'race', # White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
'sex', # Female, Male.
'capital_gain', # continuous.
'capital_loss', # continuous.
'hours_per_week', # continuous.
'native_country', # United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
'income' # >50K, <=50K
]
data_raw.columns = columns
data_raw.head()

data_raw.dtypes

#%% EDA
data_raw.count()
data_raw.shape    # (32561, 15)
data_raw.info()

data_raw['income'].value_counts()

#%% we shall work on the copy of raw data
data = data_raw.copy()
data.info()

#%%
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labenc = LabelEncoder()
labenc.fit(data_raw['income'])

labenc.classes_

#%%
income_labels = labenc.transform(data_raw['income'])

pd.Series(income_labels).value_counts()   #!

data.income = income_labels
data.income.value_counts()

#%% finding out categorical variables
data.dtypes         # 'income' was already changed
type(data.dtypes)   # pd.Series

cat_vars = data.columns[data.dtypes == 'object']

#%% searching for improper level names
for var in cat_vars:
    print(var)
    print(data_raw[var].unique())

#%% freq tables for all categoricals
for var in cat_vars:
    print(var.center(30, "-"))
    print(data_raw[var].value_counts())
    print()

#%% One Hot Encoding of the Categorical features
#%%
import functools as ft
import operator as op
import time
from typing import List

from rcando.ak.builtin import timeit

def cat2onehot(data: pd.core.frame.DataFrame, varname: str) -> pd.core.frame.DataFrame:
    """replacing variable varname with its one-hot encoding"""
    data_cat = pd.get_dummies(data[varname])  # DataFrame of one-hot encoding of a variable
    data = data.drop(varname, axis=1)
    data = pd.concat((data, data_cat), axis=1)
    return data

@timeit
def cat2onehot_list_func(data: pd.core.frame.DataFrame, varnames: List[str]) -> pd.core.frame.DataFrame:
    data1 = ft.reduce(cat2onehot, cat_vars, data)
    return data1

@timeit
def cat2onehot_list_loop(data: pd.core.frame.DataFrame, varnames: List[str]) -> pd.core.frame.DataFrame:
    data1 = data.copy()
    for col in varnames:
        df_oh = pd.get_dummies(data1[col])
        data1.drop(col, axis=1, inplace=True)
        data1 = pd.concat((data1, df_oh), axis=1)
    return data1

#%%
data1 = cat2onehot_list_func(data, cat_vars)   # 0.10935
data1 = cat2onehot_list_loop(data, cat_vars)   # 0.10935
data1.info()
data1.columns
data1.sum(axis=0)
data1.shape   # (32561, 109)

data = data1.copy()
del data1

#%%
#%% duplicate columns

_, i = np.unique(data.columns, return_index=True)
_
i
len(i)  # 107

[k for k in np.arange(109) if k not in i]  # [39, 67]
data.columns[[39, 67]]
# '?', '?'

# simpler:
colcounts = data.columns.value_counts()
colcounts[colcounts > 1]   # ?    3

# origin of duplicates; see also "searching for improper level names" cell above
[var for var in cat_vars if '?' in data_raw[var].values]   # or .unique() or .to_numpy()
# ['workclass', 'occupation', 'native_country']

data_columns = data.columns.values
np.where(data_columns == '?')
data_columns[data_columns == '?']

data_columns[data_columns == '?'] = \
    [var + "_unknown" for var in cat_vars if '?' in data_raw[var].values]

'?' in data_columns    # False OK

data.columns = data_columns
'?' in data.columns.values   # False OK

#%%
data.info()

#%%
#%%
yy = data['income']
xx = data.drop('income', axis=1)

yy.count()
xx.count()

# Imputing missing values in our target variable -- no need for this here
yy.fillna(yy.mode()[0], inplace=True)

#%%
from sklearn.model_selection import train_test_split

xx_train, xx_test, yy_train, yy_test = train_test_split(xx, yy, test_size=.3)

xx_train.shape  # 22792, 108
xx_test.shape   # 9769, 108
yy_train.shape  # (22792,)
yy_test.shape   # (9769,)

all(xx_test.index == yy_test.index)

#%%
#%% xgboost
# https://xgboost.readthedocs.io/en/latest/python/python_api.html

dir(xgb)
help(xgb)

""" this is big pain:
turning pd.DataFrame into some internal binary format:
- we loose e.g. .index from data
- after all it should be internal procedure hidden from the user
"""
d_train = xgb.DMatrix(xx_train, label=yy_train)
d_test = xgb.DMatrix(xx_test, label=yy_test)
d_test0 = xgb.DMatrix(xx_test)

d_train.index   #! AttributeError: 'DMatrix' object has no attribute 'index'

#%%
parameters = {'max_depth':7, 'eta':1, 'silent':1, 'objective':'binary:logistic',
              'eval_metric':'auc', 'learning_rate':.05}

evallist = [(d_test, 'eval'), (d_train, 'train')]

#%%

@timeit
def xgb_train():
    return xgb.train(parameters, d_train, num_boost_round=50)

xg = xgb_train()  # Execution time: 0.87089

#%%

dir(xg)
yy_pred0_score = xg.predict(d_test0)
yy_pred0 = (yy_pred0_score >= .5) * 1

yy_pred0.index  # no index! it's numpy! not a pd.Series
    #! AttributeError: 'numpy.ndarray' object has no attribute 'index'

yy_pred_score = xg.predict(d_test)
yy_pred = (yy_pred_score >= .5) * 1
yy_pred

yy_pred.index  #! AttributeError: 'numpy.ndarray' object has no attribute 'index'

all(yy_pred0 == yy_pred)   # 0, ok they're the same
sum(yy_pred != yy_test)    # 1308
sum(yy_pred != yy_test) / len(yy_test)   # err_rate = 0.133893
sum(yy_pred == yy_test) / len(yy_test)   # accuracy = 0.866107

#%%
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

accuracy_score(yy_test, yy_pred)       # 0.866107   exactly the same
roc_auc_score(yy_test, yy_pred_score)  # 0.9174  -- very high!!!

#%% https://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc
#
roc_auc_score(yy_test, yy_pred)        # 0.769 -- this usage is wrong !!!

yy_pred_pd = pd.Series(yy_pred, index=yy_test.index)
all(yy_pred_pd.index == yy_test.index)  # True, OK

np.random.seed(1)
idx = np.random.permutation(yy_test.index)
# idx = np.random.permutation(idx)
all(idx == yy_test.index)   # False, OK

accuracy_score(yy_test[idx], yy_pred_pd[idx])
roc_auc_score(yy_test[idx], yy_pred_pd[idx])

fpr, tpr, tresh = roc_curve(yy_test, yy_pred, pos_label=2)

import matplotlib.pyplot as plt

plt.plot(fpr, tpr)
plt.show()

#%%

sum(yy_pred > yy_test)  # 350
sum(yy_pred < yy_test)  # 958

yy_pred_2 = (yy_pred_score > .3) * 1

accuracy_score(yy_test, yy_pred_2)
roc_auc_score(yy_test, yy_pred_2)

for tresh in np.arange(.1, 1, .1):
    yy_pred_ = (yy_pred_score > tresh) * 1
    print("   {:1.1f}".format(tresh))
    print('accuracy: {:3.4f}'.format(accuracy_score(yy_test, yy_pred_)))
    print('   auroc: {:3.4f}'.format(roc_auc_score(yy_test, yy_pred_)))
    print()


#%%
#%%  LightGBM
#%%

dset_train = lgb.Dataset(xx_train, label=yy_train)
dset_test = lgb.Dataset(xx_test, label=yy_test)
dset_test0 = lgb.Dataset(xx_test)

param = {'num_leaves': 150, 'objective': 'binary', 'max_depth': 7, 'learning_rate': .05, 'max_bin': 200 }
param['metric'] = ['auc', 'binary_logloss']

#%%
@timeit
def lgb_train():
    return lgb.train(param, dset_train, num_boost_round=50)

lg = lgb_train()  # Execution time: 0.71374

#%%
yy_pred = (lg.predict(xx_test) >= .5) * 1

sum(yy_pred != yy_test)  # 1320
sum(yy_pred != yy_test) / len(yy_test)   # 0.1351213
sum(yy_pred == yy_test) / len(yy_test)   # 0.8648787

#%%
accuracy_lgb = accuracy_score(yy_test, yy_pred)
accuracy_lgb    # 0.8648787 the same

#%%



#%%



#%%



#%%


