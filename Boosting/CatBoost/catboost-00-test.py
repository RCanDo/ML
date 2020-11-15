#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: CatBoost test
subtitle:
version: 1.0
type: example
keywords: [CatBoost, trees, boosting]
description: |
    Most basic example of CatBoost usage.
remarks:
todo:
sources:
    - title: CatBoost test
      link: https://catboost.ai/docs/installation/python-installation-test-catboost.html
file:
    usage:
        interactive: True   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    name: catboost-test.py
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
import numpy as np
from catboost import CatBoostRegressor

dataset = np.array([[1,4,5,6],[4,5,6,7],[30,40,50,60],[20,15,85,60]])
train_labels = [1.2,3.4,9.5,24.5]
model = CatBoostRegressor(learning_rate=1, depth=6, loss_function='RMSE')
fit_model = model.fit(dataset, train_labels)

fit_model.get_params()

dir(fit_model)
fit_model.predict(dataset)

dataset2 = np.array([[1,4,5,6],[4,5,6,7],[30,40,50,60],[20,15,85,60], [2,3,6,7]])
fit_model.predict(dataset2)

#%%
