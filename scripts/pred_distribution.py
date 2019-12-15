#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 15:15:00 2019

@author: zqwu
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

from generate_X import feat_name
from split import random_split, position_split, maf_split, chromosome_split
from cross_validation import cv
sns.set(style="whitegrid")

### Settings ###
data_name = 'assembled_balanced_dataset_123_Xy.pkl'
d_name = 'assembled_balanced_dataset_123.pkl'

with open('../Data/' + data_name, 'r') as f:
  all_X, all_y = pickle.load(f)
with open('../Data/' + d_name, 'r') as f:
  all_dat = pickle.load(f)
  all_dat = all_dat['pos'] + all_dat['neg']

inds = np.arange(all_X.shape[0])
np.random.seed(123)
np.random.shuffle(inds)
X = all_X[inds, :]
y = all_y[inds]
dat = [all_dat[i] for i in inds]

# Gather model predictions: Random split
cv_inds = random_split(X, y, dat, K=10)
name = './random_assembled_balanced_dataset_123_Xy'
with open(name + "_models.pkl", 'r') as f:
  models = pickle.load(f)
full_models = models['FULL']
labels = []
preds = []
for i, (train_inds, valid_inds) in enumerate(cv_inds):
  valid_y = y[valid_inds]
  valid_X = X[valid_inds]
  model = full_models[i]
  valid_y_pred = model.predict_proba(valid_X)
  labels.append(valid_y)
  preds.append(valid_y_pred)
labels = np.concatenate(labels)
preds = np.concatenate(preds)[:, 1]

# Distribution of prediction values: Random split
plt.clf()
data_val = np.concatenate([preds[np.where(labels == 0)], preds[np.where(labels == 1)]])
data_label = ['Negative'] * np.where(labels == 0)[0].shape[0] + ['Positive'] * np.where(labels == 1)[0].shape[0]
df = pd.DataFrame({'val': data_val, 'label': data_label})
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
sns.violinplot(x='val',
               y='label',
               data=df,
               palette=[(0.4, 0.7607843137254902, 0.6470588235294118),
                        (0.9882352941176471, 0.5529411764705883, 0.3843137254901961)])
plt.xlim(-0.18, 1.18)
plt.ylabel('')
plt.xlabel('Predicted eQTL Prob.', fontdict={"size": 15})
plt.tight_layout()
plt.savefig('prediction_distri_random.png', dpi=300)

# Gather model predictions: Position split
cv_inds = position_split(X, y, dat, K=10)
name = './position_assembled_balanced_dataset_123_Xy'
with open(name + "_models.pkl", 'r') as f:
  models = pickle.load(f)
full_models = models['FULL']
labels = []
preds = []
for i, (train_inds, valid_inds) in enumerate(cv_inds):
  valid_y = y[valid_inds]
  valid_X = X[valid_inds]
  model = full_models[i]
  valid_y_pred = model.predict_proba(valid_X)
  labels.append(valid_y)
  preds.append(valid_y_pred)
labels = np.concatenate(labels)
preds = np.concatenate(preds)[:, 1]

# Distribution of prediction values: Position split
plt.clf()
data_val = np.concatenate([preds[np.where(labels == 0)], preds[np.where(labels == 1)]])
data_label = ['Negative'] * np.where(labels == 0)[0].shape[0] + ['Positive'] * np.where(labels == 1)[0].shape[0]
df = pd.DataFrame({'val': data_val, 'label': data_label})
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
sns.violinplot(x='val',
               y='label',
               data=df,
               palette=[(0.4, 0.7607843137254902, 0.6470588235294118),
                        (0.9882352941176471, 0.5529411764705883, 0.3843137254901961)])
plt.xlim(-0.18, 1.18)
plt.ylabel('')
plt.xlabel('Predicted eQTL Prob.', fontdict={"size": 15})
plt.tight_layout()
plt.savefig('prediction_distri_position.png', dpi=300)