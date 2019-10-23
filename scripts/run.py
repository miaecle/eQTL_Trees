# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

from generate_X import feat_name, normalize
from split import random_split, position_split, maf_split, chromosome_split
from cross_validation import cv, cv_validate

#=Settings=====================================================
data_name = 'assembled_balanced_dataset_123_Xy.pkl'
d_name = 'assembled_balanced_dataset_123.pkl'

normalized = False

#model_ins = RandomForestClassifier(n_estimators=500)
#model_ins = LogisticRegression()
model_ins = XGBClassifier(n_estimators=900, max_depth=5)
split = 'random'
model_name = split + '_' + os.path.splitext(data_name)[0] + 'xgb'
#==============================================================

with open('../Data/' + data_name, 'r') as f:
  all_X, all_y = pickle.load(f)
with open('../Data/' + d_name, 'r') as f:
  all_dat = pickle.load(f)
  all_dat = all_dat['pos'] + all_dat['neg']

# Shuffle and normalize features
inds = np.arange(all_X.shape[0])
np.random.seed(123)
np.random.shuffle(inds)
X = all_X[inds, :]
if normalized:
  X, feat_name = normalize(X, feat_name)
y = all_y[inds]
dat = [all_dat[i] for i in inds]

if split == 'random':
  cv_inds = random_split(X, y, dat, K=10)
elif split == 'position':
  cv_inds = position_split(X, y, dat, K=10)
elif split == 'maf':
  cv_inds = maf_split(X, y, dat)
elif split == 'chromosome':
  cv_inds = chromosome_split(X, y, dat)

SCORES = cv(model_ins, X, y, cv_inds, model_name, feat_name=feat_name)