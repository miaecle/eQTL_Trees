#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:16:15 2019

@author: zqwu
"""

import pickle
import numpy as np
from xgboost import XGBClassifier
from cross_validation import cv_validate
from split import random_split, position_split, maf_split, chromosome_split

#=Settings=====================================================
data_name = 'assembled_balanced_dataset_123_Xy.pkl'
d_name = 'assembled_balanced_dataset_123.pkl'

model_ins = XGBClassifier(n_estimators=900, max_depth=5)
#==============================================================

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

non_nearest_neighbor_vars = set(pickle.load(open('../Data/assembled_balanced_dataset_123_nonneighbor_vars.pkl', 'rb')))

#==============================================================
test_dat = pickle.load(open('../Data/assembled_balanced_dataset_123_var_neighbor_pairs.pkl', 'rb'))
test_dat = test_dat['pos'] + test_dat['neg']
test_X, test_y = pickle.load(open('../Data/assembled_balanced_dataset_123_var_neighbor_pairs_Xy.pkl', 'rb'))
#==============================================================

cv_inds = position_split(X, y, dat, K=10)
test_inds = []
for inds in cv_inds:
  valid_vars = set([dat[i][1] for i in inds[1]])
  test_selected_inds = [i for i, d in enumerate(test_dat) if d[1] in valid_vars and d[1] in non_nearest_neighbor_vars]
  test_inds.append((None, test_selected_inds))

model_ins = XGBClassifier(n_estimators=900, max_depth=5)
scores = cv_validate(model_ins, 
                     test_X, 
                     test_y, 
                     test_inds, 
                     #'random_assembled_balanced_dataset_123_Xy')
                     'position_assembled_balanced_dataset_123_Xy')
for s in scores:
  print(np.mean(s, 0))


###
model = pickle.load(open('./random_assembled_balanced_dataset_123_Xy_models.pkl', 'r'))['FULL'][0]
test_y_pred = model.predict_proba(test_X)

prediction_results = {}
for i in test_inds:
  if test_dat[i][1] not in prediction_results:
    prediction_results[test_dat[i][1]] = []
  prediction_results[test_dat[i][1]].append((test_dat[i][2], test_y[i], test_y_pred[i][1]))

ct = 0
pos_scores = 0
neg_scores = 0
for var in prediction_results:
  pos_preds = [re[2] for re in prediction_results[var] if re[1] == 1]
  neg_preds = [re[2] for re in prediction_results[var] if re[1] == 0]
  if max(neg_preds) < min(pos_preds):
    ct += 1
  pos_scores += min(pos_preds)
  neg_scores += max(neg_preds)

print((pos_scores - neg_scores)/len(prediction_results))