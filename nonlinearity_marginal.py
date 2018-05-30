#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 09:54:25 2018

@author: zqwu
"""
import pickle
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from xgboost import XGBClassifier
#from simple_model import NNmodel
import argparse
import copy

pos_samples = './LCL_pos_samples_180124.csv'
neg_samples = './LCL_neg_samples_180210.csv'
inter_samples = './LCL_inter_samples_180210.csv'

model_name = 'xgb'
use_dis = False


if model_name == 'rf':
  model_ins = RandomForestClassifier(n_estimators=1000)
elif model_name == 'logreg':
  model_ins = LogisticRegression()
elif model_name == 'xgb':
  model_ins = XGBClassifier(n_estimators=500)

pos_data_name = os.path.splitext(pos_samples)[0]
neg_data_name = os.path.splitext(neg_samples)[0]
if not inter_samples is None:
  inter_data_name = os.path.splitext(inter_samples)[0]

with open(pos_data_name+'.pkl', 'r') as f:
  pos_data = pickle.load(f)
  n_pos_samples = pos_data.shape[0]
with open(neg_data_name+'.pkl', 'r') as f:
  neg_data = pickle.load(f)
  n_neg_samples = neg_data.shape[0]
if not inter_samples is None:
  with open(inter_data_name+'.pkl', 'r') as f:
    inter_data = pickle.load(f)
    n_inter_samples = inter_data.shape[0]

if not inter_samples is None:
  all_X = np.concatenate([pos_data, inter_data, neg_data], axis=0)
  all_y = np.array([1]*n_pos_samples + [0]*n_inter_samples + [0]*n_neg_samples)
else:
  all_X = np.concatenate([pos_data, neg_data], axis=0)
  all_y = np.array([1]*n_pos_samples + [0]*n_neg_samples)

if not use_dis:
  all_X[:, 93] = 0
  all_X[:, 126] = 0

assert all_X.shape[0] == all_y.shape[0]
n_samples = all_y.shape[0]

if not use_dis:
  all_X[:, 93] = 0
  all_X[:, 126] = 0

inds = np.arange(all_X.shape[0])
np.random.shuffle(inds)
X = all_X[inds]
y = all_y[inds]
X = normalize(X, axis=0)


model = copy.deepcopy(model_ins)
model.fit(X, y)

gene_selection = [214, 149, 131, 147, 192, 146, 160, 199, 194, 198, 144]
#variant_selection = [24, 44, 99, 45, 61, 43, 54]
variant_selection = [23, 43, 98, 44, 60, 42, 53]

results = {}
for gene_f in gene_selection:
  for variant_f in variant_selection:
    test_pair = (gene_f, variant_f)
    nonlinearity_scores = []
    for id_sample in range(1000):
      sample = X[id_sample, :]
      switch_pairs = []
      remain_inds = np.arange(len(X))
      np.random.shuffle(remain_inds)
      i = 0
      while len(switch_pairs) < 10000:
        #if X[i, test_pair[0]] != sample[test_pair[0]] and \
        #   X[i, test_pair[1]] != sample[test_pair[1]]:
        switch_pairs.append(X[remain_inds[i]])
        i += 1
      switch_pairs = np.array(switch_pairs)
      control_1 = copy.deepcopy(switch_pairs)
      control_2 = copy.deepcopy(switch_pairs)
      control_1[:, test_pair[0]] = sample[test_pair[0]]
      control_1[:, test_pair[1]] = sample[test_pair[1]]
      
      sample_pred = model.predict_proba(np.expand_dims(sample, 0))[:, 1]
      pred = model.predict_proba(switch_pairs)[:, 1]
      pred_control_1 = model.predict_proba(control_1)[:, 1]
      pred_control_2 = model.predict_proba(control_2)[:, 1]


      nonlinear = pred + sample_pred - pred_control_1 - pred_control_2
      nonlinearity_scores.append(np.mean(np.abs(nonlinear)))
    results[test_pair] = np.mean(nonlinearity_scores)
    print(test_pair)
    print(np.mean(nonlinearity_scores))