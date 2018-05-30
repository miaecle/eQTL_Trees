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

model_name = 'rf'
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
variant_selection = [24, 44, 99, 45, 61, 43, 54]

results = {}
for gene_f in gene_selection:
  for variant_f in variant_selection:
    test_pair = (gene_f, variant_f)
    choices1 = np.unique(X[:, test_pair[0]])
    if len(choices1) > 200:
      choices1 = np.linspace(np.min(choices1), np.max(choices1), 200)
    choices2 = np.unique(X[:, test_pair[1]])
    if len(choices2) > 200:
      choices2 = np.linspace(np.min(choices2), np.max(choices2), 200)
    total_len = len(choices1) * len(choices2)
    c1, c2 = np.meshgrid(choices2, choices1)
    c1 = c1.flatten()
    c2 = c2.flatten()
    nonlinearity_scores = []
    for id_sample in range(1000):
      sample = X[id_sample, :]
      id_i = np.argmin(np.abs(choices1 - sample[test_pair[0]]))
      id_j = np.argmin(np.abs(choices2 - sample[test_pair[1]]))
      targets = np.stack([sample] * total_len, axis=0)
      targets[:, test_pair[0]] = c1
      targets[:, test_pair[1]] = c2
      pred = model.predict_proba(targets)[:, 1]
      pred = pred.reshape((len(choices1), len(choices2)))
      nonlinear = pred + pred[id_i, id_j] - pred[id_i:(id_i+1), :] - pred[:, id_j:(id_j+1)]
      nonlinearity_scores.append(np.mean(np.abs(nonlinear)))
    results[test_pair] = np.mean(nonlinearity_scores)
    print(test_pair)
    print(np.mean(nonlinearity_scores))