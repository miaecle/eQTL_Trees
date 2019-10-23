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
from seq_model import SimpleModel
from seq_model_trainer import SimpleTrainer
import torch as t

#=Settings=====================================================
data_name = 'assembled_balanced_dataset_123_Xy.pkl'
d_name = 'assembled_balanced_dataset_123.pkl'

normalized = True

split = 'random'
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

# Model Configuration
class Config:
  lr = 0.001
  batch_size = 64
  max_epoch = 5 # =1 when debug
  workers = 2
  gpu = True # use gpu or not

opt=Config()

v_feat_inds = [i for i in range(X.shape[1]) if feat_name[i].endswith('_v')]
g_feat_inds = [i for i in range(X.shape[1]) if feat_name[i].endswith('_g')]
p_feat_inds = [i for i in range(X.shape[1]) if feat_name[i].endswith('_p')]

### FULL MODEL ###
for feat_inds in [np.array(v_feat_inds),
                  np.array(g_feat_inds),
                  np.array(p_feat_inds),
                  np.array(v_feat_inds + g_feat_inds),
                  np.array(v_feat_inds + p_feat_inds),
                  np.array(g_feat_inds + p_feat_inds),
                  np.array(v_feat_inds + g_feat_inds + p_feat_inds)]:
  X_ = X[:, feat_inds]
  feat_name_ = [feat_name[i] for i in feat_inds]

  roc_scores = [[] for _ in range(len(cv_inds))]
  f1_scores = [[] for _ in range(len(cv_inds))]
  pr_scores = [[] for _ in range(len(cv_inds))]
  re_scores = [[] for _ in range(len(cv_inds))]

  np.random.seed(123)
  t.manual_seed(123)
  for fold_i, (train_inds, valid_inds) in enumerate(cv_inds):
    net = SimpleModel(len(feat_name_), gpu=opt.gpu)
    model = SimpleTrainer(net, opt, t.nn.NLLLoss())
    train_data = (X_[train_inds], 
                  y[train_inds])
    valid_data = (X_[valid_inds],
                  y[valid_inds])
    
    for i in range(20):
      model.train(train_data, n_epochs=2)
      valid_preds = np.concatenate(model.predict(valid_data), 0)
      roc_scores[fold_i].append(roc_auc_score(y[valid_inds], valid_preds[:, 1]))
      f1_scores[fold_i].append(f1_score(y[valid_inds], np.argmax(valid_preds, axis=1)))
      pr_scores[fold_i].append(precision_score(y[valid_inds], np.argmax(valid_preds, axis=1)))
      re_scores[fold_i].append(recall_score(y[valid_inds], np.argmax(valid_preds, axis=1)))

  inds = [np.argmax(s) for s in roc_scores]
  print(np.mean([s[i] for i, s in zip(inds, roc_scores)]))
  print(np.mean([s[i] for i, s in zip(inds, f1_scores)]))
  print(np.mean([s[i] for i, s in zip(inds, pr_scores)]))
  print(np.mean([s[i] for i, s in zip(inds, re_scores)]))