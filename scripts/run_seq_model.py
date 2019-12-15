# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn import preprocessing

from generate_X import feat_name
from split import random_split, position_split, maf_split, chromosome_split
from cross_validation import cv

from seq_model import SeqModel, TransformerModel
from seq_model_trainer import Trainer
import torch as t

data_name = 'assembled_balanced_dataset_123_Xy.pkl'
inter_data_name = 'assembled_balanced_dataset_123_Xy_inter_selected.pkl'
d_name = 'assembled_balanced_dataset_123.pkl'

split = 'random'

with open('../Data/' + data_name, 'r') as f:
  all_X, all_y = pickle.load(f)
with open('../Data/' + inter_data_name, 'r') as f:
  all_inter = pickle.load(f)
with open('../Data/' + d_name, 'r') as f:
  all_dat = pickle.load(f)
  all_dat = all_dat['pos'] + all_dat['neg']

# Shuffle and normalize features
inds = np.arange(all_X.shape[0])
np.random.seed(123)
np.random.shuffle(inds)
X = all_X[inds, :]
y = all_y[inds]
inter = [all_inter[i] for i in inds]
dat = [all_dat[i] for i in inds]

if split == 'random':
  cv_inds = random_split(X, y, dat, K=10)
elif split == 'position':
  cv_inds = position_split(X, y, dat, K=10)
elif split == 'maf':
  cv_inds = maf_split(X, y, dat)
elif split == 'chromosome':
  cv_inds = chromosome_split(X, y, dat)


# Preprocessing
static_feat_inds = [i for i in range(len(feat_name)) \
                 if feat_name[i].endswith('_v') or \
                    feat_name[i].endswith('_g') or \
                    feat_name[i].startswith('HiC') or \
                    feat_name[i].startswith('dist')]
X = X[:, np.array(static_feat_inds)]

for i in range(X.shape[1]):
  feat = X[:, i]
  if np.max(feat) > 5:
    X[:, i] = X[:, i] / np.std(feat)

running_sum2 = np.zeros((inter[0].shape[1]))
running_sum = np.zeros((inter[0].shape[1]))
lengths = 0
for item in inter:
  running_sum2 += np.square(item).sum(0)
  running_sum += item.sum(0)
  lengths += item.shape[0]
var = running_sum2/lengths - np.square(running_sum/lengths)
inter_std = np.sqrt(var)
multiplier = np.array([1/i_std if i_std > 1 else 1 for i_std in inter_std]).reshape((1, -1))
inter_normed = [item * multiplier for item in inter]

# Model Configuration
class Config:
  lr = 0.001
  batch_size = 8
  max_epoch = 5 # =1 when debug
  workers = 2
  gpu = True # use gpu or not

opt=Config()

#net = SeqModel(17, 320, gpu=opt.gpu)
net = TransformerModel(17, 320, gpu=opt.gpu)
model = Trainer(net, opt, t.nn.NLLLoss())

train_inds = cv_inds[0][0]
valid_inds = cv_inds[0][1]
train_data = (X[train_inds], 
              [inter_normed[i] for i in train_inds], 
              y[train_inds], 
              [dat[i] for i in train_inds])
valid_data = (X[valid_inds], 
              [inter_normed[i] for i in valid_inds], 
              y[valid_inds], 
              [dat[i] for i in valid_inds])
for i in range(10):
  print("Fold %d" % i)
  model.train(train_data, n_epochs=1)
  valid_preds = np.concatenate(model.predict(valid_data), 0)
  print(roc_auc_score(y[valid_inds], valid_preds[:, 1]))
  print(f1_score(y[valid_inds], valid_preds[:, 1] >= 0.5))
train_preds = model.predict(train_data)
valid_preds = model.predict(valid_data)