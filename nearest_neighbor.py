# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from xgboost import XGBClassifier
import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    description='random_K_fold')
parser.add_argument(
    '-n',
    action='append',
    dest='neg',
    default=[],
    help='neg')
parser.add_argument(
    '-p',
    action='append',
    dest='pos',
    default=[],
    help='pos')

args = parser.parse_args()
pos_samples = './LCL_pos_samples_180124.csv'
neg_samples = './LCL_neg_samples_180210.csv'
inter_samples = './LCL_inter_samples_180210.csv'

pos_data_name = os.path.splitext(pos_samples)[0]
neg_data_name = os.path.splitext(neg_samples)[0]
inter_data_name = os.path.splitext(inter_samples)[0]

pos_data = pd.read_csv(pos_data_name + '.csv')
n_pos_samples = pos_data.shape[0]

neg_data = pd.read_csv(neg_data_name + '.csv')
n_neg_samples = neg_data.shape[0]

inter_data = pd.read_csv(inter_data_name + '.csv')
n_inter_samples = inter_data.shape[0]

all_X = np.concatenate([np.array(pos_data), np.array(inter_data), np.array(neg_data)], axis=0)
all_y = np.array([1]*n_pos_samples + [0]*n_inter_samples + [0]*n_neg_samples)

inds = np.arange(n_pos_samples + n_inter_samples + n_neg_samples)
np.random.seed(123)
np.random.shuffle(inds)
X = all_X[inds, :]
y = all_y[inds]

pos_inds = np.where(y==1)[0]
neg_inds = np.where(y==0)[0]
K = 10
pos_valid = np.linspace(0, len(pos_inds), K+1, dtype=int)
neg_valid = np.linspace(0, len(neg_inds), K+1, dtype=int)
full_scores = []

for i in range(K):
    valid_inds = pos_inds[pos_valid[i]:pos_valid[i+1]]
    valid_inds = np.concatenate([valid_inds,neg_inds[neg_valid[i]:neg_valid[i+1]]])
    valid_inds.sort()
    train_inds = np.concatenate([pos_inds[:pos_valid[i]], pos_inds[pos_valid[i+1]:]])
    train_inds = np.concatenate([train_inds, neg_inds[:neg_valid[i]], neg_inds[neg_valid[i+1]:]])
    train_inds.sort()
    
    train_y = y[train_inds]
    train_X = X[train_inds]
    valid_y = y[valid_inds]
    valid_X = X[valid_inds]

    valid_y_predictions = []
    for sample in valid_X:
      same_gene_pair_inds = np.where(train_X[:, 0] == sample[0])[0]
      same_gene_pairs = train_X[same_gene_pair_inds, :]
      if len(same_gene_pair_inds) > 0:
        y_pred = train_y[same_gene_pair_inds[np.argmin(np.abs(np.array(same_gene_pairs[:, 2], dtype=int) - sample[2]))]]
        valid_y_predictions.append(y_pred)
      else:
        valid_y_predictions.append(np.random.randint(0, 2))
    full_scores.append((accuracy_score(valid_y, valid_y_predictions), 
                        precision_score(valid_y, valid_y_predictions),
                        recall_score(valid_y, valid_y_predictions),
                        f1_score(valid_y, valid_y_predictions)))
    
print(np.mean(np.array(full_scores), axis=0))