import pickle
import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from xgboost import XGBClassifier
import argparse
import pandas as pd

pos_samples = 'LCL_pos_samples_180510.csv'
neg_samples = 'LCL_neg_samples_180510.csv'
inter_samples = None

pos_data_name = os.path.splitext(pos_samples)[0]
neg_data_name = os.path.splitext(neg_samples)[0]
if inter_samples is not None:
  inter_data_name = os.path.splitext(inter_samples)[0]

pos_data = pd.read_csv(pos_data_name + '.csv')
n_pos_samples = pos_data.shape[0]

neg_data = pd.read_csv(neg_data_name + '.csv')
n_neg_samples = neg_data.shape[0]

if inter_samples is not None:
  inter_data = pd.read_csv(inter_data_name + '.csv')
  n_inter_samples = inter_data.shape[0]


if inter_samples is not None:
  all_X = np.concatenate([np.array(pos_data), np.array(inter_data), np.array(neg_data)], axis=0)
  all_y = np.array([1]*n_pos_samples+[0]*n_inter_samples+[0]*n_neg_samples)
else:
  all_X = np.concatenate([np.array(pos_data), np.array(neg_data)], axis=0)
  all_y = np.array([1]*n_pos_samples+[0]*n_neg_samples)
n_samples = all_X.shape[0]



mapping = {}
for i, pair in enumerate(all_X):
  if pair[1] not in mapping.keys():
    mapping[pair[1]] = [(pair, all_y[i])]
  else:
    mapping[pair[1]].append((pair, all_y[i]))

outs = {}
for var in mapping.keys():
  pair_list = mapping[var]
  pairs = np.stack([p[0] for p in pair_list])
  ys = [p[1] for p in pair_list]
  ind = np.argmin(np.abs(pairs[:, 2]))
  outs[var] = ys[ind]

all_states = []
for i, pair in enumerate(all_X):
  # 1 indicates abnormal
  # 0 indicates normal
  all_states.append(1 - outs[pair[1]])
  

with open('./state_' + pos_data_name + neg_data_name + '.pkl', 'w') as f:
  pickle.dump(np.array(all_states), f)