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

pos_rate = float(sum(all_y))/len(all_y)

inds = np.arange(n_samples)
np.random.seed(123)
np.random.shuffle(inds)
X = all_X[inds, :]
y = all_y[inds]

pos_inds = np.where(y==1)[0]
neg_inds = np.where(y==0)[0]
K = 10
pos_valid = np.linspace(0, len(pos_inds), K+1, dtype=int)
neg_valid = np.linspace(0, len(neg_inds), K+1, dtype=int)

all_nn_features = []

mapping = {}
for i, pair in enumerate(X):
  if pair[0] not in mapping.keys():
    mapping[pair[0]] = [(pair, i, y[i])]
  else:
    mapping[pair[0]].append((pair, i, y[i]))


for j in range(K):
  valid_inds = pos_inds[pos_valid[j]:pos_valid[j+1]]
  valid_inds = np.concatenate([valid_inds,neg_inds[neg_valid[j]:neg_valid[j+1]]])
  valid_inds.sort()
  train_inds = np.concatenate([pos_inds[:pos_valid[j]], pos_inds[pos_valid[j+1]:]])
  train_inds = np.concatenate([train_inds, neg_inds[:neg_valid[j]], neg_inds[neg_valid[j+1]:]])
  train_inds.sort()
  
  
  nns = []
  for pair in all_X:
    potential_pairs = [p for p in mapping[pair[0]] if p[1] not in valid_inds and p[0][1] != pair[1]]
    if len(potential_pairs) == 0:
      nns.append((np.random.rand() < pos_rate * 1, 1))
    else:
      ind = np.argmin(np.abs(np.stack([p[0] for p in potential_pairs])[:, 2] - pair[2]))
      nns.append((potential_pairs[ind][2], 0))
  all_nn_features.append(np.array(nns))
with open('./nn_' + pos_data_name + neg_data_name + '.pkl', 'w') as f:
  pickle.dump(all_nn_features, f)