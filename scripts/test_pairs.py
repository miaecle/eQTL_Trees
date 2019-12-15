#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:14:29 2019

@author: zqwu
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.preprocessing import normalize

from generate_X import feat_name
from split import random_split, position_split, maf_split, chromosome_split
from cross_validation import cv, cv_validate
from copy import deepcopy
sns.set(style="whitegrid")

test_data = pickle.load(open('../Data/test_pairs.pkl', 'r'))
test_X, test_y = pickle.load(open('../Data/test_pairs_Xy.pkl', 'r'))

X = test_X[np.where(test_y == 1)[0]]
v_feat_inds = [i for i in range(X.shape[1]) if feat_name[i].endswith('_v')]
g_feat_inds = [i for i in range(X.shape[1]) if feat_name[i].endswith('_g')]
p_feat_inds = [i for i in range(X.shape[1]) if feat_name[i].endswith('_p')]
X_v = X[:, np.array(v_feat_inds)]
X_g = X[:, np.array(g_feat_inds)]
X_p = X[:, np.array(p_feat_inds)]

models = pickle.load(open('./random_assembled_balanced_dataset_123_Xy_models.pkl', 'r'))
full_models = models['FULL']
v_models = models['VARIANT']
g_models = models['GENE']
p_models = models['PAIR']
p_v_models = models['PAIR+VARIANT']
p_g_models = models['PAIR+GENE']
v_g_models = models['VARIANT+GENE']

####################################################
i = 0
model = full_models[i]
valid_y_pred = model.predict_proba(X)
full_pred = deepcopy(valid_y_pred)
print(np.mean(valid_y_pred[:, 1]))
print(np.sum(valid_y_pred[:, 1] > 0.5001))
model = v_models[i]
valid_y_pred = model.predict_proba(X_v)
print(np.mean(valid_y_pred[:, 1]))
print(np.sum(valid_y_pred[:, 1] > 0.5001))
model = g_models[i]
valid_y_pred = model.predict_proba(X_g)
gene_pred = deepcopy(valid_y_pred)
print(np.mean(valid_y_pred[:, 1]))
print(np.sum(valid_y_pred[:, 1] > 0.5001))
model = p_models[i]
valid_y_pred = model.predict_proba(X_p)
pair_pred = deepcopy(valid_y_pred)
print(np.mean(valid_y_pred[:, 1]))
print(np.sum(valid_y_pred[:, 1] > 0.5001))
model = p_v_models[i]
valid_y_pred = model.predict_proba(np.concatenate([X_v, X_p], axis=1))
print(np.mean(valid_y_pred[:, 1]))
print(np.sum(valid_y_pred[:, 1] > 0.5001))
model = p_g_models[i]
valid_y_pred = model.predict_proba(np.concatenate([X_g, X_p], axis=1))
print(np.mean(valid_y_pred[:, 1]))
print(np.sum(valid_y_pred[:, 1] > 0.5001))
model = v_g_models[i]
valid_y_pred = model.predict_proba(np.concatenate([X_v, X_g], axis=1))
print(np.mean(valid_y_pred[:, 1]))
print(np.sum(valid_y_pred[:, 1] > 0.5001))

# Prediction distribution for test samples
data_name = 'assembled_balanced_dataset_123_Xy.pkl'
d_name = 'assembled_balanced_dataset_123.pkl'
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
y = all_y[inds]
dat = [all_dat[i] for i in inds]
cv_inds = random_split(X, y, dat, K=10)

train_inds, valid_inds = cv_inds[0]
train_y = y[train_inds]
train_X = X[train_inds]
valid_y = y[valid_inds]
valid_X = X[valid_inds]
i = 0
original_pred = full_models[i].predict_proba(valid_X)
original_pos_pred = original_pred[np.where(valid_y == 1)]

data_val = np.concatenate([original_pos_pred[:, 1],
                           full_pred[:, 1],
                           pair_pred[:, 1],
                           gene_pred[:, 1]])
data_label = ['full - \npositive\nvalidation\nsamples'] * len(original_pos_pred) + ['full'] * len(full_pred) + ['intermediate'] * len(pair_pred) + ['gene'] * len(gene_pred)
df = pd.DataFrame({'val': data_val, 'label': data_label})

plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
sns.violinplot(x='val',
               y='label',
               data=df,
               palette=[(0.7, 0.9, 0.8),
                        (0.4, 0.7607843137254902, 0.6470588235294118),
                        (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
                        (0.5529411764705883, 0.6274509803921569, 0.796078431372549)])
plt.xlim(-0.18, 1.18)
plt.vlines(0.5, -5, 5, 'k')
plt.ylabel('')
plt.xlabel('Predicted Prob.', fontdict={"size": 15})
plt.tight_layout()
plt.savefig('fig_distri.png', dpi=300)

# Rank analysis for test samples
model = full_models[0]

test_dat = test_data['pos'] + test_data['neg']
dist_thr_inds = [i for i, l in enumerate(test_X) if np.abs(int(l[305])) < 100000]
low_dist_vs = set([test_dat[i][1] for i in dist_thr_inds])
vs = set([l[1] for l in test_data['pos']]) & set([l[1] for l in test_data['neg']]) & low_dist_vs
vs_in_order = np.array([l[1] for l in test_data['pos']] + [l[1] for l in test_data['neg']])

pos_ct = 0
neg_ct = 0
ranks = []
lengths = []
for v in vs:
  inds = np.where(vs_in_order == v)[0]
  inds = np.array(list(set(inds) & set(dist_thr_inds)))
  X = test_X[inds, :]
  y = test_y[inds]
  if not 1 in y:
    print("No pos for %s" % v)
    continue
  if not 0 in y:
    print("No neg for %s" % v)
    continue
  pos_ct += np.where(y==1)[0].shape[0]
  neg_ct += np.where(y==0)[0].shape[0]
  pred = model.predict_proba(X)[:, 1]
  
  pred_order = list(np.argsort(-pred))
  for i, _y in enumerate(y):
    if _y == 1:
      rank = pred_order.index(i)
      ranks.append(rank)
      lengths.append(len(inds))

x = np.arange(max(lengths))
line1, _ = np.histogram(ranks, bins=x)
aver_line = np.zeros((max(lengths)))
for length in lengths:
  aver_line[:length] += 1./length

plt.bar(x[:-1]+1-0.4, line1, width=0.4, color=(0.4, 0.7607843137254902, 0.6470588235294118), label='full pred')
plt.bar(x+1., aver_line, width=0.4, color=(0.8, 0.4, 0.4), label='random baseline')
plt.ylabel("Count", fontdict={"size": 15})
plt.xlabel("Rank", fontdict={"size": 15})
plt.xticks(np.arange(1, 17))
plt.legend()
plt.tight_layout()
plt.savefig('fig_rank.png', dpi=300)