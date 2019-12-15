#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:02:12 2019

@author: zqwu
"""
import numpy as np
import os
import pickle
from generate_X import feat_name, normalize
from split import random_split, position_split, maf_split, chromosome_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

ranking_dat = pickle.load(open('../Data/ranking_analysis.pkl', 'rb'))
ranking_X, ranking_y = pickle.load(open('../Data/ranking_analysis_Xy.pkl', 'rb'))

ranking_vs = set(l[1] for l in ranking_dat)
ranking_dat_vs = {v: [] for v in ranking_vs}
ranking_X_vs = {v: [] for v in ranking_vs}
ranking_scores_vs = {v: [] for v in ranking_vs}

for pair, _X in zip(ranking_dat, ranking_X):
  if pair[1] in ranking_vs:
    ranking_dat_vs[pair[1]].append(pair)
    ranking_X_vs[pair[1]].append(_X)

# Get original cv index
with open('../Data/assembled_balanced_dataset_123_Xy.pkl', 'rb') as f:
  all_X, all_y = pickle.load(f)
with open('../Data/assembled_balanced_dataset_123.pkl', 'rb') as f:
  all_dat = pickle.load(f)
  all_dat = all_dat['pos'] + all_dat['neg']
inds = np.arange(all_X.shape[0])
np.random.seed(123)
np.random.shuffle(inds)
X = all_X[inds, :]
y = all_y[inds]
dat = [all_dat[i] for i in inds]
cv_inds = random_split(X, y, dat, K=10)

# Gather prediction scores
models = pickle.load(open('./random_assembled_balanced_dataset_123_Xy_models.pkl', 'rb'))
for i, (train_inds, valid_inds) in enumerate(cv_inds):
  model = models["FULL"][i]
  valid_vs = set([dat[i][1] for i in valid_inds])
  for v in valid_vs & ranking_vs:
    _X = np.stack(ranking_X_vs[v])
    pred_score = model.predict_proba(_X)
    ranking_scores_vs[v] = pred_score[:, 1]

# Rankings
ranks1 = {i: 0 for i in range(50)} # Pred
ranks2 = {i: 0 for i in range(50)} # Distance
ranks3 = {i: 0 for i in range(50)} # Random
for v in ranking_vs:
  dists = np.array([np.abs(int(pair[2])) for pair in ranking_dat_vs[v]])
  scores_pred = -ranking_scores_vs[v]
  rank_pred = list(np.argsort(scores_pred))
  scores_dist = np.array([np.abs(int(pair[2])) for pair in ranking_dat_vs[v]])
  rank_dist = list(np.argsort(scores_dist))
  y_indexes = np.where(np.array([pair[6] <= 1e-6 for pair in ranking_dat_vs[v]]))[0]
  assert not len(scores_dist) == len(y_indexes)
  for y_index in y_indexes:
    ranks1[rank_pred.index(y_index)] += 1
    ranks2[rank_dist.index(y_index)] += 1
    for _i in range(len(scores_dist)):
      ranks3[_i] += 1./len(scores_dist)

line1 = [ranks1[i] for i in range(16)]
line2 = [ranks2[i] for i in range(16)]
line3 = [ranks3[i] for i in range(16)]

x = np.arange(16)
sns.set(style="whitegrid")
plt.bar(x+1-0.45, line1, width=0.3, color=(0.4, 0.7607843137254902, 0.6470588235294118), label='full pred')
plt.bar(x+1-0.15, line2, width=0.3, color=(0.13, 0.42, 0.71), label='distance baseline')
plt.bar(x+1.15, line3, width=0.3, color=(0.8, 0.4, 0.4), label='random baseline')
plt.ylabel("Count", fontdict={"size": 15})
plt.xlabel("Rank", fontdict={"size": 15})
plt.xticks(np.arange(1, 17))
plt.legend()
plt.tight_layout()
plt.savefig('fig_rank2.png', dpi=300)

