#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 20:50:39 2019

@author: zqwu
"""
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.preprocessing import normalize

from generate_X import feat_name
from split import random_split, position_split, maf_split, chromosome_split
from cross_validation import cv

#=Settings=====================================================
data_name = 'assembled_balanced_dataset_123_Xy.pkl'
d_name = 'assembled_balanced_dataset_123.pkl'
normalized = False

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
  X = normalize(X, axis=0)
y = all_y[inds]
dat = [all_dat[i] for i in inds]

#### Distribution of var-gene distances ####
neg_dists = all_X[np.where(all_y == 0)[0]][:, 305]
pos_dists = all_X[np.where(all_y == 1)[0]][:, 305]
bins = np.arange(-300000, 300000, 20000)
plt.hist(pos_dists, bins=bins, color=(1, 0, 0, 0.3), label='eQTL')
plt.hist(neg_dists, bins=bins, color=(0, 0, 1, 0.3), label='Non-eQTL')
plt.legend()
plt.xlabel('Variant-Gene Distance (bp)')
plt.ylabel('Count')
plt.savefig('Distri_distance.png', dpi=300)

#### Feature effect: GC ##########
model = pickle.load(open('./random_assembled_balanced_dataset_123_Xy_models.pkl', 'r'))['FULL'][0]
test_feat = 'GC_p'
test_feat_ind = feat_name.index(test_feat)

plt.clf()
plt.hist(X[:, test_feat_ind], bins=100, color=(0.4, 0.7607843137254902, 0.6470588235294118))
plt.ylim(0, 4000)
plt.ylabel("Count", fontsize=16)
plt.xlabel("GC_p", fontsize=16)
plt.tight_layout()
plt.savefig("GC_p_distri.png", dpi=300)

# plt.clf()
# plt.hist(X[np.where(y==1)[0]][:, test_feat_ind], 
#          bins=100,
#          color=(1, 0, 0, 0.5),
#          label='Positive')
# plt.hist(X[np.where(y==0)[0]][:, test_feat_ind], 
#          bins=100,
#          color=(0, 0, 1, 0.5),
#          label='negative')
# plt.ylabel("Count", fontsize=16)
# plt.xlabel("GC_p", fontsize=16)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.savefig("GC_p_distri_sep.png", dpi=300)

unique_values = sorted(np.unique(X[:, test_feat_ind]))
start_point = int(len(unique_values) * 0.05)
end_point = int(len(unique_values) * 0.95)
unique_values = unique_values[start_point:(end_point+1)]
sample_values = [unique_values[i*(len(unique_values)//300)] for i in range(300)]

np.random.seed(123)
test_inds = np.random.choice(inds, size=(5000,))
preds = []
for ind in test_inds:
  X_sample = X[ind]
  X_sample = np.stack([X_sample] * len(sample_values), 0)
  X_sample[:, test_feat_ind] = np.array(sample_values)
  pred = model.predict_proba(X_sample)
  preds.append(pred[:, 1])
preds = np.mean(np.array(preds), 0)
plt.clf()
plt.plot(sample_values, preds, '.-', linewidth=3, color=(0.4, 0.7607843137254902, 0.6470588235294118))
plt.xlim(0.36, 0.55)
plt.xlabel("GC_p", fontsize=16)
plt.ylabel("Predicted eQTL Prob.", fontsize=16)
plt.tight_layout()
plt.savefig("GC_p_change.png", dpi=300)

np.random.seed(123)
short_dist_inds = np.where(np.abs(X[:, 305]) < 50000)[0]
medium_dist_inds = np.array(list(set(np.where(np.abs(X[:, 305]) > 50000)[0]) & \
                                 set(np.where(np.abs(X[:, 305]) < 300000)[0])))
long_dist_inds = np.where(np.abs(X[:, 305]) > 300000)[0]
test_inds_short_dist = np.random.choice(short_dist_inds, size=(5000,))
test_inds_medium_dist = np.random.choice(medium_dist_inds, size=(5000,))
test_inds_long_dist = np.random.choice(long_dist_inds, size=(5000,))
preds = []
for ind in test_inds_short_dist:
  X_sample = X[ind]
  X_sample = np.stack([X_sample] * len(sample_values), 0)
  X_sample[:, test_feat_ind] = np.array(sample_values)
  pred = model.predict_proba(X_sample)
  preds.append(pred[:, 1])
preds_short_dist = np.mean(np.array(preds), 0)
preds = []
for ind in test_inds_medium_dist:
  X_sample = X[ind]
  X_sample = np.stack([X_sample] * len(sample_values), 0)
  X_sample[:, test_feat_ind] = np.array(sample_values)
  pred = model.predict_proba(X_sample)
  preds.append(pred[:, 1])
preds_medium_dist = np.mean(np.array(preds), 0)
preds = []
for ind in test_inds_long_dist:
  X_sample = X[ind]
  X_sample = np.stack([X_sample] * len(sample_values), 0)
  X_sample[:, test_feat_ind] = np.array(sample_values)
  pred = model.predict_proba(X_sample)
  preds.append(pred[:, 1])
preds_long_dist = np.mean(np.array(preds), 0)
plt.clf()
plt.plot(sample_values, preds_short_dist, '.-', linewidth=3, label='<50kb var-gene dist', color=(0.5, 0.7, 0.9))
plt.plot(sample_values, preds_medium_dist, '.-', linewidth=3, label='50kb~300kb var-gene dist', color=(0.4, 0.7607843137254902, 0.6470588235294118))
plt.plot(sample_values, preds_long_dist, '.-', linewidth=3, label='>300kb var-gene dist', color=(0.5, 0.3, 0.45))
plt.legend(fontsize=12)
plt.xlim(0.36, 0.55)
plt.xlabel("GC_p", fontsize=16)
plt.ylabel("Predicted eQTL Prob.", fontsize=16)
plt.tight_layout()
plt.savefig("GC_p_change_dist.png", dpi=300)

#### Feature effect: H3K27me3 ##########
model = pickle.load(open('./random_assembled_balanced_dataset_123_Xy_models.pkl', 'r'))['FULL'][0]
test_feat = 'H3K27me3_p'
test_feat_ind = feat_name.index(test_feat)

plt.clf()
plt.hist(X[:, test_feat_ind], bins=np.arange(0, 10, 0.1), color=(1., 0.55, 0.38))
plt.xlim(-1, 10)
plt.ylabel("Count", fontsize=16)
plt.xlabel("H3K27me3_p", fontsize=16)
plt.tight_layout()
plt.savefig("H3K27me3_p_distri.png", dpi=300)

# plt.clf()
# plt.hist(X[np.where(y==1)[0]][:, test_feat_ind], 
#          bins=np.arange(0, 10, 0.1), 
#          color=(1, 0, 0, 0.5),
#          label='Positive')
# plt.hist(X[np.where(y==0)[0]][:, test_feat_ind], 
#          bins=np.arange(0, 10, 0.1), 
#          color=(0, 0, 1, 0.5),
#          label='negative')
# plt.xlim(-1, 10)
# plt.ylabel("Count", fontsize=16)
# plt.xlabel("H3K27me3_p", fontsize=16)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.savefig("H3K27me3_p_distri_sep.png", dpi=300)

unique_values = sorted(np.unique(X[:, test_feat_ind]))
start_point = int(len(unique_values) * 0.05)
end_point = int(len(unique_values) * 0.90)
unique_values = unique_values[start_point:(end_point+1)]
sample_values = [unique_values[i*(len(unique_values)//300)] for i in range(300)]

np.random.seed(123)
test_inds = np.random.choice(inds, size=(5000,))
preds = []
for ind in test_inds:
  X_sample = X[ind]
  X_sample = np.stack([X_sample] * len(sample_values), 0)
  X_sample[:, test_feat_ind] = np.array(sample_values)
  pred = model.predict_proba(X_sample)
  preds.append(pred[:, 1])
preds = np.mean(np.array(preds), 0)
plt.clf()
plt.plot(sample_values, preds, '.-', linewidth=3, color=(1., 0.55, 0.38))
plt.xlabel("H3K27me3_p", fontsize=16)
plt.ylabel("Predicted eQTL Prob.", fontsize=16)
plt.tight_layout()
plt.savefig("H3K27me3_p_change.png", dpi=300)

np.random.seed(123)
short_dist_inds = np.where(np.abs(X[:, 305]) < 50000)[0]
medium_dist_inds = np.array(list(set(np.where(np.abs(X[:, 305]) > 50000)[0]) & \
                                 set(np.where(np.abs(X[:, 305]) < 300000)[0])))
long_dist_inds = np.where(np.abs(X[:, 305]) > 300000)[0]
test_inds_short_dist = np.random.choice(short_dist_inds, size=(5000,))
test_inds_medium_dist = np.random.choice(medium_dist_inds, size=(5000,))
test_inds_long_dist = np.random.choice(long_dist_inds, size=(5000,))
preds = []
for ind in test_inds_short_dist:
  X_sample = X[ind]
  X_sample = np.stack([X_sample] * len(sample_values), 0)
  X_sample[:, test_feat_ind] = np.array(sample_values)
  pred = model.predict_proba(X_sample)
  preds.append(pred[:, 1])
preds_short_dist = np.mean(np.array(preds), 0)
preds = []
for ind in test_inds_medium_dist:
  X_sample = X[ind]
  X_sample = np.stack([X_sample] * len(sample_values), 0)
  X_sample[:, test_feat_ind] = np.array(sample_values)
  pred = model.predict_proba(X_sample)
  preds.append(pred[:, 1])
preds_medium_dist = np.mean(np.array(preds), 0)
preds = []
for ind in test_inds_long_dist:
  X_sample = X[ind]
  X_sample = np.stack([X_sample] * len(sample_values), 0)
  X_sample[:, test_feat_ind] = np.array(sample_values)
  pred = model.predict_proba(X_sample)
  preds.append(pred[:, 1])
preds_long_dist = np.mean(np.array(preds), 0)
plt.clf()
plt.plot(sample_values, preds_short_dist, '.-', linewidth=3, label='<50kb var-gene dist', color=(0.5, 0.7, 1.0))
plt.plot(sample_values, preds_medium_dist, '.-', linewidth=3, label='50kb~300kb var-gene dist', color=(1., 0.55, 0.38))
plt.plot(sample_values, preds_long_dist, '.-', linewidth=3, label='>300kb var-gene dist', color=(0.3, 0.8, 0.35))
plt.legend(fontsize=12)
plt.xlabel("H3K27me3_p", fontsize=16)
plt.ylabel("Predicted eQTL Prob.", fontsize=16)
plt.ylim(0.45, 0.53)
plt.tight_layout()
plt.savefig("H3K27me3_p_change_dist.png", dpi=300)

#### Feature effect:  Hi-C #######
HiC_feats_names = [n for n in feat_name if n.startswith('HiC')]
HiC_feats_inds = np.array([feat_name.index(n) for n in HiC_feats_names])
HiC_feats = X[:, HiC_feats_inds]

test_feat = 'HiCNormed_100kb_p'
test_feat_ind = HiC_feats_names.index(test_feat)
unique_values, unique_inds = np.unique(HiC_feats[:, test_feat_ind], return_index=True)

plt.clf()
plt.hist(HiC_feats[:, test_feat_ind], bins=np.arange(0, 80000, 800), color=(0.55, 0.63, 0.80))
plt.ylabel("Count", fontsize=16)
plt.xlabel("HiCNormed_100kb_p", fontsize=16)
plt.tight_layout()
plt.savefig("HiCNormed_100kb_p_distri.png", dpi=300)

# plt.clf()
# plt.hist(HiC_feats[np.where(y==1)[0]][:, test_feat_ind], 
#          bins=np.arange(0, 80000, 800), 
#          color=(1, 0, 0, 0.5),
#          label='Positive')
# plt.hist(HiC_feats[np.where(y==0)[0]][:, test_feat_ind], 
#          bins=np.arange(0, 80000, 800), 
#          color=(0, 0, 1, 0.5),
#          label='negative')
# plt.ylabel("Count", fontsize=16)
# plt.xlabel("HiCNormed_100kb_p", fontsize=16)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.savefig("HiCNormed_100kb_p_distri_sep.png", dpi=300)

start_point = int(len(unique_values) * 0.05)
end_point = int(len(unique_values) * 0.98)
unique_values = unique_values[start_point:(end_point+1)]
unique_inds = unique_inds[start_point:(end_point+1)]

sample_values = []
for i in range(300):
  center = i*(len(unique_values)//300) + 50
  inds = [unique_inds[i] for i in range(center-30, min(center+30, len(unique_inds)))]
  sample_value = np.median(HiC_feats[inds], 0)
  sample_values.append(sample_value)
sample_values = np.array(sample_values)
rep_values = sample_values[:, test_feat_ind]

np.random.seed(123)
long_dist_inds = np.where(np.abs(X[:, 305]) > 300000)[0]
test_inds_long_dist = np.random.choice(long_dist_inds, size=(5000,))
preds = []
for ind in test_inds_long_dist:
  X_sample = X[ind]
  X_sample = np.stack([X_sample] * len(sample_values), 0)
  X_sample[:, HiC_feats_inds] = sample_values
  pred = model.predict_proba(X_sample)
  preds.append(pred[:, 1])
preds_long_dist = np.mean(np.array(preds), 0)

plt.clf()
plt.plot(rep_values, preds_long_dist, '.-', linewidth=3, label='>300kb var-gene dist', color=(0.55, 0.63, 0.80))
plt.legend(fontsize=12)
plt.ylabel("Predicted eQTL Prob.", fontsize=16)
plt.xlabel("HiCNormed_100kb_p", fontsize=16)
plt.tight_layout()
plt.savefig("HiCNormed_100kb_p_change.png", dpi=300)

############# Feature importance plot #############################
import xgbfir
import pickle
import pandas as pd

model = pickle.load(open('./random_assembled_balanced_dataset_123_Xy_models.pkl', 'r'))['FULL'][0]
xgbfir.saveXgbFI(model, feat_name, OutputXlsxFile='random_model.xlsx')

dfs = pd.read_excel('random_model.xlsx', sheetname=None)
order_0 = dfs[u'Interaction Depth 0']
order_0_map = [(k, v) for k, v in zip(order_0['Interaction'], order_0['Gain'])][:40]

color_mapping = {'p': (0.4, 0.7607843137254902, 0.6470588235294118),
                 'g': (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
                 'v': (0.5, 0.5, 0.796078431372549)}
  
names = [p[0] for p in order_0_map]
gains = [p[1] for p in order_0_map]
colors = [color_mapping[s[-1]] for s in names]
fig, ax = plt.subplots()
y = np.arange(len(names))
ax.barh(y, gains, color=colors)
v_done = False
g_done = False
p_done = False
for i, gain, name in zip(y, gains, names):
  if name.endswith('v') and not v_done:
    ax.barh(i, gain, color=color_mapping['v'], label='Variant features')
    v_done = True
  if name.endswith('g') and not g_done:
    ax.barh(i, gain, color=color_mapping['g'], label='Gene features')
    g_done = True
  if name.endswith('p') and not p_done:
    ax.barh(i, gain, color=color_mapping['p'], label='Intermediate features')
    p_done = True
ax.set_yticks(y)
ax.set_yticklabels(names, fontdict={"size": 6})
ax.set_xlabel("Gain")
plt.tight_layout()
plt.legend()
plt.savefig("fig_feat_importance.png", dpi=300)