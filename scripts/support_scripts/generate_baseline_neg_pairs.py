#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:01:08 2019

@author: zqwu
"""

import os
import pickle
import os
import joblib
import numpy as np

os.chdir('/home/zqwu/eQTL/Data/bkp_123')
tissues = ['Blood', 'Brain', 'LCL', 'Muscle', 'Skin']

all_pairs = pickle.load(open('../assembled_balanced_dataset_123.pkl', 'r'))



# Distance bins
def ct_bins(bins, pairs):
  cts = np.zeros((len(bins),))
  dists = np.array([p[2].astype(int) for p in pairs])
  for i in range(len(cts)):
    cts[i] = np.where((dists < bins[i][1]) & (dists >= bins[i][0]))[0].shape[0]
  return cts
n_bins = 50
distances = list(sorted([p[2].astype(int) for p in all_pairs['pos']]))
n_points = len(distances)
bin_boundaries = [distances[(n_points * i)//n_bins] for i in range(n_bins)]
bin_boundaries[0] = -1000000
bin_boundaries.append(1000000)
assert len(bin_boundaries) == n_bins + 1
bins = [(bin_boundaries[i], bin_boundaries[i+1]) for i in range(n_bins)]
pos_cts = ct_bins(bins, all_pairs['pos'])

# Pairs that should not appear
existing_neg_names = set(v[0] + ';' + v[1] for v in all_pairs['neg'])
pos_vars = set([v[1] for v in all_pairs['pos']])
unique_pos_pairs_names = set()
for tissue in tissues:
  pos_samples = os.listdir(os.path.join('..', '..', tissue, 'pos'))
  pos_samples = [sample for sample in pos_samples if sample.startswith('pos_pair')]
  for sample in pos_samples:
    path = os.path.join('..', '..', tissue, 'pos', sample)
    dat = np.array(joblib.load(path))
    names = [line[0] + ';' + line[1] for line in dat if line[1] in pos_vars]
    unique_pos_pairs_names = unique_pos_pairs_names | set(names)


# Same var
pool = set()
all_same_var_neg_pairs = []
for tissue in tissues:
  neg_files = [f_n for f_n in os.listdir('.') if tissue in f_n and 'neg' in f_n]
  for f_n in neg_files:
    dat = pickle.load(open(f_n, 'r'))
    for v in dat:
      name = v[0] + ';' + v[1]
      if name not in existing_neg_names and \
         name not in unique_pos_pairs_names and \
         name not in pool:
        pool.add(name)
        all_same_var_neg_pairs.append(v)


np.random.seed(123)
selected_inds = np.random.choice(np.arange(len(all_same_var_neg_pairs)), (10000,), replace=False)
random_picked_samples = [all_same_var_neg_pairs[i] for i in selected_inds]
dat = {'pos':[], 'neg':random_picked_samples}
with open('./assembled_balanced_dataset_123_same_var_baseline.pkl', 'w') as f:
  pickle.dump(dat, f)


np.random.seed(123)
neg_cts = (pos_cts/np.sum(pos_cts) * 10000).astype(int)
selected_inds = np.arange(len(all_same_var_neg_pairs))
np.random.shuffle(selected_inds)
matched_picked_samples = []
def which_bin(bins, line):
  dis = int(line[2])
  return sum(dis > np.array(bins)[:, 0]) - 1
for i in selected_inds:
  if np.sum(neg_cts) == 0:
    break
  b = which_bin(bins, all_same_var_neg_pairs[i])
  if neg_cts[b] > 0:
    neg_cts[b] -= 1
    matched_picked_samples.append(all_same_var_neg_pairs[i])
dat = {'pos':[], 'neg':matched_picked_samples}
with open('./assembled_balanced_dataset_123_same_var_baseline_matched.pkl', 'w') as f:
  pickle.dump(dat, f)


# Different var
all_pos_pairs_names = set()
for tissue in tissues:
  pos_samples = os.listdir(os.path.join('..', '..', tissue, 'pos'))
  pos_samples = [sample for sample in pos_samples if sample.startswith('pos_pair')]
  for sample in pos_samples:
    path = os.path.join('..', '..', tissue, 'pos', sample)
    dat = np.array(joblib.load(path))
    names = [line[0] + ';' + line[1] for line in dat]
    all_pos_pairs_names = all_pos_pairs_names | set(names)

np.random.seed(123)
neg_selected = []
neg_selected_names = set()
for tissue in tissues:
  neg_samples = os.listdir(os.path.join('..', '..', tissue, 'neg'))
  neg_samples = [sample for sample in neg_samples if sample.startswith('neg_pair')]
  
  selected_neg_samples = np.random.choice(neg_samples, (int(0.05*len(neg_samples)),), replace=False)
  for sample in selected_neg_samples:
    path = os.path.join('..', '..', tissue, 'neg', sample)
    dat = np.array(joblib.load(path))
    np.random.shuffle(dat)
    for line in dat[:5000]:
      name = line[0] + ';' + line[1]
      if line[1] not in pos_vars and \
         name not in all_pos_pairs_names and \
         name not in neg_selected_names:
        neg_selected_names.add(name)
        neg_selected.append(line)
      
np.random.seed(123)
selected_inds = np.random.choice(np.arange(len(neg_selected)), (10000,), replace=False)
random_picked_samples = [neg_selected[i] for i in selected_inds]
dat = {'pos':[], 'neg':random_picked_samples}
with open('./assembled_balanced_dataset_123_diff_var_baseline.pkl', 'w') as f:
  pickle.dump(dat, f)


np.random.seed(123)
neg_cts = (pos_cts/np.sum(pos_cts) * 10000).astype(int)
selected_inds = np.arange(len(neg_selected))
np.random.shuffle(selected_inds)
matched_picked_samples = []
def which_bin(bins, line):
  dis = int(line[2])
  return sum(dis > np.array(bins)[:, 0]) - 1
for i in selected_inds:
  if np.sum(neg_cts) == 0:
    break
  b = which_bin(bins, neg_selected[i])
  if neg_cts[b] > 0:
    neg_cts[b] -= 1
    matched_picked_samples.append(neg_selected[i])
dat = {'pos':[], 'neg':matched_picked_samples}
with open('./assembled_balanced_dataset_123_diff_var_baseline_matched.pkl', 'w') as f:
  pickle.dump(dat, f)