#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 23:59:27 2018

@author: zqwu
"""

import os
import joblib
import numpy as np
import pickle
import copy
tissues = ['./Blood', './Brain', './LCL', './Muscle', './Skin']
SEED = 123

with open('./total_pos_selected.pkl', 'r') as f:
  pos_pairs = pickle.load(f)
pos_vars, pos_vars_cts = np.unique([line[1] for line in pos_pairs], return_counts=True)
  
#==============================================================================

print("Pick all negative pairs including selected variants with positive eQTLs ")
neg_dict = {}
for var in pos_vars:
  neg_dict[var] = []

for tissue in tissues:
  j = 0
  neg_this_tissue = []
  neg_samples = os.listdir(os.path.join(tissue, 'neg'))
  neg_samples = [sample for sample in neg_samples if sample.startswith('neg_pair')]
  for sample in neg_samples:
    path = os.path.join(tissue, 'neg', sample)
    dat = np.array(joblib.load(path))
    unique_vars = np.unique(dat[:, 1])
    overlap_vars = set(list(pos_vars)) & set(list(unique_vars))
    neg_lines = [line for line in dat if line[1] in overlap_vars]
    
    neg_this_tissue.extend(neg_lines)
    if len(neg_this_tissue) > 100000:
      with open(tissue + '_neg' + str(j) + '.pkl', 'w') as f:
        pickle.dump(neg_this_tissue[:100000], f)
      neg_this_tissue = neg_this_tissue[100000:]
      j += 1
  with open(tissue + '_neg' + str(j) + '.pkl', 'w') as f:
    pickle.dump(neg_this_tissue, f)

#==============================================================================

print(" Excluding all pairs that exist in any tissue as positive eQTLs ")
unique_pos_pairs_names = set()
for tissue in tissues:
  pos_samples = os.listdir(os.path.join(tissue, 'pos'))
  pos_samples = [sample for sample in pos_samples if sample.startswith('pos_pair')]
  for sample in pos_samples:
    path = os.path.join(tissue, 'pos', sample)
    dat = np.array(joblib.load(path))
    names = [line[0] + ';' + line[1] for line in dat if line[1] in pos_vars]
    unique_pos_pairs_names = unique_pos_pairs_names | set(names)
#==============================================================================

print(" Generate 50 var-gene distance bins ")
def ct_bins(bins, pairs):
  cts = np.zeros((len(bins),))
  dists = np.array([p[2].astype(int) for p in pairs])
  for i in range(len(cts)):
    cts[i] = np.where((dists < bins[i][1]) & (dists >= bins[i][0]))[0].shape[0]
  return cts
  
n_bins = 50
distances = list(sorted([p[2].astype(int) for p in pos_pairs]))
n_points = len(distances)
bin_boundaries = [distances[(n_points * i)//n_bins] for i in range(n_bins)]
bin_boundaries[0] = -1000000
bin_boundaries.append(1000000)
assert len(bin_boundaries) == n_bins + 1
bins = [(bin_boundaries[i], bin_boundaries[i+1]) for i in range(n_bins)]
pos_cts = ct_bins(bins, pos_pairs)

#==============================================================================

print(" Based on number of samples in the smallest bin, \
        randomly pick 3*min(n_samples_each_bin) samples in each bin ")
for tissue in tissues:
  neg_samples_in_this_tisse = [name for name in os.listdir('.') if 'neg' in name and not 'selected' in name and name.startswith(tissue[2:])]
  neg_pairs_in_this_tissue = [pickle.load(open(name, 'r')) for name in neg_samples_in_this_tisse]
  neg_pairs_in_this_tissue = np.concatenate([np.stack(p, 0) for p in neg_pairs_in_this_tissue], 0)
  
  neg_pairs_in_this_tissue_non_overlapping = \
      [line for line in neg_pairs_in_this_tissue if line[0] + ';' + line[1] not in unique_pos_pairs_names]
  
  print(tissue[2:] + ": %d" % len(neg_pairs_in_this_tissue_non_overlapping))
  neg_cts_in_this_tissue = ct_bins(bins, neg_pairs_in_this_tissue_non_overlapping)
  

  np.random.seed(SEED)
  samples_each_bin = np.min(neg_cts_in_this_tissue).astype(int)
  selected_neg_pairs_in_this_tissue = []
  neg_distances_this_tissue = np.array([p[2].astype(int) for p in neg_pairs_in_this_tissue_non_overlapping])
  for b in bins:
    ids = np.where((neg_distances_this_tissue < b[1]) & (neg_distances_this_tissue >= b[0]))[0]
    np.random.shuffle(ids)
    samples_here = min(len(ids), 4*samples_each_bin)
    selected_neg_pairs_in_this_tissue.extend([neg_pairs_in_this_tissue_non_overlapping[i] for i in ids[:samples_here]])
  print(tissue[2:] + " selected: %d" % len(selected_neg_pairs_in_this_tissue))
  
  with open(tissue+'_neg_selected.pkl', 'w') as f:
    pickle.dump(selected_neg_pairs_in_this_tissue, f)

#==============================================================================

print(" Combine all tissues, exclude identical pairs ")
all_neg_selected = []
for tissue in tissues:
  all_neg_selected.extend(pickle.load(open(tissue+'_neg_selected.pkl', 'r')))
  
names = [line[0] + ';' + line[1] for line in all_neg_selected]
uname, inds = np.unique(names, return_index=True)
all_neg_selected_unique = [all_neg_selected[i] for i in inds]
dist = [p[2].astype(int) for p in all_neg_selected_unique]

#==============================================================================

print(" Calculate number of pairs needed for each variant ")
neg_dict = {}
ct_dict = {}
for p_var, p_var_ct in zip(pos_vars, pos_vars_cts):
  neg_dict[p_var] = []
  ct_dict[p_var] = p_var_ct

for line in all_neg_selected_unique:
  neg_dict[line[1]].append(line)

#==============================================================================

print(" Loop through each variant, \
        select same number of negative pairs involving that variant, \
        and satisfy distance bin distribution. ")
neg_cts = copy.deepcopy(pos_cts)
np.random.seed(SEED)

def which_bin(bins, line):
  dis = int(line[2])
  return sum(dis > np.array(bins)[:, 0]) - 1

final_neg_pairs = []
total_ct = 0
order = list(pos_vars)
order.sort(key=lambda x: len(neg_dict[x]))
for var in order:
  pool = neg_dict[var]
  n_needed = ct_dict[var]
  pool.sort(key=lambda x: np.abs(int(x[2]) + int(10 ** np.random.normal(0, 6))))
  ct = 0
  for i in range(len(pool)):
    if ct >= n_needed:
      break
    bin_belonging = which_bin(bins, pool[i])
    if neg_cts[bin_belonging] > 0:
      ct += 1
      final_neg_pairs.append(pool[i])
      total_ct += 1
      neg_cts[bin_belonging] -= 1

#==============================================================================

print(" Look for variants that have more pos pairs than neg pairs ")
neg_ct_dict = {}
for line in final_neg_pairs:
  if line[1] not in neg_ct_dict:
    neg_ct_dict[line[1]] = 1
  else:
    neg_ct_dict[line[1]] += 1

missing_vars = dict()
for var in pos_vars:
  if not var in neg_ct_dict:
    missing_vars[var] = ct_dict[var]
  else:
    if ct_dict[var] > neg_ct_dict[var]:
      missing_vars[var] = ct_dict[var] - neg_ct_dict[var]

#==============================================================================

print(" Extract all neg pairs for missing variants ")
missing_pairs = []
for tissue in tissues:
  neg_samples_in_this_tisse = [name for name in os.listdir('.') if 'neg' in name and not 'selected' in name and name.startswith(tissue[2:])]
  neg_pairs_in_this_tissue = [pickle.load(open(name, 'r')) for name in neg_samples_in_this_tisse]
  neg_pairs_in_this_tissue = np.concatenate([np.stack(p, 0) for p in neg_pairs_in_this_tissue], 0)
  
  neg_pairs_in_this_tissue_non_overlapping = \
      [line for line in neg_pairs_in_this_tissue if line[0] + ';' + line[1] not in unique_pos_pairs_names]
  
  missing_pairs.extend([line for line in neg_pairs_in_this_tissue_non_overlapping if line[1] in missing_vars])
  
names = [line[0] + ';' + line[1] for line in missing_pairs]
uname, inds = np.unique(names, return_index=True)
missing_pairs_unique = [missing_pairs[i] for i in inds]

missing_pair_dict = dict()
for var in missing_vars:
  missing_pair_dict[var] = []
for line in missing_pairs_unique:
  missing_pair_dict[line[1]].append(line)

#==============================================================================

print(" Fill in remaining gap, prioritizing distance bins requirement, \
        remove extra positive pairs if necessary ")
np.random.seed(SEED)
pos_remove_ids = []
neg_extra_pairs = []
for var in missing_vars:
  pool = missing_pair_dict[var]
  n_needed = ct_dict[var]
  np.random.shuffle(pool)
  already_selected = [line[0] + ';' + line[1] for line in final_neg_pairs if line[1] == var]
  ct = len(already_selected)
  for i in range(len(pool)):
    if pool[i][0] + ';' + pool[i][1] not in already_selected:
      if ct >= n_needed:
        break
      bin_belonging = which_bin(bins, pool[i])
      if neg_cts[bin_belonging] > 0:
        ct += 1
        neg_extra_pairs.append(pool[i])
        already_selected.append(pool[i][0] + ';' + pool[i][1])
        total_ct += 1
        neg_cts[bin_belonging] -= 1

  if (ct < n_needed):
    for i in range(len(pool)):
      if pool[i][0] + ';' + pool[i][1] not in already_selected:
        bin_belonging = which_bin(bins, pool[i])
        neg_extra_pairs.append(pool[i])
        total_ct += 1
        neg_cts[bin_belonging] -= 1
        ct += 1
        i += 1
        if ct == n_needed:
          break
  if ct < n_needed:
    pos_selected_pairs_id = [idd for idd, line in enumerate(pos_pairs) if line[1] == var]
    for i in range(n_needed - ct):
      pos_remove_ids.append(pos_selected_pairs_id[i])
final_neg_pairs.extend(neg_extra_pairs)
final_pos_pairs = [line for i, line in enumerate(pos_pairs) if i not in pos_remove_ids]

print("Number of pos: %d" % len(final_pos_pairs))
print("Number of neg: %d" % len(final_neg_pairs))
with open("assembled_balanced_dataset_" + str(SEED) + ".pkl", "w") as f:
  pickle.dump({"pos": final_pos_pairs, "neg": final_neg_pairs}, f)
