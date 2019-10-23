#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 21:22:26 2018

@author: zqwu
"""

import os
import joblib
import numpy as np
import pickle

SEED=123

tissues = ['./Blood', './Brain', './LCL', './Muscle', './Skin']

'''
['gene_id',
 'variant_id',
 'tss_distance',
 'ma_samples',
 'ma_count',
 'maf',
 'pval_nominal',
 'slope',
 'slope_se']
'''

pos_pairs_tissues = []
print("First  pass")
for tissue in tissues:
  print("On %s" % tissue)
  pos_pairs = dict()
  pos_samples = os.listdir(os.path.join(tissue, 'pos'))
  pos_samples = [sample for sample in pos_samples if sample.startswith('pos_pair')]
  for sample in pos_samples:
    path = os.path.join(tissue, 'pos', sample)
    dat = np.array(joblib.load(path))
    unique_genes = np.unique(dat[:, 0])
    for g in unique_genes:
      if g not in pos_pairs:
        pos_pairs[g] = []
    for line in dat:
      var_position = line[1].split('_')[:2]
      pos_pairs[line[0]].append((var_position[0], int(var_position[1]), tissue, float(line[6])))
  pos_pairs_tissues.append(pos_pairs)

print("Gather genetic distances")
for tissue_pairs, tissue in zip(pos_pairs_tissues, tissues):
  print("On %s" % tissue)
  cwd = os.getcwd()
  for key, val in tissue_pairs.items():
    assert(len(np.unique(np.array(val)[:, 0])) == 1)
    val = sorted(val, key=lambda x: x[1])
    os.chdir('/home/zqwu/eQTL/utils/RecombinationMap')
    with open('infile', 'w') as f:
      for line in val:
        f.write(line[0] + ' ' + str(line[1]) + '\n')
    os.system('rm infile.out')
    assert os.system('./find_genetic_distances infile') == 0
    assert os.path.exists('./infile.out')
    with open('infile.out', 'r') as f:
      lines = f.readlines()
    val = [(p[0], p[1], float(l), p[2], p[3]) for l, p in zip(lines, val)]
    tissue_pairs[key] = val
  os.chdir(cwd)
  with open(tissue + '_pos.pkl', 'w') as f:
    pickle.dump(tissue_pairs, f)
  
total_pos_pairs = dict()
for p in pos_pairs_tissues:
  for key, val in p.items():
    if key not in total_pos_pairs:
      total_pos_pairs[key] = []
    total_pos_pairs[key].extend(val)
with open('./total_pos.pkl', 'w') as f:
    pickle.dump(total_pos_pairs, f)

print("Selecting variants")
threshold = 5.
selected_pos = []
np.random.seed(SEED)
for gene in total_pos_pairs:
  vals = total_pos_pairs[gene]
  if len(vals) == 1:
    selected_pos.append((gene, vals[0][0], vals[0][1], vals[0][2], vals[0][3]))
  else:
    vals.sort(key = lambda x: x[2])
    ind = np.argmax([v[4] for v in vals])
    #ind = np.random.randint(0, len(vals))
    selected_pos.append((gene, vals[ind][0], vals[ind][1], vals[ind][2], vals[ind][3]))
    thr_bottom = vals[ind][2]
    thr_top = vals[ind][2]
    for i in range(ind, len(vals)):
      if vals[i][2] - thr_bottom > threshold:
        selected_pos.append((gene, vals[i][0], vals[i][1], vals[i][2], vals[i][3]))
        thr_bottom = vals[i][2]
    for i in range(ind, 0, -1):
      if - vals[i][2] + thr_top > threshold:
        selected_pos.append((gene, vals[i][0], vals[i][1], vals[i][2], vals[i][3]))
        thr_top = vals[i][2]

selected_data = []
print("Retrieve full lines")
for tissue in tissues:
  selected_data_this_tissue = []
  tissue_selected = [p for p in selected_pos if p[4] == tissue]
  pair_names = [(p[0], p[1] + '_' + str(p[2])) for p in tissue_selected]
  tissue_selected_dict = {}
  for name in pair_names:
    if name[0] not in tissue_selected_dict:
      tissue_selected_dict[name[0]] = []
    tissue_selected_dict[name[0]].append(name[1])
                
  
  pos_samples = os.listdir(os.path.join(tissue, 'pos'))
  pos_samples = [sample for sample in pos_samples if sample.startswith('pos_pair')]
  for sample in pos_samples:
    path = os.path.join(tissue, 'pos', sample)
    dat = np.array(joblib.load(path))
    for line in dat:
      if line[0] in tissue_selected_dict:
        for v in tissue_selected_dict[line[0]]:
          if line[1].startswith(v):
            selected_data_this_tissue.append(line)
  selected_names = [(line[0], '_'.join(line[1].split('_')[:2])) for line in selected_data_this_tissue]
  assert set(selected_names) == set(pair_names)
  selected_data.extend(selected_data_this_tissue)

with open('./total_pos_selected.pkl', 'w') as f:
  pickle.dump(selected_data, f)
