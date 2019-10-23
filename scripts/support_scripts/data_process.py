#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:22:35 2017

@author: zqwu
"""
import os
os.chdir('/home/zqwu/eQTL')
EXONS_FILE = './exons.pkl'
TOTAL_DATA_FILE = './Brain/Brain_Cortex.allpairs.txt'
WORKING_DIR = './Brain/'

import pickle
import joblib
import numpy as np

with open(EXONS_FILE, 'r') as f:
  EXONS = pickle.load(f)

def valid(variant):
  eles = variant.split('_')
  if len(eles[2]) > 1 or len(eles[3]) > 1:
    return False
  chrom_num = eles[0]
  if chrom_num == 'X':
    chrom_num = '23'
  elif chrom_num == 'Y':
    chrom_num = '24'
  try:
    chrom_num = int(chrom_num)
    position = int(eles[1])
  except:
    return False
  exons_positions = EXONS[chrom_num-1][1]
  temp = np.sign(exons_positions - position)
  if len(np.unique(temp[:, 0] * temp[:, 1])) > 1:
    return False
  return True
  
  
  
j = [0] * 6
counts = [0] * 6
buffers = [[] for i in range(6)]
filenames = [WORKING_DIR+'pos/pos_pair_',
             WORKING_DIR+'pos/excluded_pos_pair_',
             WORKING_DIR+'inter/inter_pair_',
             WORKING_DIR+'inter/excluded_inter_pair_',
             WORKING_DIR+'neg/neg_pair_',
             WORKING_DIR+'neg/excluded_neg_pair_']
             
with open(TOTAL_DATA_FILE, 'r') as f:
  for line in f:
    '''
  0 ['gene_id',
  1  'variant_id',
  2  'tss_distance',
  3  'ma_samples',
  4  'ma_count',
  5  'maf',
  6  'pval_nominal',
  7  'slope',
  8 'slope_se']
    '''
    if line[0:7] == 'gene_id':
      continue
    else:
      elements = line.split()
      elements[2] = int(elements[2])
      elements[3] = int(elements[3])
      elements[4] = int(elements[4])
      elements[5] = float(elements[5])
      elements[6] = float(elements[6])
      elements[7] = float(elements[7])
      elements[8] = float(elements[8])
      if not valid(elements[1]):
        if elements[6] < 10**-5:
          buffers[1].append(elements)
          counts[1] += 1
          if counts[1] == 10**5:
            joblib.dump(buffers[1], filenames[1] + str(j[1]) + '.joblib')
            buffers[1] = []
            counts[1] = 0
            j[1] += 1
        elif elements[6] >= 10**-5 and elements[6] < 0.1:
          buffers[3].append(elements)
          counts[3] += 1
          if counts[3] == 10**5:
            joblib.dump(buffers[3], filenames[3] + str(j[3]) + '.joblib')
            buffers[3] = []
            counts[3] = 0
            j[3] += 1
        elif elements[6] > 0.1:
          buffers[5].append(elements)
          counts[5] += 1
          if counts[5] == 10**5:
            joblib.dump(buffers[5], filenames[5] + str(j[5]) + '.joblib')
            buffers[5] = []
            counts[5] = 0
            j[5] += 1
      else:
        if elements[6] < 10**-5:
          buffers[0].append(elements)
          counts[0] += 1
          if counts[0] == 10**5:
            joblib.dump(buffers[0], filenames[0] + str(j[0]) + '.joblib')
            buffers[0] = []
            counts[0] = 0
            j[0] += 1
        elif elements[6] >= 10**-5 and elements[6] < 0.1:
          buffers[2].append(elements)
          counts[2] += 1
          if counts[2] == 10**5:
            joblib.dump(buffers[2], filenames[2] + str(j[2]) + '.joblib')
            buffers[2] = []
            counts[2] = 0
            j[2] += 1
        elif elements[6] > 0.1:
          buffers[4].append(elements)
          counts[4] += 1
          if counts[4] == 10**5:
            joblib.dump(buffers[4], filenames[4] + str(j[4]) + '.joblib')
            buffers[4] = []
            counts[4] = 0
            j[4] += 1
      
joblib.dump(buffers[0], filenames[0] + str(j[0]) + '.joblib')
joblib.dump(buffers[1], filenames[1] + str(j[1]) + '.joblib')
joblib.dump(buffers[2], filenames[2] + str(j[2]) + '.joblib')
joblib.dump(buffers[3], filenames[3] + str(j[3]) + '.joblib')
joblib.dump(buffers[4], filenames[4] + str(j[4]) + '.joblib')
joblib.dump(buffers[5], filenames[5] + str(j[5]) + '.joblib')

n_pos_files = j[0] + 1
n_neg_files = j[4] + 1

data = []
for i in range(n_pos_files):
  data.extend(joblib.load(filenames[0] + str(i) + '.joblib'))
genes = np.unique([d[0] for d in data])
all_pos = {}
for gene in genes:
  all_pos[gene] = []
for line in data:
  all_pos[line[0]].append(line)

pos_selected_map = {}
for gene in genes:
  p_val = [l[6] for l in all_pos[gene]]
  indice = np.argmax(p_val)
  pos_selected_map[gene] = all_pos[gene][indice]

joblib.dump(pos_selected_map.values(), WORKING_DIR+'pos_selected_' + str(len(pos_selected_map)) + '.joblib')

neg_selected_map = {}
for gene in genes:
  neg_selected_map[gene] = []
for i in range(n_neg_files):
  data2 = joblib.load(filenames[4] + str(i) + '.joblib')
  for line in data2:
    if line[0] in genes:
      if np.abs(line[5] - pos_selected_map[line[0]][5]) < 0.05:
        neg_selected_map[line[0]].append(line)

joblib.dump(neg_selected_map, WORKING_DIR+'neg_selected_MAF_0_05.joblib')

tss_s = [p[2] for p in pos_selected_map.values()]
tss_s = sorted(tss_s)
bins = []
prev = -10**6
n_bins = 30
n_samples_in_bin = len(tss_s) // n_bins
for i in range(1, n_bins):
  next_val = tss_s[i*n_samples_in_bin]
  bins.append((prev, next_val))
  prev = next_val
bins.append((prev, 10**6))

joblib.dump(bins, WORKING_DIR+'bins.joblib')

pos_counts = np.zeros((n_bins,))
for sample in pos_selected_map.values():
  for i in range(n_bins):
    if sample[2] >= bins[i][0] and sample[2] < bins[i][1]:
      pos_counts[i] += 1

neg_bins = [[] for i in range(n_bins)]
for key in neg_selected_map.keys():
  for line in neg_selected_map[key]:
    for i in range(30):
      if line[2] >= bins[i][0] and line[2] < bins[i][1]:
        neg_bins[i].append(line)

for j in range(10):
  neg_samples = []
  selected_genes = []
  order = [14, 15, 13, 16, 12, 17, 11, 18, 10, 19, 9, 20, 8, 21, 7, 22, 6, 23, 5, 24, 4, 25, 3, 26, 2, 27, 1, 28, 0, 29]
  for i in order:
    pool = neg_bins[i]
    np.random.shuffle(pool)
    count = 0
    for line in pool:
      if line[0] not in selected_genes and count < pos_counts[i]:
        neg_samples.append(line)
        selected_genes.append(line[0])
        count += 1
  joblib.dump(neg_samples, WORKING_DIR+'neg_selected_set'+str(j) + '_'+str(len(neg_samples))+'.joblib')
