#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:56:38 2019

@author: zqwu
"""

import pickle
import numpy as np
import os

fs = ['assembled_balanced_dataset_123_extended_Blood.pkl',
      'assembled_balanced_dataset_123_extended_Muscle.pkl',
      'assembled_balanced_dataset_123_extended_LCL.pkl',
      'assembled_balanced_dataset_123_extended_Skin.pkl',
      'assembled_balanced_dataset_123_extended_Brain.pkl']
    

d_name = 'assembled_balanced_dataset_123.pkl'
with open(d_name, 'r') as f:
  all_dat = pickle.load(f)
  all_dat = all_dat['pos'] + all_dat['neg']

unique_vars = set([d[1] for d in all_dat])

outputs = []
for f in fs:
  dat = pickle.load(open(f, 'r'))
  # Pos-upstream, pos-downstream, neg-upstream, neg-downstream
  closest_pairs = {var:[None, None, None, None] for var in unique_vars}
  for key in dat.keys():
    assert key in unique_vars
    for pair in dat[key]:
      if pair[6] <= 1e-5: # Positive
        if pair[2] > 0: # Upstream
          if closest_pairs[key][0] is None or \
             pair[2] < closest_pairs[key][0][2]:
            closest_pairs[key][0] = pair
        else: # Downstream
          if closest_pairs[key][1] is None or \
             pair[2] > closest_pairs[key][1][2]:
            closest_pairs[key][1] = pair
      else: # Negative
        if pair[2] > 0: # Upstream
          if closest_pairs[key][2] is None or \
             pair[2] < closest_pairs[key][2][2]:
            closest_pairs[key][2] = pair
        else: # Downstream
          if closest_pairs[key][3] is None or \
             pair[2] > closest_pairs[key][3][2]:
            closest_pairs[key][3] = pair
  outputs.append(closest_pairs)
  with open('closest_' + f, 'w') as f_out:
    pickle.dump(closest_pairs, f_out)
        
combined = {var:[None, None, None, None] for var in unique_vars}
for var in unique_vars:
  choices = [out[var] for out in outputs]
  #pos-up
  pairs = [c[0] for c in choices if c[0] is not None]
  if len(pairs) > 0:
    dists = np.array([pair[2] for pair in pairs]).astype(int)
    assert np.all(dists > 0)
    combined[var][0] = pairs[np.argmin(dists)]
  #pos-down
  pairs = [c[1] for c in choices if c[1] is not None]
  if len(pairs) > 0:
    dists = np.array([pair[2] for pair in pairs]).astype(int)
    assert np.all(dists < 0)
    combined[var][1] = pairs[np.argmax(dists)]

  #neg-up
  pairs = [c[2] for c in choices if c[2] is not None]
  if len(pairs) > 0:
    dists = np.array([pair[2] for pair in pairs]).astype(int)
    assert np.all(dists > 0)
    combined[var][2] = pairs[np.argmin(dists)]
  #pos-down
  pairs = [c[3] for c in choices if c[3] is not None]
  if len(pairs) > 0:
    dists = np.array([pair[2] for pair in pairs]).astype(int)
    assert np.all(dists < 0)
    combined[var][3] = pairs[np.argmax(dists)]

with open('assembled_balanced_dataset_123_closest_combined.pkl', 'w') as f_out:
  pickle.dump(combined, f_out)
  
  
vs = []
for var in combined:
  pairs = combined[var]
  if pairs[0] is None:
    pos_dist = np.abs(pairs[1][2])
  elif pairs[1] is None:
    pos_dist = np.abs(pairs[0][2])
  else:
    pos_dist = min(np.abs(pairs[0][2]),
                   np.abs(pairs[1][2]))

  if pairs[2] is None:
    neg_dist = np.abs(pairs[3][2])
  elif pairs[3] is None:
    neg_dist = np.abs(pairs[2][2])
  else:
    neg_dist = min(np.abs(pairs[2][2]),
                   np.abs(pairs[3][2]))
  
  if neg_dist < pos_dist:
    if pairs[0] is not None and pairs[2] is not None:
      if pairs[2][2] < pairs[0][2]:
        vs.append(var)
        continue
    if pairs[1] is not None and pairs[3] is not None:
      if pairs[3][2] > pairs[1][2]:
        vs.append(var)
        continue

# Non-neighbor variants: defined as variants with negatively-related gene closer to positively-related gene (on one side)
with open('./assembled_balanced_dataset_123_nonneighbor_vars.pkl', 'w') as f:
    pickle.dump(vs, f)