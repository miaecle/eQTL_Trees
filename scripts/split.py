#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 22:42:08 2018

@author: zqwu
"""
import numpy as np

def random_split(X, y, dat, K=10):
  cv_inds = []

  all_variants = np.unique([d[1] for d in dat])
  mapping = {}
  for var in all_variants:
    mapping[var] = []
  for i in range(len(dat)):
    mapping[dat[i][1]].append(i)

  np.random.seed(123)
  np.random.shuffle(all_variants)
  split_points = np.linspace(0, len(all_variants), K+1, dtype=int)
  for i in range(K):
    valid_variants = np.array(all_variants[split_points[i]:split_points[i+1]])
    train_variants = np.array([v for v in all_variants if v not in valid_variants])
    valid_inds = np.concatenate([mapping[var] for var in valid_variants])
    np.random.shuffle(valid_inds)
    train_inds = np.concatenate([mapping[var] for var in train_variants])
    np.random.shuffle(train_inds)
    assert len(set(train_inds.astype(int)) & set(valid_inds.astype(int))) == 0
    cv_inds.append((train_inds.astype(int), valid_inds.astype(int)))
  return cv_inds
    
def position_split(X, y, dat, K=10):
  cv_inds = []
  names = [line[0] + ';' + line[1] for line in dat]
  inds_by_position = sorted(range(len(names)), 
       key=lambda x: (names[x].split(';')[1].split('_')[0], int(names[x].split(';')[1].split('_')[1])))
  
  K = 10
  split_points = np.linspace(0, len(inds_by_position), K+1, dtype=int)
  for i in range(K):
    valid_inds = np.array(inds_by_position[split_points[i]:split_points[i+1]])
    np.random.shuffle(valid_inds)
    train_inds = np.concatenate([inds_by_position[:split_points[i]], 
                                 inds_by_position[split_points[i+1]:]])
    np.random.shuffle(train_inds)
    assert len(set(train_inds.astype(int)) & set(valid_inds.astype(int))) == 0
    cv_inds.append((train_inds.astype(int), valid_inds.astype(int)))
  return cv_inds

def maf_split(X, y, dat, thrs=[0.01]):
  cv_inds = []
  maf = np.array([float(d[5]) for d in dat])
  for thr in thrs:
    valid_inds = np.where(maf <= thr)[0]
    valid_variants = set([dat[i][1] for i in valid_inds])
    valid_inds = np.array([i for i in range(len(dat)) if dat[i][1] in valid_variants])
    np.random.shuffle(valid_inds)
    train_inds = np.array([i for i in range(len(dat)) if dat[i][1] not in valid_variants])
    np.random.shuffle(train_inds)
    assert len(set(train_inds.astype(int)) & set(valid_inds.astype(int))) == 0
    cv_inds.append((train_inds.astype(int), valid_inds.astype(int)))
  return cv_inds
    
def chromosome_split(X, y, dat, ch_leaveout_sets=None):
  cv_inds = []
  chroms = np.array([d[1].split('_')[0] for d in dat])
  if ch_leaveout_sets is None:
    ch_leaveout_sets = [['1'], ['2'], ['3'], ['4'], ['5'], ['6'], ['7'], ['8'], ['9'], ['10'],
                        ['11'], ['12'], ['13'], ['14'], ['15'], ['16'], ['17'], ['18'], ['19'],
                        ['20'], ['21'], ['22'], ['X']]
  
  for ch_leaveout in ch_leaveout_sets:
    n_leaveout = []
    for ch in ch_leaveout:
      n_leaveout.append(np.where(chroms == ch)[0])
    valid_inds = np.concatenate(n_leaveout)
    np.random.shuffle(valid_inds)
    train_inds = np.array([ind for ind in np.arange(len(chroms)) if ind not in valid_inds])
    np.random.shuffle(train_inds)
    assert len(set(train_inds.astype(int)) & set(valid_inds.astype(int))) == 0
    cv_inds.append((train_inds.astype(int), valid_inds.astype(int)))
  return cv_inds
