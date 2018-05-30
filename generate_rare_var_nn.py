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

pos_nns = np.array([None]*n_samples)
unique_genes = np.unique(all_X[:, 0])

for i in range(3):
  dat = np.array(joblib.load('../pos/pos_pair_'+str(i)+'.joblib'))
  dat = dat[np.where(dat[:, 5].astype(float) > 0.05)[0]]
            
  genes_in_dat = set(np.unique(dat[:, 0])) & set(unique_genes)
  for gene in genes_in_dat:
    queries = np.where(all_X[:, 0] == gene)[0]
    targets = np.where(dat[:, 0] == gene)[0]
    dis1 = all_X[queries][:, 2].astype(int).reshape((-1, 1))
    dis2 = dat[targets][:, 2].astype(int).reshape((1, -1))
    dis_dis = np.abs(dis1 - dis2)
    dis_dis[np.where(dis_dis ==0)] = 1000000
    closest_targets = np.argmin(np.abs(dis_dis), axis=1)
    
    for ii, ind in enumerate(queries):
      potential_target = dat[targets[closest_targets[ii]]]
      if pos_nns[ind] is None or np.abs(pos_nns[ind][2].astype(int) - \
                                        int(all_X[ind, 2])) > \
                                 np.abs(potential_target[2].astype(int) - \
                                        int(all_X[ind, 2])):
        if int(potential_target[2]) != int(all_X[ind, 2]):
          pos_nns[ind] = potential_target


neg_nns = np.array([None]*n_samples)
unique_genes = np.unique(all_X[:, 0])

for i in range(160):
  if i%20 == 0:
    print("on %d" % i)
  dat = np.array(joblib.load('../inter/inter_pair_'+str(i)+'.joblib'))
  dat = dat[np.where(dat[:, 5].astype(float) > 0.05)[0]]
            
  genes_in_dat = set(np.unique(dat[:, 0])) & set(unique_genes)
  for gene in genes_in_dat:
    queries = np.where(all_X[:, 0] == gene)[0]
    targets = np.where(dat[:, 0] == gene)[0]
    dis1 = all_X[queries][:, 2].astype(float).reshape((-1, 1))
    dis2 = dat[targets][:, 2].astype(float).reshape((1, -1))
    dis_dis = np.abs(dis1 - dis2)
    dis_dis[np.where(dis_dis ==0)] = 1000000
    closest_targets = np.argmin(np.abs(dis_dis), axis=1)
    
    for ii, ind in enumerate(queries):
      potential_target = dat[targets[closest_targets[ii]]]
      if neg_nns[ind] is None or np.abs(neg_nns[ind][2].astype(int) - \
                                        int(all_X[ind, 2])) > \
                                 np.abs(potential_target[2].astype(int) - \
                                        int(all_X[ind, 2])):
        if int(potential_target[2]) != int(all_X[ind, 2]):
          neg_nns[ind] = potential_target

for i in range(1234):
  if i%20 == 0:
    print("on %d" % i)
  dat = np.array(joblib.load('../neg/neg_pair_'+str(i)+'.joblib'))
  dat = dat[np.where(dat[:, 5].astype(float) > 0.05)[0]]
            
  genes_in_dat = set(np.unique(dat[:, 0])) & set(unique_genes)
  for gene in genes_in_dat:
    queries = np.where(all_X[:, 0] == gene)[0]
    targets = np.where(dat[:, 0] == gene)[0]
    dis1 = all_X[queries][:, 2].astype(float).reshape((-1, 1))
    dis2 = dat[targets][:, 2].astype(float).reshape((1, -1))
    dis_dis = np.abs(dis1 - dis2)
    dis_dis[np.where(dis_dis ==0)] = 1000000
    closest_targets = np.argmin(np.abs(dis_dis), axis=1)
    
    for ii, ind in enumerate(queries):
      potential_target = dat[targets[closest_targets[ii]]]
      if neg_nns[ind] is None or np.abs(neg_nns[ind][2].astype(int) - \
                                        int(all_X[ind, 2])) > \
                                 np.abs(potential_target[2].astype(int) - \
                                        int(all_X[ind, 2])):
        if int(potential_target[2]) != int(all_X[ind, 2]):
          neg_nns[ind] = potential_target