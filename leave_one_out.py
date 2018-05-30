# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from xgboost import XGBClassifier
#from simple_model import NNmodel
import argparse
import copy

CHROM_LEAVEOUT = [['1'], ['2'], ['3'], ['4'], ['5'], ['6'], ['7'], ['8'], ['9'], ['10'],
                  ['11'], ['12'], ['13'], ['14'], ['15'], ['16'], ['17'], ['18'], ['19'],
                  ['20'], ['21'], ['22'], ['X']]

pos_samples = './LCL_pos_samples_180510.csv'
neg_samples = './LCL_neg_samples_180510.csv'
inter_samples = None

model_name = 'xgb'

if model_name == 'rf':
  model_ins = RandomForestClassifier(n_estimators=500)
elif model_name == 'logreg':
  model_ins = LogisticRegression()
elif model_name == 'xgb':
  model_ins = XGBClassifier(n_estimators=500, max_depth=5)

pos_data_name = os.path.splitext(pos_samples)[0]
neg_data_name = os.path.splitext(neg_samples)[0]
if not inter_samples is None:
  inter_data_name = os.path.splitext(inter_samples)[0]



with open(pos_data_name+'.pkl', 'r') as f:
  pos_data = pickle.load(f)
  n_pos_samples = pos_data.shape[0]
with open(neg_data_name+'.pkl', 'r') as f:
  neg_data = pickle.load(f)
  n_neg_samples = neg_data.shape[0]
if not inter_samples is None:
  with open(inter_data_name+'.pkl', 'r') as f:
    inter_data = pickle.load(f)
    n_inter_samples = inter_data.shape[0]


pos_d = np.array(pd.read_csv(pos_data_name+'.csv'))
neg_d = np.array(pd.read_csv(neg_data_name+'.csv'))
if not inter_samples is None:
  inter_d = np.array(pd.read_csv(inter_data_name+'.csv'))

pos_chrom_id = pos_d[:, 1]
pos_chrom_id = [chrom_id.split('_')[0] for chrom_id in pos_chrom_id]
neg_chrom_id = neg_d[:, 1]
neg_chrom_id = [chrom_id.split('_')[0] for chrom_id in neg_chrom_id]
if not inter_samples is None:
  inter_chrom_id = inter_d[:, 1]
  inter_chrom_id = [chrom_id.split('_')[0] for chrom_id in inter_chrom_id]


if not inter_samples is None:
  all_X = np.concatenate([pos_data, inter_data, neg_data], axis=0)
  all_X_d = np.concatenate([np.array(pos_d), np.array(inter_d), np.array(neg_d)], axis=0)
  all_y = np.array([1]*n_pos_samples + [0]*n_inter_samples + [0]*n_neg_samples)
  all_chrom_id = np.concatenate([pos_chrom_id, inter_chrom_id, neg_chrom_id])
else:
  all_X = np.concatenate([pos_data, neg_data], axis=0)
  all_X_d = np.concatenate([np.array(pos_d), np.array(neg_d)], axis=0)
  all_y = np.array([1]*n_pos_samples + [0]*n_neg_samples)
  all_chrom_id = np.concatenate([pos_chrom_id, neg_chrom_id])


assert all_X.shape[0] == all_y.shape[0] == all_chrom_id.shape[0]
n_samples = all_y.shape[0]


inds = np.arange(all_X.shape[0])
np.random.seed(123)
np.random.shuffle(inds)
X = all_X[inds, :]
X[:, 126] = 0
X = normalize(X, axis=0)
y = all_y[inds]
X_d = all_X_d[inds]
chrom_id = all_chrom_id[inds]

X_dists = X[:, 93:94]
X_variants = np.concatenate([X[:, :93], X[:, 94:126], X[:, 127:129]], axis=1)
X_genes = X[:, 129:]
#Interaction_term = np.expand_dims(X_variants, axis=2) * np.expand_dims(X_genes, axis=1)
#Interaction_term = np.reshape(Interaction_term, (Interaction_term.shape[0], -1))
#X = np.concatenate([X, Interaction_term], axis=1)


full_scores = []
variant_scores = []
gene_scores = []
dist_scores = []
dist_variant_scores = []
dist_gene_scores = []
variant_gene_scores = []

full_models = []
variant_models = []
gene_models = []
dist_models = []
dist_variant_models = []
dist_gene_models = []

for ch_leaveout in CHROM_LEAVEOUT:
    n_leaveout = []
    for ch in ch_leaveout:
      n_leaveout.append(np.where(chrom_id == ch)[0])
    valid_inds = np.concatenate(n_leaveout)
    train_inds = [ind for ind in np.arange(y.shape[0]) if ind not in valid_inds]
    
    train_y = y[train_inds]
    train_X = X[train_inds]
    train_X_dists = X_dists[train_inds]
    train_X_variants = X_variants[train_inds]
    train_X_genes = X_genes[train_inds]

    valid_y = y[valid_inds]
    valid_X = X[valid_inds]
    valid_X_dists = X_dists[valid_inds]
    valid_X_variants = X_variants[valid_inds]
    valid_X_genes = X_genes[valid_inds]


    model = copy.deepcopy(model_ins)
    model.fit(train_X, train_y)
    valid_y_pred = model.predict_proba(valid_X)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    #full_models.append(model)
    full_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                        f1_score(valid_y, valid_y_prediction)))
    
    # VARIANT
    model = copy.deepcopy(model_ins)
    model.fit(train_X_variants, train_y)
    valid_y_pred = model.predict_proba(valid_X_variants)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    #variant_models.append(model)
    variant_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                        f1_score(valid_y, valid_y_prediction)))
    
    # GENE
    model = copy.deepcopy(model_ins)
    model.fit(train_X_genes, train_y)
    valid_y_pred = model.predict_proba(valid_X_genes)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    #gene_models.append(model)
    gene_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                        f1_score(valid_y, valid_y_prediction)))
    
    # DIST
    model = copy.deepcopy(model_ins)
    model.fit(train_X_dists, train_y)
    valid_y_pred = model.predict_proba(valid_X_dists)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    #dist_models.append(model)
    dist_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                        f1_score(valid_y, valid_y_prediction)))
    
    # DIST+VARIANT
    model = copy.deepcopy(model_ins)
    model.fit(np.concatenate([train_X_variants, train_X_dists], axis=1), train_y)
    valid_y_pred = model.predict_proba(np.concatenate([valid_X_variants, valid_X_dists], axis=1))
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    #dist_variant_models.append(model)
    dist_variant_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                                f1_score(valid_y, valid_y_prediction)))
    
    # DIST+GENE
    model = copy.deepcopy(model_ins)
    model.fit(np.concatenate([train_X_genes, train_X_dists], axis=1), train_y)
    valid_y_pred = model.predict_proba(np.concatenate([valid_X_genes, valid_X_dists], axis=1))
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    #dist_gene_models.append(model)
    dist_gene_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                             f1_score(valid_y, valid_y_prediction)))
    
    #VARIANT+GENE
    model = copy.deepcopy(model_ins)
    model.fit(np.concatenate([train_X_variants, train_X_genes], axis=1), train_y)
    valid_y_pred = model.predict_proba(np.concatenate([valid_X_variants, valid_X_genes], axis=1))
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    variant_gene_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                        f1_score(valid_y, valid_y_prediction)))

print(np.mean(np.array(full_scores), axis=0))
print(np.mean(np.array(variant_scores), axis=0))
print(np.mean(np.array(gene_scores), axis=0))
print(np.mean(np.array(dist_scores), axis=0))
print(np.mean(np.array(dist_variant_scores), axis=0))
print(np.mean(np.array(dist_gene_scores), axis=0))