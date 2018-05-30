# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from xgboost import XGBClassifier
import argparse
import copy

pos_samples = 'LCL_pos_samples_180510.csv'
neg_samples = 'LCL_neg_samples_180510.csv'
inter_samples = None
model_name = 'xgb'
use_dis = True

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

if not inter_samples is None:
  all_X = np.concatenate([pos_data, inter_data, neg_data], axis=0)
  all_y = np.array([1]*n_pos_samples + [0]*n_inter_samples + [0]*n_neg_samples)
else:
  all_X = np.concatenate([pos_data, neg_data], axis=0)
  all_y = np.array([1]*n_pos_samples + [0]*n_neg_samples)

if not use_dis:
  all_X[:, 93] = 0
  all_X[:, 126] = 0

with open('./nn_' + pos_data_name + neg_data_name + '.pkl', 'r') as f:
  all_nn_features = pickle.load(f)

with open('./state_' + pos_data_name + neg_data_name + '.pkl', 'r') as f:
  all_states = pickle.load(f)

inds = np.arange(all_X.shape[0])
np.random.seed(123)
np.random.shuffle(inds)
X = all_X[inds, :]
X[:, 126] = 0
X = normalize(X, axis=0)
y = all_y[inds]
nn_features = [feat[inds] for feat in all_nn_features]
states = all_states[inds]

X_dists = X[:, 93:94]
X_variants = np.concatenate([X[:, :93], X[:, 94:126], X[:, 127:129]], axis=1)
X_genes = X[:, 129:]

pos_inds = np.where(y==1)[0]
neg_inds = np.where(y==0)[0]
K = 10
pos_valid = np.linspace(0, len(pos_inds), K+1, dtype=int)
neg_valid = np.linspace(0, len(neg_inds), K+1, dtype=int)

full_scores = []
variant_scores = []
gene_scores = []
dist_scores = []
dist_variant_scores = []
dist_gene_scores = []
full_nn_scores = []
nn_scores = []

full_models = []
variant_models = []
gene_models = []
dist_models = []
dist_variant_models = []
dist_gene_models = []
full_nn_models = []

for i in range(K):
    print("On Split %d" % i)
    valid_inds = pos_inds[pos_valid[i]:pos_valid[i+1]]
    valid_inds = np.concatenate([valid_inds,neg_inds[neg_valid[i]:neg_valid[i+1]]])
    valid_inds.sort()
    train_inds = np.concatenate([pos_inds[:pos_valid[i]], pos_inds[pos_valid[i+1]:]])
    train_inds = np.concatenate([train_inds, neg_inds[:neg_valid[i]], neg_inds[neg_valid[i+1]:]])
    train_inds.sort()
    
    train_y = y[train_inds]
    train_X = X[train_inds]
    train_X_dists = X_dists[train_inds]
    train_X_variants = X_variants[train_inds]
    train_X_genes = X_genes[train_inds]
    train_X_nn = nn_features[i][train_inds]

    # dat includes selection of valid samples, 1 represents excluded
    #valid_inds = valid_inds[np.where(dat[valid_inds] == 0)[0]]
    valid_y = y[valid_inds]
    valid_X = X[valid_inds]
    valid_X_dists = X_dists[valid_inds]
    valid_X_variants = X_variants[valid_inds]
    valid_X_genes = X_genes[valid_inds]
    valid_X_nn = nn_features[i][valid_inds]

    abnorm_inds = np.where(states[valid_inds] == 0)[0]

    # FULL
    model = copy.deepcopy(model_ins)
    model.fit(train_X, train_y)
    valid_y_pred = model.predict_proba(valid_X)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    full_models.append(model)
    full_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                        f1_score(valid_y, valid_y_prediction),
                        roc_auc_score(valid_y[abnorm_inds], valid_y_pred[abnorm_inds, 1]),
                        f1_score(valid_y[abnorm_inds], valid_y_prediction[abnorm_inds])))
    
    # VARIANT
    model = copy.deepcopy(model_ins)
    model.fit(train_X_variants, train_y)
    valid_y_pred = model.predict_proba(valid_X_variants)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    variant_models.append(model)
    variant_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                        f1_score(valid_y, valid_y_prediction),
                        roc_auc_score(valid_y[abnorm_inds], valid_y_pred[abnorm_inds, 1]),
                        f1_score(valid_y[abnorm_inds], valid_y_prediction[abnorm_inds])))
    
    # GENE
    model = copy.deepcopy(model_ins)
    model.fit(train_X_genes, train_y)
    valid_y_pred = model.predict_proba(valid_X_genes)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    gene_models.append(model)
    gene_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                        f1_score(valid_y, valid_y_prediction),
                        roc_auc_score(valid_y[abnorm_inds], valid_y_pred[abnorm_inds, 1]),
                        f1_score(valid_y[abnorm_inds], valid_y_prediction[abnorm_inds])))
    
    # DIST
    model = copy.deepcopy(model_ins)
    model.fit(train_X_dists, train_y)
    valid_y_pred = model.predict_proba(valid_X_dists)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    dist_models.append(model)
    dist_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                        f1_score(valid_y, valid_y_prediction),
                        roc_auc_score(valid_y[abnorm_inds], valid_y_pred[abnorm_inds, 1]),
                        f1_score(valid_y[abnorm_inds], valid_y_prediction[abnorm_inds])))
    
    # DIST+VARIANT
    model = copy.deepcopy(model_ins)
    model.fit(np.concatenate([train_X_variants, train_X_dists], axis=1), train_y)
    valid_y_pred = model.predict_proba(np.concatenate([valid_X_variants, valid_X_dists], axis=1))
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    dist_variant_models.append(model)
    dist_variant_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                                f1_score(valid_y, valid_y_prediction),
                                roc_auc_score(valid_y[abnorm_inds], valid_y_pred[abnorm_inds, 1]),
                                f1_score(valid_y[abnorm_inds], valid_y_prediction[abnorm_inds])))
    
    # DIST+GENE
    model = copy.deepcopy(model_ins)
    model.fit(np.concatenate([train_X_genes, train_X_dists], axis=1), train_y)
    valid_y_pred = model.predict_proba(np.concatenate([valid_X_genes, valid_X_dists], axis=1))
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    dist_gene_models.append(model)
    dist_gene_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                             f1_score(valid_y, valid_y_prediction),
                             roc_auc_score(valid_y[abnorm_inds], valid_y_pred[abnorm_inds, 1]),
                             f1_score(valid_y[abnorm_inds], valid_y_prediction[abnorm_inds])))
    
    # FULL+NN
    model = copy.deepcopy(model_ins)
    model.fit(np.concatenate([train_X, train_X_nn], axis=1), train_y)
    valid_y_pred = model.predict_proba(np.concatenate([valid_X, valid_X_nn], axis=1))
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    full_nn_models.append(model)
    full_nn_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                           f1_score(valid_y, valid_y_prediction),
                           roc_auc_score(valid_y[abnorm_inds], valid_y_pred[abnorm_inds, 1]),
                           f1_score(valid_y[abnorm_inds], valid_y_prediction[abnorm_inds])))

    nn_scores.append((f1_score(valid_y, valid_X_nn[:, 0]),
                      f1_score(valid_y[abnorm_inds], valid_X_nn[:, 0][abnorm_inds])))
        
    with open("model_" + pos_data_name + neg_data_name+".pkl", 'w') as f:
      pickle.dump({'FULL':full_models, 'VARIANT':variant_models, 'GENE': gene_models, 
      'DIST': dist_models, 'DIST+VARIANT':dist_variant_models, 'DIST+GENE':dist_gene_models, 'FULL+NN': full_nn_models}, f)
    
print(np.mean(np.array(full_scores), axis=0))
print(np.mean(np.array(variant_scores), axis=0))
print(np.mean(np.array(gene_scores), axis=0))
print(np.mean(np.array(dist_scores), axis=0))
print(np.mean(np.array(dist_variant_scores), axis=0))
print(np.mean(np.array(dist_gene_scores), axis=0))
print(np.mean(np.array(full_nn_scores), axis=0))
print(np.mean(np.array(nn_scores), axis=0))
