#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 21:59:37 2018

@author: zqwu
"""

import copy
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from generate_X import feat_name

def cv(model_ins, X, y, cv_inds, name, feat_name=feat_name):
  
  v_feat_inds = [i for i in range(X.shape[1]) if feat_name[i].endswith('_v')]
  g_feat_inds = [i for i in range(X.shape[1]) if feat_name[i].endswith('_g')]
  p_feat_inds = [i for i in range(X.shape[1]) if feat_name[i].endswith('_p')]
  X_v = X[:, np.array(v_feat_inds)]
  X_g = X[:, np.array(g_feat_inds)]
  X_p = X[:, np.array(p_feat_inds)]
  
  full_scores = []
  v_scores = []
  g_scores = []
  p_scores = []
  p_v_scores = []
  p_g_scores = []
  v_g_scores = []
  
  full_models = []
  v_models = []
  g_models = []
  p_models = []
  p_v_models = []
  p_g_models = []
  v_g_models = []

  for i, (train_inds, valid_inds) in enumerate(cv_inds):
    print("Starting fold %d" % i)
    train_y = y[train_inds]
    train_X = X[train_inds]
    train_X_p = X_p[train_inds]
    train_X_v = X_v[train_inds]
    train_X_g = X_g[train_inds]

    # dat includes selection of valid samples, 1 represents excluded
    #valid_inds = valid_inds[np.where(dat[valid_inds] == 0)[0]]
    valid_y = y[valid_inds]
    valid_X = X[valid_inds]
    valid_X_p = X_p[valid_inds]
    valid_X_v = X_v[valid_inds]
    valid_X_g = X_g[valid_inds]

    model = copy.deepcopy(model_ins)
    model.fit(train_X, train_y)
    valid_y_pred = model.predict_proba(valid_X)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    full_models.append(model)
    full_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                        f1_score(valid_y, valid_y_prediction),
                        precision_score(valid_y, valid_y_prediction),
                        recall_score(valid_y, valid_y_prediction),
                        accuracy_score(valid_y, valid_y_prediction)))
    print("FULL: " + str(full_scores[-1]))
    
    # VARIANT
    model = copy.deepcopy(model_ins)
    model.fit(train_X_v, train_y)
    valid_y_pred = model.predict_proba(valid_X_v)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    v_models.append(model)
    v_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                     f1_score(valid_y, valid_y_prediction),
                     precision_score(valid_y, valid_y_prediction),
                     recall_score(valid_y, valid_y_prediction),
                     accuracy_score(valid_y, valid_y_prediction)))
    print("VAR: " + str(v_scores[-1]))
    
    # GENE
    model = copy.deepcopy(model_ins)
    model.fit(train_X_g, train_y)
    valid_y_pred = model.predict_proba(valid_X_g)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    g_models.append(model)
    g_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                     f1_score(valid_y, valid_y_prediction),
                     precision_score(valid_y, valid_y_prediction),
                     recall_score(valid_y, valid_y_prediction),
                     accuracy_score(valid_y, valid_y_prediction)))
    print("GENE: " + str(g_scores[-1]))
    
    # PAIR
    model = copy.deepcopy(model_ins)
    model.fit(train_X_p, train_y)
    valid_y_pred = model.predict_proba(valid_X_p)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    p_models.append(model)
    p_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                     f1_score(valid_y, valid_y_prediction),
                     precision_score(valid_y, valid_y_prediction),
                     recall_score(valid_y, valid_y_prediction),
                     accuracy_score(valid_y, valid_y_prediction)))
    print("PAIR: " + str(p_scores[-1]))
    
    # DIST+VARIANT
    model = copy.deepcopy(model_ins)
    model.fit(np.concatenate([train_X_v, train_X_p], axis=1), train_y)
    valid_y_pred = model.predict_proba(np.concatenate([valid_X_v, valid_X_p], axis=1))
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    p_v_models.append(model)
    p_v_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                       f1_score(valid_y, valid_y_prediction),
                       precision_score(valid_y, valid_y_prediction),
                       recall_score(valid_y, valid_y_prediction),
                       accuracy_score(valid_y, valid_y_prediction)))
    print("PAIR+VAR: " + str(p_v_scores[-1]))
    
    # DIST+GENE
    model = copy.deepcopy(model_ins)
    model.fit(np.concatenate([train_X_g, train_X_p], axis=1), train_y)
    valid_y_pred = model.predict_proba(np.concatenate([valid_X_g, valid_X_p], axis=1))
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    p_g_models.append(model)
    p_g_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                       f1_score(valid_y, valid_y_prediction),
                       precision_score(valid_y, valid_y_prediction),
                       recall_score(valid_y, valid_y_prediction),
                       accuracy_score(valid_y, valid_y_prediction)))
    print("PAIR+GENE: " + str(p_g_scores[-1]))

    # VAR+GENE
    model = copy.deepcopy(model_ins)
    model.fit(np.concatenate([train_X_v, train_X_g], axis=1), train_y)
    valid_y_pred = model.predict_proba(np.concatenate([valid_X_v, valid_X_g], axis=1))
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    v_g_models.append(model)
    v_g_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                       f1_score(valid_y, valid_y_prediction),
                       precision_score(valid_y, valid_y_prediction),
                       recall_score(valid_y, valid_y_prediction),
                       accuracy_score(valid_y, valid_y_prediction)))
    print("VAR+GENE: " + str(v_g_scores[-1]))
#    with open(name + "_models.pkl", 'w') as f:
#      pickle.dump({'FULL':full_models, 'VARIANT':v_models, 'GENE': g_models,
#      'PAIR': p_models, 'PAIR+VARIANT':p_v_models, 'PAIR+GENE':p_g_models, 
#      'VARIANT+GENE':v_g_models}, f)

  return full_scores, v_scores, g_scores, p_scores, p_v_scores, p_g_scores, v_g_scores
  
def cv_validate(model_ins, X, y, cv_inds, name):

  with open(name + "_models.pkl", 'r') as f:
    models = pickle.load(f)
  
  v_feat_inds = [i for i in range(X.shape[1]) if feat_name[i].endswith('_v')]
  g_feat_inds = [i for i in range(X.shape[1]) if feat_name[i].endswith('_g')]
  p_feat_inds = [i for i in range(X.shape[1]) if feat_name[i].endswith('_p')]
  X_v = X[:, np.array(v_feat_inds)]
  X_g = X[:, np.array(g_feat_inds)]
  X_p = X[:, np.array(p_feat_inds)]
  
  full_scores = []
  v_scores = []
  g_scores = []
  p_scores = []
  p_v_scores = []
  p_g_scores = []
  v_g_scores = []
  
  full_models = models['FULL']
  v_models = models['VARIANT']
  g_models = models['GENE']
  p_models = models['PAIR']
  p_v_models = models['PAIR+VARIANT']
  p_g_models = models['PAIR+GENE']
  v_g_models = models['VARIANT+GENE']  

  for i, (train_inds, valid_inds) in enumerate(cv_inds):
    print("Starting fold %d" % i)
    #train_y = y[train_inds]
    #train_X = X[train_inds]
    #train_X_p = X_p[train_inds]
    #train_X_v = X_v[train_inds]
    #train_X_g = X_g[train_inds]

    # dat includes selection of valid samples, 1 represents excluded
    #valid_inds = valid_inds[np.where(dat[valid_inds] == 0)[0]]
    valid_y = y[valid_inds]
    valid_X = X[valid_inds]
    valid_X_p = X_p[valid_inds]
    valid_X_v = X_v[valid_inds]
    valid_X_g = X_g[valid_inds]

    model = full_models[i]
    valid_y_pred = model.predict_proba(valid_X)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    full_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                        f1_score(valid_y, valid_y_prediction),
                        precision_score(valid_y, valid_y_prediction),
                        recall_score(valid_y, valid_y_prediction),
                        accuracy_score(valid_y, valid_y_prediction)))
    print("FULL: " + str(full_scores[-1]))
    
    # VARIANT
    model = v_models[i]
    valid_y_pred = model.predict_proba(valid_X_v)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    v_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                     f1_score(valid_y, valid_y_prediction),
                     precision_score(valid_y, valid_y_prediction),
                     recall_score(valid_y, valid_y_prediction),
                     accuracy_score(valid_y, valid_y_prediction)))
    print("VAR: " + str(v_scores[-1]))
    
    # GENE
    model = g_models[i]
    valid_y_pred = model.predict_proba(valid_X_g)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    g_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                     f1_score(valid_y, valid_y_prediction),
                     precision_score(valid_y, valid_y_prediction),
                     recall_score(valid_y, valid_y_prediction),
                     accuracy_score(valid_y, valid_y_prediction)))
    print("GENE: " + str(g_scores[-1]))
    
    # PAIR
    model = p_models[i]
    valid_y_pred = model.predict_proba(valid_X_p)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    p_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                     f1_score(valid_y, valid_y_prediction),
                     precision_score(valid_y, valid_y_prediction),
                     recall_score(valid_y, valid_y_prediction),
                     accuracy_score(valid_y, valid_y_prediction)))
    print("PAIR: " + str(p_scores[-1]))
    
    # DIST+VARIANT
    model = p_v_models[i]
    valid_y_pred = model.predict_proba(np.concatenate([valid_X_v, valid_X_p], axis=1))
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    p_v_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                       f1_score(valid_y, valid_y_prediction),
                       precision_score(valid_y, valid_y_prediction),
                       recall_score(valid_y, valid_y_prediction),
                       accuracy_score(valid_y, valid_y_prediction)))
    print("PAIR+VAR: " + str(p_v_scores[-1]))
    
    # DIST+GENE
    model = p_g_models[i]
    valid_y_pred = model.predict_proba(np.concatenate([valid_X_g, valid_X_p], axis=1))
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    p_g_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                       f1_score(valid_y, valid_y_prediction),
                       precision_score(valid_y, valid_y_prediction),
                       recall_score(valid_y, valid_y_prediction),
                       accuracy_score(valid_y, valid_y_prediction)))
    print("PAIR+GENE: " + str(p_g_scores[-1]))

    # VAR+GENE
    model = v_g_models[i]
    valid_y_pred = model.predict_proba(np.concatenate([valid_X_v, valid_X_g], axis=1))
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    v_g_scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                       f1_score(valid_y, valid_y_prediction),
                       precision_score(valid_y, valid_y_prediction),
                       recall_score(valid_y, valid_y_prediction),
                       accuracy_score(valid_y, valid_y_prediction)))
    print("VAR+GENE: " + str(v_g_scores[-1]))

  return full_scores, v_scores, g_scores, p_scores, p_v_scores, p_g_scores, v_g_scores

def feature_importance_test(model_ins, X, y, cv_inds, feat_order):
  scores = []
  train_inds = cv_inds[0]
  valid_inds = cv_inds[1]
  for run in range(len(feat_order)):
    X_ = X[:, np.array(feat_order[:(run+1)])]
    print("Starting fold %d" % run)
    train_y = y[train_inds]
    train_X_ = X_[train_inds]

    valid_y = y[valid_inds]
    valid_X_ = X_[valid_inds]

    model = copy.deepcopy(model_ins)
    model.fit(train_X_, train_y)
    valid_y_pred = model.predict_proba(valid_X_)
    valid_y_prediction = np.argmax(valid_y_pred, axis=1)
    
    scores.append((roc_auc_score(valid_y, valid_y_pred[:, 1]),
                   f1_score(valid_y, valid_y_prediction),
                   precision_score(valid_y, valid_y_prediction),
                   recall_score(valid_y, valid_y_prediction),
                   accuracy_score(valid_y, valid_y_prediction)))
    print("fold %d: "  % run + str(scores[-1]))
  return scores
