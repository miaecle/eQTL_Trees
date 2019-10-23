#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:19:51 2018

@author: zqwu
"""
import os
import pickle


inter_feats = ['GC_p',
'H3K27me3_p',
'cpg_island_p',
'TRAN_p',
'avg_daf_p',
'H3K4me3_p',
'H2AFZ_p',
'POLR2A_p',
'WEAK_ENH_p',
'TSS_p',
'in_cpg_p',
'CTCF_REG_p',
'avg_het_p',
'ZNF274_p',
'YY1_p',
'REP_p',
'CTCF_p']

inter_feat_inds = np.array([feat_name[320:].index(feat) for feat in inter_feats])

pairs = [(p[0], p[1]) for p in all_dat]         
inter_feats = [None] * len(pairs)

inter_feat_files = [path for path in os.listdir('.') if path.endswith('.pkl2')]
for f_n in inter_feat_files:
  dat = pickle.load(open(f_n, 'r'))
  for key in dat:
    inter_feats[pairs.index(key)] = dat[key][:, inter_feat_inds]
