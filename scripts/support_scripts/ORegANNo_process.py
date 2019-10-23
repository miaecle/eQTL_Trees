#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:53:05 2019

@author: zqwu
"""

import numpy as np

f_name = 'ORegAnno_Combined_2016.01.19.tsv'
lines = []
with open(f_name, 'r') as f:
    for i, line in enumerate(f):
        if i==0: head = line.split('\t')
        line = line.split('\t')
        if line[-5] == 'hg19':
            lines.append(line)

positives = np.array([l for l in lines if l[2] == 'POSITIVE OUTCOME' and l[3] == 'REGULATORY POLYMORPHISM' and l[10] != 'N/A' and int(l[16]) == int(l[17])])

gene_IDs = set([l[5] for l in positives])
for g in gene_IDs:
  if not g.startswith('ENSG'):
    print(g)

gene_ID_mapping = {}
with open('./bioDBnet_db2db_190411193443_483925829.txt', 'r') as f:
  for i, line in enumerate(f):
    if i==0: continue
    line = line.split('\t')
    if ';' in line[1]:
      line[1] = line[1].split(';')[0]
    gene_ID_mapping[line[0]] = line[1]

for i, line in enumerate(positives):
  if not line[5].startswith('ENSG'):
    positives[i, 5] = gene_ID_mapping[positives[i, 5]]

positives = np.concatenate([positives[:, 0:1], positives[:, 5:6], positives[:, 10:11], positives[:, 15:18]], 1)

import os
import json

pairs = []
for line in positives:
  rs_id = line[2]
  os.system('curl -X GET "https://api.ncbi.nlm.nih.gov/variation/v0/beta/refsnp/%s" -H "accept: application/json" > test.json' % rs_id[2:])
  dat = json.load(open('test.json', 'r'))
  dat = dat['present_obs_movements']
  
  try:
    inserted = set()
    deleted = set()
    for d in dat:
      d = d['observation']
      inserted.add(d[u'inserted_sequence'])
      deleted.add(d[u'deleted_sequence'])
    assert len(deleted) == 1
    inserted = list(inserted - deleted)[0]
    deleted = list(deleted)[0]
    
    var_name = str(line[3][3:] + '_' + line[4] + '_' + deleted + '_' + inserted +'_' + 'b37')
    pairs.append((line[1], var_name))
  except:
    print(rs_id)
  