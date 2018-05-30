#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 22:41:37 2018

@author: zqwu
"""
import argparse
import pandas as pd
import numpy as np
import pickle
import os
import csv

var_keys = [
 'Consequence',
 'GC',
 'CpG',
 'priPhCons',
 'mamPhCons',
 'verPhCons',
 'priPhyloP',
 'mamPhyloP',
 'verPhyloP',
 'GerpN',
 'GerpS',
 'bStatistic',
 'mutIndex',
 'dnaHelT',
 'dnaMGW',
 'dnaProT',
 'dnaRoll',
 'mirSVR',
 'cHmmTssA',
 'cHmmTssAFlnk',
 'cHmmTxFlnk',
 'cHmmTx',
 'cHmmTxWk',
 'cHmmEnhG',
 'cHmmEnh',
 'cHmmZnfRpts',
 'cHmmHet',
 'cHmmTssBiv',
 'cHmmBivFlnk',
 'cHmmEnhBiv',
 'cHmmReprPC',
 'cHmmReprPCWk',
 'cHmmQuies',
 'EncExp',
 'EncH3K27Ac',
 'EncH3K4Me1',
 'EncH3K4Me3',
 'EncNucleo',
 'EncOCC',
 'EncOCCombPVal',
 'EncOCDNasePVal',
 'EncOCFairePVal',
 'EncOCpolIIPVal',
 'EncOCctcfPVal',
 'EncOCmycPVal',
 'EncOCDNaseSig',
 'EncOCFaireSig',
 'EncOCpolIISig',
 'EncOCctcfSig',
 'EncOCmycSig',
 'Segway',
 'tOverlapMotifs',
 'TFBS',
 'TFBSPeaks',
 'TFBSPeaksMax',
 'disTSS',
 'CTCF',
 'DNase',
 'FAIRE',
 'H2AFZ',
 'H3K27ac',
 'H3K27me3',
 'H3K36me3',
 'H3K4me1',
 'H3K4me2',
 'H3K4me3',
 'H3K79me2',
 'H3K9ac',
 'H3K9me1',
 'H3K9me3',
 'H4K20me1',
 'MYC',
 'POLR2A',
 'POLR2A_elongating',
 'POLR3A',
 'pwm',
 'cpg_island',
 'avg_gerp',
 'dnase_fps',
 'UTR5',
 'INTRON',
 'STOP',
 'UTR3',
 'START',
 'EXON',
 'CDS',
 'DONOR',
 'ACCEPTOR',
 'ss_dist',
 'in_cpg',
 'repeat']

gene_keys = ['GC',
 'CpG',
 'cHmmTssA',
 'cHmmTssAFlnk',
 'cHmmTxFlnk',
 'cHmmTx',
 'cHmmTxWk',
 'cHmmEnhG',
 'cHmmEnh',
 'cHmmZnfRpts',
 'cHmmHet',
 'cHmmTssBiv',
 'cHmmBivFlnk',
 'cHmmEnhBiv',
 'cHmmReprPC',
 'cHmmReprPCWk',
 'cHmmQuies',
 'EncExp',
 'EncH3K27Ac',
 'EncH3K4Me1',
 'EncH3K4Me3',
 'EncNucleo',
 'EncOCC',
 'EncOCCombPVal',
 'EncOCDNasePVal',
 'EncOCFairePVal',
 'EncOCpolIIPVal',
 'EncOCctcfPVal',
 'EncOCmycPVal',
 'EncOCDNaseSig',
 'EncOCFaireSig',
 'EncOCpolIISig',
 'EncOCctcfSig',
 'EncOCmycSig',
 'Segway',
 'tOverlapMotifs',
 'TFBS',
 'TFBSPeaks',
 'TFBSPeaksMax',
 'CTCF',
 'DNase',
 'FAIRE',
 'H2AFZ',
 'H3K27ac',
 'H3K27me3',
 'H3K36me3',
 'H3K4me1',
 'H3K4me2',
 'H3K4me3',
 'H3K79me2',
 'H3K9ac',
 'H3K9me1',
 'H3K9me3',
 'H4K20me1',
 'MYC',
 'POLR2A',
 'POLR2A_elongating',
 'POLR3A',
 'pwm',
 'cpg_island',
 'avg_gerp',
 'dnase_fps',
 'in_cpg',
 'repeat']
 
parser = argparse.ArgumentParser(
    description='eQTL_sampler')
parser.add_argument(
    '-i',
    action='append',
    dest='inputs',
    default=[],
    help='inputs')
parser.add_argument(
    '-v',
    action='append',
    dest='variant_data',
    default=[],
    help='input_variant_data')
parser.add_argument(
    '-g',
    action='append',
    dest='gene_data',
    default=[],
    help='input_gene_data')
parser.add_argument(
    '-o',
    action='append',
    dest='outfile',
    default=[],
    help='output_file')

args = parser.parse_args()
infiles = args.inputs
v_data_names = args.variant_data
g_data_names = args.gene_data
outfiles = args.outfile
 
for i, infile in enumerate(infiles):
  variants_feat = []
  genes_feat = []
  with open(v_data_names[i], 'r') as f:
    variants_data = pickle.load(f)
  with open(g_data_names[i], 'r') as f:
    genes_data = pickle.load(f)
  
  with open(infile, 'r') as f:
    header = f.readline()
    reader = csv.reader(f)
    for line in reader:
      gene_id = line[0]
      variant_id = line[1]
      tss_dis = int(line[2])
      
      var_feat = []
      var_chr = variant_id.split('_')[0]
      var_pos = int(variant_id.split('_')[1])
      num_id = np.where(variants_data['Pos'] == var_pos)[0]
      if len(num_id) == 1:
        num_id = num_id[0]
      else:
        for num in num_id:
          if str(variants_data['Chrom'][num, 0]) == var_chr:
            num_id = num
            break
      assert num_id.__class__ is np.int64
      for key in var_keys:
        if key == 'disTSS':
          var_feat.append(np.reshape(np.array(tss_dis), (1,)))
        else:
          var_feat.append(variants_data[key][num_id])
      var_feat = np.concatenate(var_feat)
        
      gene_feat = []
      gene_num_id = np.where(np.array(genes_data['gene_name']) == gene_id)[0]
      assert len(gene_num_id) >= 1
      gene_num_id = gene_num_id[0]
      for key in gene_keys:
        gene_feat.append(genes_data[key][gene_num_id])
      gene_feat = np.concatenate(gene_feat)
      
      variants_feat.append(var_feat)
      genes_feat.append(gene_feat)
    variants_feat = np.stack(variants_feat, axis=0)
    genes_feat = np.stack(genes_feat, axis=0)
    all_feat = np.concatenate([variants_feat, genes_feat], axis=1)
    assert all_feat.shape[1] == 218
  with open(outfiles[i], 'w') as f:
    pickle.dump(all_feat, f)
