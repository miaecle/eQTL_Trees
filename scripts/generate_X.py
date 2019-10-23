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

var_CADD_keys = [
 ('Consequence', 'INTERGENIC'),
 ('GC', 0.42),
 ('CpG', 0.02),
 ('motifECount', 0),
 ('priPhCons', 0.115),
 ('mamPhCons', 0.079),
 ('verPhCons', 0.094),
 ('priPhyloP', -0.033),
 ('mamPhyloP', -0.038),
 ('verPhyloP', 0.017),
 ('bStatistic', 800),
 ('cHmmTssA', 0.0667),
 ('cHmmTssAFlnk', 0.0667),
 ('cHmmTxFlnk', 0.0667),
 ('cHmmTx', 0.0667),
 ('cHmmTxWk', 0.0667),
 ('cHmmEnhG', 0.0667),
 ('cHmmEnh', 0.0667),
 ('cHmmZnfRpts', 0.0667),
 ('cHmmHet', 0.0667),
 ('cHmmTssBiv', 0.0667),
 ('cHmmBivFlnk', 0.0667),
 ('cHmmEnhBiv', 0.0667),
 ('cHmmReprPC', 0.0667),
 ('cHmmReprPCWk', 0.0667),
 ('cHmmQuies', 0.0667),
 ('GerpN', 1.91),
 ('GerpS', -0.2),
 ('TFBS', 0),
 ('TFBSPeaks', 0),
 ('TFBSPeaksMax', 0),
 ('tOverlapMotifs', 0),
 ('motifDist', 0),
 ('Segway', 'UNKNOWN'),
 ('EncH3K27Ac', 0),
 ('EncH3K4Me1', 0),
 ('EncH3K4Me3', 0),
 ('EncExp', 0),
 ('EncNucleo', 0),
 ('EncOCC', 0),
 ('EncOCCombPVal', 0),
 ('EncOCDNasePVal', 0),
 ('EncOCFairePVal', 0),
 ('EncOCpolIIPVal', 0),
 ('EncOCctcfPVal', 0),
 ('EncOCmycPVal', 0),
 ('EncOCDNaseSig', 0),
 ('EncOCFaireSig', 0),
 ('EncOCpolIISig', 0),
 ('EncOCctcfSig', 0),
 ('EncOCmycSig', 0),
 ('Freq100bp', 0),
 ('Rare100bp', 0),
 ('Sngl100bp', 0),
 ('Freq1000bp', 0),
 ('Rare1000bp', 0),
 ('Sngl1000bp', 0),
 ('Freq10000bp', 0),
 ('Rare10000bp', 0),
 ('Sngl10000bp', 0),
 ('RawScore', 0),
 ('PHRED', 0)]

gene_CADD_keys = [
 ('CpG', 0.02),
 ('motifECount', 0),
 ('priPhCons', 0.115),
 ('mamPhCons', 0.079),
 ('verPhCons', 0.094),
 ('priPhyloP', -0.033),
 ('mamPhyloP', -0.038),
 ('verPhyloP', 0.017),
 ('bStatistic', 800),
 ('cHmmTssA', 0.0667),
 ('cHmmTssAFlnk', 0.0667),
 ('cHmmTxFlnk', 0.0667),
 ('cHmmTx', 0.0667),
 ('cHmmTxWk', 0.0667),
 ('cHmmEnhG', 0.0667),
 ('cHmmEnh', 0.0667),
 ('cHmmZnfRpts', 0.0667),
 ('cHmmHet', 0.0667),
 ('cHmmTssBiv', 0.0667),
 ('cHmmBivFlnk', 0.0667),
 ('cHmmEnhBiv', 0.0667),
 ('cHmmReprPC', 0.0667),
 ('cHmmReprPCWk', 0.0667),
 ('cHmmQuies', 0.0667),
 ('GerpN', 1.91),
 ('GerpS', -0.2),
 ('TFBS', 0),
 ('TFBSPeaks', 0),
 ('TFBSPeaksMax', 0),
 ('tOverlapMotifs', 0),
 ('motifDist', 0),
 ('Segway', 'UNKNOWN'),
 ('EncH3K27Ac', 0),
 ('EncH3K4Me1', 0),
 ('EncH3K4Me3', 0),
 ('EncExp', 0),
 ('EncNucleo', 0),
 ('EncOCC', 0),
 ('EncOCCombPVal', 0),
 ('EncOCDNasePVal', 0),
 ('EncOCFairePVal', 0),
 ('EncOCpolIIPVal', 0),
 ('EncOCctcfPVal', 0),
 ('EncOCmycPVal', 0),
 ('EncOCDNaseSig', 0),
 ('EncOCFaireSig', 0),
 ('EncOCpolIISig', 0),
 ('EncOCctcfSig', 0),
 ('EncOCmycSig', 0),
 ('Freq100bp', 0),
 ('Rare100bp', 0),
 ('Sngl100bp', 0),
 ('Freq1000bp', 0),
 ('Rare1000bp', 0),
 ('Sngl1000bp', 0),
 ('Freq10000bp', 0),
 ('Rare10000bp', 0),
 ('Sngl10000bp', 0),
 ('RawScore', 0),
 ('PHRED', 0)]

var_GWAVA_keys = [
 ('CTCF', 0),
 ('DNase', 0),
 ('FAIRE', 0),
 ('H2AFZ', 0),
 ('H3K27ac', 0),
 ('H3K27me3', 0),
 ('H3K36me3', 0),
 ('H3K4me1', 0),
 ('H3K4me2', 0),
 ('H3K4me3', 0),
 ('H3K79me2', 0),
 ('H3K9ac', 0),
 ('H3K9me1', 0),
 ('H3K9me3', 0),
 ('H4K20me1', 0),
 ('MYC', 0),
 ('POLR2A', 0),
 ('POLR2A_elongating', 0),
 ('POLR3A', 0),
 ('pwm', 0),
 ('cpg_island', 0),
 ('avg_gerp', 0),
 ('dnase_fps', 0), 
 ('in_cpg', 0),
 ('repeat', 0)]
 
pair_HiCs_keys = [
 ('HiCRAW_5kb', 0),
 ('HiCNormed_5kb', 0),
 ('HiCRAW_10kb', 0),
 ('HiCNormed_10kb', 0),
 ('HiCRAW_25kb', 0),
 ('HiCNormed_25kb', 0),
 ('HiCRAW_50kb', 0),
 ('HiCNormed_50kb', 0),
 ('HiCRAW_100kb', 0),
 ('HiCNormed_100kb', 0),
 ('HiCRAW_250kb', 0),
 ('HiCNormed_250kb', 0),
 ('HiCRAW_500kb', 0),
 ('HiCNormed_500kb', 0)]
 
pair_GWAVA_keys = [
 ('ATF3',0),
 ('BATF',0),
 ('BCL11A',0),
 ('BCL3',0),
 ('BCLAF1',0),
 ('BDP1',0),
 ('BHLHE40',0),
 ('BRCA1',0),
 ('BRF1',0),
 ('BRF2',0),
 ('CCNT2',0),
 ('CEBPB',0),
 ('CHD2',0),
 ('CTBP2',0),
 ('CTCF',0),
 ('CTCFL',0),
 ('E2F1',0),
 ('E2F4',0),
 ('E2F6',0),
 ('EBF1',0),
 ('EGR1',0),
 ('ELF1',0),
 ('ELK4',0),
 ('EP300',0),
 ('ERALPHAA',0),
 ('ESRRA',0),
 ('ETS1',0),
 ('Eralphaa',0),
 ('FAM48A',0),
 ('FOS',0),
 ('FOSL1',0),
 ('FOSL2',0),
 ('FOXA1',0),
 ('FOXA2',0),
 ('GABPA',0),
 ('GATA1',0),
 ('GATA2',0),
 ('GATA3',0),
 ('GTF2B',0),
 ('GTF2F1',0),
 ('GTF3C2',0),
 ('HDAC2',0),
 ('HDAC8',0),
 ('HEY1',0),
 ('HMGN3',0),
 ('HNF4A',0),
 ('HNF4G',0),
 ('HSF1',0),
 ('IRF1',0),
 ('IRF3',0),
 ('IRF4',0),
 ('JUN',0),
 ('JUNB',0),
 ('JUND',0),
 ('KAT2A',0),
 ('MAFF',0),
 ('MAFK',0),
 ('MAX',0),
 ('MEF2A',0),
 ('MEF2_complex',0),
 ('MXI1',0),
 ('MYC',0),
 ('NANOG',0),
 ('NFE2',0),
 ('NFKB1',0),
 ('NFYA',0),
 ('NFYB',0),
 ('NR2C2',0),
 ('NR3C1',0),
 ('NR4A1',0),
 ('NRF1',0),
 ('PAX5',0),
 ('PBX3',0),
 ('POU2F2',0),
 ('POU5F1',0),
 ('PPARGC1A',0),
 ('PRDM1',0),
 ('RAD21',0),
 ('RDBP',0),
 ('REST',0),
 ('RFX5',0),
 ('RXRA',0),
 ('SETDB1',0),
 ('SIN3A',0),
 ('SIRT6',0),
 ('SIX5',0),
 ('SLC22A2',0),
 ('SMARCA4',0),
 ('SMARCB1',0),
 ('SMARCC1',0),
 ('SMARCC2',0),
 ('SMC3',0),
 ('SP1',0),
 ('SP2',0),
 ('SPI1',0),
 ('SREBF1',0),
 ('SREBF2',0),
 ('SRF',0),
 ('STAT1',0),
 ('STAT2',0),
 ('STAT3',0),
 ('SUZ12',0),
 ('TAF1',0),
 ('TAF7',0),
 ('TAL1',0),
 ('TBP',0),
 ('TCF12',0),
 ('TCF7L2',0),
 ('TFAP2A',0),
 ('TFAP2C',0),
 ('THAP1',0),
 ('TRIM28',0),
 ('USF1',0),
 ('USF2',0),
 ('WRNIP1',0),
 ('XRCC4',0),
 ('YY1',0),
 ('ZBTB33',0),
 ('ZBTB7A',0),
 ('ZEB1',0),
 ('ZNF143',0),
 ('ZNF263',0),
 ('ZNF274',0),
 ('ZZZ3',0),
 ('H2AFZ',0),
 ('H3K27ac',0),
 ('H3K27me3',0),
 ('H3K36me3',0),
 ('H3K4me1',0),
 ('H3K4me2',0),
 ('H3K4me3',0),
 ('H3K79me2',0),
 ('H3K9ac',0),
 ('H3K9me1',0),
 ('H3K9me3',0),
 ('H4K20me1',0),
 ('POLR2A',0),
 ('POLR2A_elongating',0),
 ('POLR3A',0),
 ('DNase',0),
 ('FAIRE',0),
 ('cpg_island',0.02),
 ('in_cpg',0.02),
 ('avg_het',0),
 ('avg_daf',0),
 ('avg_gerp',0),
 ('gerp',0),
 ('GC',0.42),
 ('WEAK_ENH',0),
 ('ENH',0),
 ('REP',0),
 ('TSS_FLANK',0),
 ('TRAN',0),
 ('TSS',0),
 ('CTCF_REG',0),
 ('dnase_fps',0),
 ('bound_motifs',0)]

gene_GWAVA_keys = pair_GWAVA_keys

var_DeepSEA_keys = [('DeepSEAScore', 0)]
                    
feat_name = [p[0]+'_v' for p in var_CADD_keys + var_GWAVA_keys + var_DeepSEA_keys] + \
    [p[0]+'_g' for p in gene_CADD_keys + gene_GWAVA_keys] + ['dist_p'] + \
    [p[0]+'_p' for p in pair_HiCs_keys + pair_GWAVA_keys]

def normalize(X, feat_name=feat_name):
  new_feat_name = []
  new_X = []
  
  segway_mapping = pickle.load(open('../utils/Segway_mapping.pkl', 'r'))
  segway_reversed = {v: k for k, v in segway_mapping.items()}
  conseq_mapping = pickle.load(open('../utils/Consequence_mapping.pkl', 'r'))
  conseq_reversed = {v: k for k, v in conseq_mapping.items()}
  assert X.shape[1] == len(feat_name)
  for i, name in enumerate(feat_name):
    X_column = X[:, i]
    if "Segway" in name:
      choices = np.eye(len(segway_mapping))
      new_X_column = np.array([choices[int(x)] for x in X_column])
      new_X.append(new_X_column)
      new_feat_name.extend(["Segway_%s_%s" % (segway_reversed[i], name[-1]) for i in range(len(segway_reversed))])
      continue
    if "Consequence" in name:
      choices = np.eye(len(conseq_mapping))
      new_X_column = np.array([choices[int(x)] for x in X_column])
      new_X.append(new_X_column)
      new_feat_name.extend(["Consequence_%s_%s" % (conseq_reversed[i], name[-1]) for i in range(len(conseq_reversed))])
      continue
    if "GC" in name:
      new_X.append(X_column.reshape((-1, 1)))
      new_feat_name.append(name)
      continue
    if np.all(X_column == X_column.astype(int)):
      if len(np.unique(X_column)) < 30:
        unique_xs, unique_xs_counts = np.unique(X_column, return_counts=True)
        unique_xs_ratio = unique_xs_counts/float(unique_xs_counts.sum())
        mapping = {}
        value = 0
        for i, x in enumerate(unique_xs):
          if unique_xs_ratio[i] >= 0.:
            mapping[int(x)] = value
            value += 1
            
        existing_keys = sorted(mapping.keys())
            
        for i, x in enumerate(unique_xs):
          if unique_xs_ratio[i] < 0.:
            closest_x = existing_keys[np.argmin([np.abs(k-x) for k in existing_keys])]
            mapping[int(x)] = mapping[closest_x]
        
        choices = np.eye(len(set(mapping.values())))
        new_X_column = np.array([choices[mapping[int(x)]] for x in X_column])
        new_X.append(new_X_column)
        new_feat_name.extend([name[:-1] + str(int(x)) + '_' + name[-1] for x in existing_keys])
        continue
    
    new_X_column = X_column/(2 * np.std(X_column))
    if np.max(new_X_column) > 3. or np.min(new_X_column) < -3.:
      new_X_column = 3 * new_X_column / max(np.abs(np.max(new_X_column)), 
                                            np.abs(np.min(new_X_column)))
    new_X.append(new_X_column.reshape((-1, 1)))
    new_feat_name.append(name)
  new_X = np.concatenate(new_X, 1)
  return new_X, new_feat_name


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='eQTL_sampler')
  parser.add_argument(
      '-i',
      action='append',
      dest='inputs',
      default=[],
      help='inputs')
  parser.add_argument(
      '-d',
      action='append',
      dest='data',
      default=[],
      help='input_data')
  parser.add_argument(
      '-o',
      action='append',
      dest='outputs',
      default=[],
      help='output_file')
  
  args = parser.parse_args()
  infile = args.inputs[0]
  data_name = args.data[0]
  outfile = args.outputs[0]
  
  segway_mapping = pickle.load(open('../utils/Segway_mapping.pkl', 'r'))
  conseq_mapping = pickle.load(open('../utils/Consequence_mapping.pkl', 'r'))
  
  def load_dat(data_name, keys):
    dat = {}
    dat_f = np.array(pd.read_csv(data_name)[['Name'] + [v[0] for v in keys]])
    for i, key in enumerate(keys):
      dat_f[np.where(dat_f[:, i+1] != dat_f[:, i+1])[0], i+1] = key[1]
            
    if ('Segway', 'UNKNOWN') in keys:
      for ct in range(dat_f.shape[0]):
        dat_f[ct, keys.index(('Segway', 'UNKNOWN')) + 1] = \
            segway_mapping[dat_f[ct, keys.index(('Segway', 'UNKNOWN')) + 1]]
    if ('Consequence', 'INTERGENIC') in keys:
      for ct in range(dat_f.shape[0]):
        dat_f[ct, keys.index(('Consequence', 'INTERGENIC')) + 1] = \
            conseq_mapping[dat_f[ct, keys.index(('Consequence', 'INTERGENIC')) + 1]]
              
    for line in dat_f:
      dat[line[0]] = line[1:]
    return dat
  
  def load_pair_dat(data_name, keys):
    dat = {}
    dat_f = np.array(pd.read_csv(data_name)[['key1', 'key2'] + [v[0] for v in keys]])
    for i, key in enumerate(keys):
      dat_f[np.where(dat_f[:, i+2] != dat_f[:, i+2])[0], i+2] = key[1]
    for line in dat_f:
      dat[line[0] + ';' + line[1]] = line[2:]
    return dat
    
  varCADD = load_dat(data_name + '.varCADD', var_CADD_keys)
  varGWAVA = load_dat(data_name + '.varGWAVA', var_GWAVA_keys)
  varDeepSEA = load_dat(data_name + '.varDeepSEA', var_DeepSEA_keys)
  geneCADD = load_dat(data_name + '.geneCADD', gene_CADD_keys)
  geneGWAVA = load_pair_dat(data_name + '.geneGWAVA', gene_GWAVA_keys)
  geneGWAVA = {key.split(';')[0]: val for key, val in geneGWAVA.items()}
  pairHiCs = load_pair_dat(data_name + '.pairHiCs', pair_HiCs_keys)
  pairGWAVA = load_pair_dat(data_name + '.pairGWAVA', pair_GWAVA_keys)
  
  gene_to_TSS = {line[0]: str(line[1]) + '_' + str(line[2]) \
                 for line in np.array(pd.read_csv('../utils/gene_TSS_gencode_v19.csv'))}
  
  variant_feats = []
  gene_feats = []
  pair_feats = []
  with open(infile, 'r') as f:
    in_pairs = pickle.load(f)
  query_pairs = [(str(line[0]), str(line[1])) for line in in_pairs['pos'] + in_pairs['neg']]
  
  for i, pair in enumerate(query_pairs):
    try:
      v_f = np.concatenate([varCADD[pair[1]], varGWAVA[pair[1]], varDeepSEA[pair[1]]])
      g_f = np.concatenate([geneCADD[gene_to_TSS[pair[0]]], geneGWAVA[gene_to_TSS[pair[0]]]])
      p_f = np.concatenate([[int(pair[1].split('_')[1]) - int(gene_to_TSS[pair[0]].split('_')[1])],
                             pairHiCs[pair[1] + ';' + gene_to_TSS[pair[0]]], 
                             pairGWAVA[pair[1] + ';' + gene_to_TSS[pair[0]]]])
      variant_feats.append(v_f)
      gene_feats.append(g_f)
      pair_feats.append(p_f)
    except:
      print("error getting pair %d: %s" % (i, str(pair)))
  
  X = [np.concatenate([v_f, g_f, p_f]) for v_f, g_f, p_f in zip(variant_feats, gene_feats, pair_feats)]
  X = np.stack(X).astype(float)
  y = np.array([1] * len(in_pairs['pos']) + [0] * len(in_pairs['neg']))
  
  with open(outfile, 'w') as f:
    pickle.dump((X, y), f)
