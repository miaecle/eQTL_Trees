#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:09:14 2018

@author: zqwu
"""
import numpy as np

feature_order = np.array(['Consequence',
 'Consequence',
 'Consequence',
 'Consequence',
 'Consequence',
 'Consequence',
 'Consequence',
 'Consequence',
 'Consequence',
 'Consequence',
 'Consequence',
 'Consequence',
 'Consequence',
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
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
 'Segway',
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
 'repeat',
 'GC_g',
 'CpG_g',
 'cHmmTssA_g',
 'cHmmTssAFlnk_g',
 'cHmmTxFlnk_g',
 'cHmmTx_g',
 'cHmmTxWk_g',
 'cHmmEnhG_g',
 'cHmmEnh_g',
 'cHmmZnfRpts_g',
 'cHmmHet_g',
 'cHmmTssBiv_g',
 'cHmmBivFlnk_g',
 'cHmmEnhBiv_g',
 'cHmmReprPC_g',
 'cHmmReprPCWk_g',
 'cHmmQuies_g',
 'EncExp_g',
 'EncH3K27Ac_g',
 'EncH3K4Me1_g',
 'EncH3K4Me3_g',
 'EncNucleo_g',
 'EncOCC_g',
 'EncOCCombPVal_g',
 'EncOCDNasePVal_g',
 'EncOCFairePVal_g',
 'EncOCpolIIPVal_g',
 'EncOCctcfPVal_g',
 'EncOCmycPVal_g',
 'EncOCDNaseSig_g',
 'EncOCFaireSig_g',
 'EncOCpolIISig_g',
 'EncOCctcfSig_g',
 'EncOCmycSig_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'Segway_g',
 'tOverlapMotifs_g',
 'TFBS_g',
 'TFBSPeaks_g',
 'TFBSPeaksMax_g',
 'CTCF_g',
 'DNase_g',
 'FAIRE_g',
 'H2AFZ_g',
 'H3K27ac_g',
 'H3K27me3_g',
 'H3K36me3_g',
 'H3K4me1_g',
 'H3K4me2_g',
 'H3K4me3_g',
 'H3K79me2_g',
 'H3K9ac_g',
 'H3K9me1_g',
 'H3K9me3_g',
 'H4K20me1_g',
 'MYC_g',
 'POLR2A_g',
 'POLR2A_elongating',
 'POLR3A_g',
 'pwm_g',
 'cpg_island_g',
 'avg_gerp_g',
 'dnase_fps_g',
 'in_cpg_g',
 'repeat_g'])