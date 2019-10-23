#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 22:29:22 2019

@author: zqwu
"""

import os
import numpy as np

input_raw = './DeepSEA/infile.vcf.out.funsig'
output = './output.varDeepSEA'

f1 = open(input_raw, 'r')
f2 = open(output, 'w')

f2.write('Name,DeepSEAScore\n')
dat = {}
for i, line in enumerate(f1):
  if i==0:
    continue
  line = line.strip().split(',')
  dat[int(line[0])] = (line[1][3:] + '_' + line[2] + '_' + line[4] + '_' + line[5] + '_b37', float(line[-1]))

for k in sorted(dat.keys()):
  f2.write(dat[k][0] + ',' + str(dat[k][1]) + '\n')

f1.close()
f2.close()