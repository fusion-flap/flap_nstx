#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 19:07:20 2020

@author: mlampert
"""

import csv
import numpy as np

db_ahmed=[]
with open('/Users/mlampert/work/NSTX_workspace/WORK_MATE/Profile_fitsfur_Mate') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        line=[]
        for data in row:
            try:
                line.append(float(data.strip(' ')))
            except:
                pass
        db_ahmed.append(line)
    
db_ahmed=np.asarray(db_ahmed).transpose()
db_dict={'shot':db_ahmed[0,:],
         'time1':db_ahmed[1,:],
         'time2':db_ahmed[2,:],
         'T_e_ped':db_ahmed[3,:],
         'n_e_ped':db_ahmed[4,:],
         'p_e_ped':db_ahmed[5,:],
         'T_e_max_grad':db_ahmed[6,:],
         'n_e_max_grad':db_ahmed[7,:],
         'p_e_max_grad':db_ahmed[8,:],
         'T_e_width':db_ahmed[9,:],
         'n_e_width':db_ahmed[10,:],
         'p_e_width':db_ahmed[11,:],
         }
print(np.mean(db_dict['T_e_max_grad']),
      np.mean(db_dict['n_e_max_grad']))