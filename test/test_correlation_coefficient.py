#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:29:13 2022

@author: mlampert
"""
import numpy as np
import matplotlib.pyplot as plt

n_data=156
n_rand=10000
result=np.zeros([n_rand])

for i in range(n_rand):
    a=np.random.rand(n_data)
    b=np.random.rand(n_data)
    result[i]=np.abs(np.sum((a-np.mean(a))*(b-np.mean(b)))/np.sqrt((np.sum((a-np.mean(a))**2)*(np.sum((b-np.mean(b))**2)))))

    
print(np.mean(result),np.var(result))