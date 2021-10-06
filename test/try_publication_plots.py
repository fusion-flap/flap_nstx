#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 01:18:46 2020

@author: mlampert
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

styled=True
if styled:
    plt.rc('font', family='serif', serif='Helvetica')
    labelsize=9
    linewidth=1
    major_ticksize=2
    plt.rc('text', usetex=False)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['lines.linewidth'] = linewidth
    plt.rcParams['axes.linewidth'] = linewidth
    plt.rcParams['axes.labelsize'] = labelsize
    plt.rcParams['axes.titlesize'] = labelsize
    
    plt.rcParams['xtick.labelsize'] = labelsize
    plt.rcParams['xtick.major.size'] = major_ticksize
    plt.rcParams['xtick.major.width'] = linewidth
    plt.rcParams['xtick.minor.width'] = linewidth/2
    plt.rcParams['xtick.minor.size'] = major_ticksize/2
    
    plt.rcParams['ytick.labelsize'] = labelsize
    plt.rcParams['ytick.major.width'] = linewidth
    plt.rcParams['ytick.major.size'] = major_ticksize
    plt.rcParams['ytick.minor.width'] = linewidth/2
    plt.rcParams['ytick.minor.size'] = major_ticksize/2
    plt.rcParams['legend.fontsize'] = labelsize
else:
    import matplotlib.style as pltstyle
    pltstyle.use('default')

pdf=PdfPages('plot.pdf')
# width as measured in inkscape
width = 3.487
height = width / 1.618

fig, ax = plt.subplots(figsize=(width,height))

x = np.arange(0.0, 3*np.pi , 0.1)
plt.plot(x, 1*np.sin(x), label='sin')

ax.set_ylabel('Some Metric (in unit)')
ax.set_xlabel('Something (in unit)',)
x1,x2=ax.get_xlim()
y1,y2=ax.get_ylim()
ax.set_aspect((x2-x1)/(y2-y1)/1.618)
ax.set_title("title")

fig.tight_layout(pad=0., w_pad=0., h_pad=0.)

pdf.savefig()
pdf.close()