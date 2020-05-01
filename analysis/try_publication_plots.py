#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 01:18:46 2020

@author: mlampert
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
#from matplotlib.backends.backend_pdf import PdfPages

#props = fm.FontProperties(fname='/System/Library/Fonts/Helvetica.ttc')
#plt.rc('font', family='serif', serif='Helvetica')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
#pdf=PdfPages('plot.pdf')
# width as measured in inkscape
width = 3.487
height = width / 1.618

fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

x = np.arange(0.0, 3*np.pi , 0.1)
plt.plot(x, np.sin(x))

ax.set_ylabel('Some Metric (in unit)')
ax.set_xlabel('Something (in unit)')
ax.set_xlim(0, 3*np.pi)

fig.set_size_inches(width, height)
#pdf.savefig()
#pdf.close()