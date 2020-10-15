#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:33:00 2020

@author: arianna
"""

# USARE LATEX CON SIUNITX PER I GRAFICI:

import numpy as np
import matplotlib.pyplot as plt

params = {'text.usetex': True,
          'font.family': 'serif',
          'text.latex.preamble': r'\usepackage{siunitx,amsmath}'}
plt.rcParams.update(params)
# plt.rcParams.get  # per controllare l'aggiornamento di rcParams

# figura di prova:
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ylabel('GNSS Altitude (\si{\meter})')


# PLOT COME OGGETTO AX

x1 = np.array(range(10))
y1 = np.array(range(10))
y1 += 5

# https://realpython.com/python-matplotlib-guide/


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x1, y1, s=0.1)
ax.text(13, 1100, r'$\Gamma$', fontsize=15)
ax.text(0.5, 0.5, r'an equation: $E=mc^2$', fontsize=15)

ax.scatter(x1, y1, s=0.1)
ax.text(0.9, 0.9, r'$\Gamma$', fontsize=15, transform=ax.transAxes)
# (plt.text?)
