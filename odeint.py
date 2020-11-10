#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 18:10:21 2020

@author: giacomo
"""

import numpy as np
import pandas as pd
from jack import SEIR, growth_fit  # , exponential
import matplotlib.pyplot as plt

# from scipy.integrate import solve_ivp

N = 1e4
perc_inf = 0.1
days = 600
beta = 0.22
tau = 20
tar = 10

y = SEIR(perc_inf, beta, tau, tar, days)

g = 1-(1/R0)


# # Interactive (jupyter)

# from ipywidgets.widgets import interact, IntSlider, FloatSlider, Layout
# style = {'description_width': '150px'}
# slider_layout = Layout(width='99%')

# interact(SEIR,
#          perc_inf=FloatSlider(min=0, max=5, step=0.01, value=0.1,
#                               description='Fraction of infected',
#                               style=style, layout=slider_layout),
#          beta=FloatSlider(min=0, max=0.5, step=0.02, value=0.2,
#                           description='Probability of infection',
#                           style=style, layout=slider_layout),
#          tau=IntSlider(min=0, max=30, step=1, value=20,
#                        description='Incubation time',
#                        style=style, layout=slider_layout),
#          tar=IntSlider(min=0, max=30, step=1, value=10,
#                        description='Recovery time',
#                        style=style, layout=slider_layout),
#          days=IntSlider(min=0, max=1200, step=100, value=900,
#                         description='Number of days',
#                         style=style, layout=slider_layout))


# a priori reproduction number
Rt = R0 * y.y[0, :]

# actual reproduction number
rt = y.y[2, 1:]/y.y[2, :-1]
# rt = Ii[tau:]/Ii[:-tau]/(tau*0.5)        # perche` divido tau per due?
# rt = np.append(np.ones(tau)*np.nan, rt)  # slittamento

# growth rate
Ki = np.gradient(np.log(y.y[2, :]))

# smoothing in funzione di tau per togliere il rumore stocastico
f = 1/np.exp(1)   # smoothing factor
D = int(f*tau)    # time interval of the measurements in cycles units

rts = pd.Series(list(rt)).rolling(window=D,
                                  min_periods=1,
                                  center=True).mean().values

Ks = pd.Series(list(Ki)).rolling(window=D,
                                 min_periods=1,
                                 center=True).mean().values

R = np.exp(Ks)  # *D)               # reproduction number from growth rate
Td = np.log(2)/Ks                  # doubling time

Is = pd.Series(list(N*y.y[2, :])).rolling(window=D,
                                          min_periods=1,
                                          center=True).mean().values

xi, pars = growth_fit(Is, 1)


figR = plt.figure(figsize=(20, 11))
plt.plot([1 for i in range(500)], 'k--')
plt.plot(Rt, 'b', label='R predicted')
plt.plot(rt, 'orange', label='R from actual increments')
plt.plot(rts, 'r', alpha=0.5, label='R moving average')
plt.plot(R, 'grey', alpha=0.5, label='R derived from K')
plt.plot(xi, np.ones(len(xi)) * np.exp(pars[1]), 'g--',
         label='R form exponential growth')

plt.legend(loc='best')
plt.xlim([0, len(rts[rts > 0]) + 2*D])
plt.ylim([0, int(R0+1)])
plt.grid()
