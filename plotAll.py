#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:14:44 2020

@author: Giacomo Roversi
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


N = 1e4
n = N/100
perc_inf = 0.1
days = 120
beta = 0.061          # infection probability
avgk = 12              # average contacts
lmbda = beta * avgk    # infection rate
tau_i = 3             # incubation time
tau_r = 3             # recovery time
R0 = lmbda * tau_r     # basic reproduction number


# Getting back the objects:
with open('pickle/all_simulations.pkl', 'rb') as f:
    watts, rando, latti, barab, holme = pickle.load(f)

# with open('pickle/smallw_long.pkl', 'rb') as f:
#     watts_long = pickle.load(f)

with open('pickle/SEIR.pkl', 'rb') as f:
    s, e, i, r, t, days, daysl, KFit, tsFit, parsFit, \
        mu, gamma, R0, K0, ts0, pars0, \
        fig02, fig03, fig04 = pickle.load(f)

with open('pickle/SIR.pkl', 'rb') as f:
    ss, ii, rr, tt, ddays, KKFit, ttsFit, pparsFit, \
        mu, R0, KK0, tts0, ppars0, \
        ffig02, ffig03, ffig04 = pickle.load(f)

with open('pickle/simulations_lockHiBC_connected.pkl', 'rb') as f:
    lock = pickle.load(f)

with open('pickle/simulations_HK_hiTau.pkl', 'rb') as f:
    nawar = pickle.load(f)

network_models = [rando, watts, barab, holme, latti, lock, nawar]
for net in network_models:
    net.plot()
    net.save()
