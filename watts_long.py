#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:20:43 2020

@author: Giacomo Roversi
"""
import time
import pickle
import networkx as nx

from pyndemic import SEIR_odet, SIR_odet, contagion_metrics, pRandNeTmic
Tic = time.perf_counter()

N = 1e4
perc_inf = 0.1
days = 1600
beta = 0.22         # infection probability
tau_i = 20          # incubation time
tau_r = 10          # recovery time
R0 = beta * tau_r   # basic reproduction number


# DETERMINISTIC well-mixed approach

# ==========
# SEIR MODEL
# ==========
print("\nSEIR deterministic model:")
s, e, i, r, t, fig02 = SEIR_odet(perc_inf, beta, tau_i, tau_r, days,
                                 "SEIR deterministic model")

K, Ki, ts, pars, Rt, Rti, Rts, Td0, Tdi, Tds, fig03, fig04 = \
    contagion_metrics(s, e, i, r, t, 0, 0, [0, 0], R0, tau_i, tau_r, N,
                      "SEIR deterministic model")

# ===================
# SMALL-WORLD NETWORK
# ===================
print("\nSEIR over small-world network:")
watts_long = pRandNeTmic('Watts-Strogatz',
                         'smallw_long',
                         nx.connected_watts_strogatz_graph(int(N), 12,
                                                           0.1, seed=1234),
                         nx.connected_watts_strogatz_graph(int(N/100), 12,
                                                           0.1, seed=1234))
watts_long.run(perc_inf, beta, tau_i, tau_r, days, t)

# with open('pickle/smallw_long.pkl', 'rb') as f:
#     watts_long = pickle.load(f)
watts_long.plot(beta, tau_i, tau_r, days, t, K, ts, pars)
watts_long.save()
