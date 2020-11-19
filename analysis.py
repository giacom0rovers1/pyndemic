#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:14:44 2020

@author: Giacomo Roversi
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 1e4
perc_inf = 0.1
days = 600
beta = 0.22         # infection probability
tau_i = 20          # incubation time
tau_r = 10          # recovery time
R0 = beta * tau_r   # basic reproduction number

# Getting back the objects:
with open('pickle/all_networks.pkl', 'rb') as f:
    watts, rando, latti, barab, holme = pickle.load(f)

with open('pickle/smallw_long.pkl', 'rb') as f:
    watts_long = pickle.load(f)

with open('pickle/SEIR.pkl', 'rb') as f:
    s, e, i, r, t, K, ts, pars, fig02, fig03, fig04 = pickle.load(f)


# Networks data
networks = pd.DataFrame(columns=["Network", "Size", "E", "N", "<k>", "<C>"])

for net in [rando, latti, watts, barab, holme]:
    newline = {"Network": net.name,
               "Size": net.G.size(),
               "E": net.G.number_of_edges(),
               "N": net.G.number_of_nodes(),
               "<k>": net.G.k_avg,
               "<C>": net.G.C_avg}
    networks = networks.append(newline, ignore_index=True)

print(networks.round(2))


# outputs data
pos = e + i
results = pd.DataFrame({"Model": "Deterministic",
                        "K0": K,
                        "ts": ts,
                        "Final_i": i[days],
                        "Final_r": r[days],
                        "peak": np.nanmax(pos),
                        "t_peak": np.min(np.where(pos == np.nanmax(pos)))})

for net in [rando, latti, watts_long, barab, holme]:
    net.pos = net.i + net.e
    newline = {"Model": net.name,
               "K0": net.K0,
               "ts": net.ts,
               "Final_i": net.i[days],
               "Final_r": net.r[days],
               "peak": np.nanmax(net.pos),
               "t_peak": np.min(np.where(net.pos == np.nanmax(net.pos)))}
    results = results.append(newline, ignore_index=True)

print(results.round(2))
