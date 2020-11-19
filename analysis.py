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

with open('pickle/SIR.pkl', 'rb') as f:
    ss, ii, rr, tt, KK, tts, ppars, ffig02, ffig03, ffig04 = pickle.load(f)

# Networks data
networks = pd.DataFrame(columns=["Net", "E", "N", "<k>", "<C>"])

for net in [rando, latti, watts, barab, holme]:
    newline = {"Net": net.name,
               # "Size": net.G.size(),
               "E": net.G.number_of_edges(),
               "N": net.G.number_of_nodes(),
               "<k>": net.G.k_avg,
               "<C>": net.G.C_avg}
    networks = networks.append(newline, ignore_index=True)

print(networks.round(3))
# networks.to_latex(buf="tex/networks.tex",
#                   index=False,
#                   caption="Networks properties.",
#                   label="tab:networks",
#                   escape=False,
#                   header=["Network", "Edges", "Nodes",
#                           "$<k>$", "$<C>$"],
#                   float_format="%.2f")


# outputs data
pos = e + i

results = pd.DataFrame(columns=["Model", "K0", "ts", "Final_i",
                                "Final_r", "peak", "t_peak"])

results = results.append({"Model": "Det. SIR",
                          "K0": KK,
                          "ts": tts,
                          "Final_i": ii[250],
                          "Final_r":  rr[250],
                          "peak":  np.nanmax(ii)*100,
                          "t_peak": np.min(np.where(ii == np.nanmax(ii)))},
                         ignore_index=True)

results = results.append({"Model": "Det. SEIR",
                          "K0": K,
                          "ts": ts,
                          "Final_i": i[days],
                          "Final_r": r[days],
                          "peak": np.nanmax(pos)*100,
                          "t_peak": np.min(np.where(pos == np.nanmax(pos)))},
                         ignore_index=True)


for net in [rando, latti, watts_long, barab, holme]:
    net.pos = net.i + net.e
    newline = {"Model": net.name,
               "K0": net.K0,
               "ts": net.ts,
               "Final_i": net.i[days],
               "Final_r": net.r[days],
               "peak": np.nanmax(net.pos)*100,
               "t_peak": np.min(np.where(net.pos == np.nanmax(net.pos)))}
    results = results.append(newline, ignore_index=True)

print(results.round(2))
# results.to_latex(buf="tex/results.tex",
#                  index=False,
#                  caption="Simulations summary.",
#                  label="tab:results",
#                  escape=False,
#                  header=["Model", "$K_{0}$", "$\tau_{s}$", "$i_{final}$",
#                          "$r_{final}$", "Peak $\%$", "on day"],
#                  float_format="%.2f")


# fine
