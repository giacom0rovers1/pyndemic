#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:14:44 2020

@author: Giacomo Roversi
"""
import pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

N = 1e4
perc_inf = 0.1
days = 600
beta = 0.22         # infection probability
tau_i = 20          # incubation time
tau_r = 10          # recovery time

# Getting back the objects:
with open('pickle/all_networks.pkl', 'rb') as f:
    watts, rando, latti, barab, holme = pickle.load(f)

with open('pickle/smallw_long.pkl', 'rb') as f:
    watts_long = pickle.load(f)

with open('pickle/SEIR.pkl', 'rb') as f:
    s, e, i, r, t, K, ts, pars, \
        mu, gamma, R0, K0, ts0, pars0, \
        fig02, fig03, fig04 = pickle.load(f)

with open('pickle/SIR.pkl', 'rb') as f:
    ss, ii, rr, tt, KK, tts, ppars, \
        mu, RR0, KK0, tts0, ppars0, \
        ffig02, ffig03, ffig04 = pickle.load(f)

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
# networks.to_latex(  # buf="tex/networks.tex",
#                   index=False,
#                   caption="Networks properties.",
#                   label="tab:networks",
#                   escape=False,
#                   header=["Network", "Edges", "Nodes",
#                           "$<k>$", "$<C>$"],
#                   float_format="%.2f")


# outputs data
p = e + i
Td0 = np.log(2)/K0
TTd0 = np.log(2)/KK0

results = pd.DataFrame(columns=["Model", "K0", "Td0", "Final_i",
                                "Final_r", "peak", "t_peak", "t_final"])

results = results.append({"Model": "Det. SIR",
                          "K0": KK,
                          "Td0": TTd0,
                          "Final_i": ii[250]*100,
                          "Final_r":  rr[250]*100,
                          "peak":  np.nanmax(ii)*100,
                          "t_peak": np.min(np.where(ii == np.nanmax(ii))),
                          "t_final": np.min(np.where(ii < 1/N))},
                         ignore_index=True)

results = results.append({"Model": "Det. SEIR",
                          "K0": K,
                          "Td0": Td0,
                          "Final_i": i[days]*100,
                          "Final_r": r[days]*100,
                          "peak": np.nanmax(p)*100,
                          "t_peak": np.min(np.where(p == np.nanmax(p))),
                          "t_final": np.min(np.where(p < 1/N))},
                         ignore_index=True)


for net in [rando, latti, watts_long, barab, holme]:
    net.p = net.i + net.e
    newline = {"Model": net.name,
               "K0": net.K0,
               "Td0": net.Td0,
               "Final_i": net.i[days]*100,
               "Final_r": net.r[days]*100,
               "peak": np.nanmax(net.p)*100,
               "t_peak": np.min(np.where(net.p == np.nanmax(net.p))),
               "t_final": np.min(np.where(net.p == 0))}
    results = results.append(newline, ignore_index=True)

print(results.round(2))
# results.to_latex(buf="tex/results.tex",
#                  index=False,
#                  caption="Simulations summary.",
#                  label="tab:results",
#                  escape=False,
#                  header=["Model", "$K_0$", "$T_d$", "$i_{final}$",
#                          "$r_{final}$", "Peak $\%$", "peak day", "end day"],
#                  float_format="%.2f")


presentation = pd.DataFrame(columns=(["N", "perc_inf", "beta",
                                      "gamma", "mu", "R0",
                                      "KK0", "TTd0", "K0", "Td0"]))
presentation = presentation.append({"N": N,
                                    "perc_inf": perc_inf,
                                    "beta": beta,
                                    "gamma": gamma,
                                    "mu": mu,
                                    "R0": R0,
                                    "KK0": KK0,
                                    "TTd0": TTd0,
                                    "K0": K0,
                                    "Td0": Td0},
                                   ignore_index=True)
print(presentation.round(3))
# presentation.to_latex(  # buf="tex/params.tex",
#                       index=False,
#                       caption="Model parameters.",
#                       label="tab:params",
#                       escape=False,
#                       header=["Total population", "$i_{start}$ $\%$",
#                               "$beta$",
#                               "$gamma$", "$mu$", "$R_0$",
#                               "$K^{SIR}_0$", "$T^{SIR}_d$",
#                               "$K^{SEIR}_0$", "$T^{SEIR}_d$"],
#                       float_format="%.2f")

# fine
