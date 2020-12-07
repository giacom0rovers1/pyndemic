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
days = 100
beta = 0.73         # infection probability
tau_i = 3          # incubation time
tau_r = 3          # recovery time
R0 = beta * tau_r   # basic reproduction number


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

# Add missing information
p = e + i
pos = p*N

Td0 = np.log(2)/K0
TTd0 = np.log(2)/KK0

TdFit = np.log(2)/KFit
TTdFit = np.log(2)/KKFit


# %%%

# Model parameters
parameters = pd.DataFrame(columns=(["N", "perc_inf", "beta",
                                    "gamma", "mu"]))
parameters = parameters.append({"N": N,
                                "perc_inf": perc_inf,
                                "beta": beta,
                                "gamma": gamma,
                                "mu": mu},
                               ignore_index=True)
print(parameters.round(2))

parameters.filename = "tex/params.tex"
if not os.path.isfile(parameters.filename):
    print("Saving TeX file...")
    parameters.to_latex(buf=parameters.filename,
                        index=False,
                        caption="Model parameters.",
                        label="tab:params",
                        escape=False,
                        header=["Total population",
                                "$i_{start}$ $(\%)$",
                                "$\beta $ $(d^{-1})$",
                                "$\gamma $ $(d^{-1})$",
                                "$\mu$ $(d^{-1})$"],
                        float_format="%.2f")

# %%%
# Derived properties
properties = pd.DataFrame(columns=("R0", "KK0", "TTd0", "K0", "Td0"))
properties = properties.append({"R0": R0,
                                "KK0": KK0,
                                "TTd0": TTd0,
                                "K0": K0,
                                "Td0": Td0},
                               ignore_index=True)
print(properties.round(2))

properties.filename = "tex/props.tex"
if not os.path.isfile(properties.filename):
    print("Saving TeX file...")
    properties.to_latex(buf=properties.filename,
                        index=False,
                        caption="Model properties.",
                        label="tab:props",
                        escape=False,
                        header=["$R_0$",
                                "$K^{SIR}_0$ $(d^{-1})$",
                                "$T^{SIR}_d$ $(d)$",
                                "$K^{SEIR}_0$ $(d^{-1})$",
                                "$T^{SEIR}_d$ $(d)$"],
                        float_format="%.2f")

# %%%
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

print(networks.round(2))

networks.filename = "tex/networks.tex"
if not os.path.isfile(networks.filename):
    print("Saving TeX file...")
    networks.to_latex(buf=networks.filename,
                      index=False,
                      caption="Networks properties.",
                      label="tab:networks",
                      escape=False,
                      header=["Network", "Edges", "Nodes",
                              "$<k>$", "$<C>$"],
                      float_format="%.2f")


# %%%
# simulation results

results = pd.DataFrame(columns=["Model", "KFit", "TdFit", "Final p",
                                "Final r", "peak", "t_peak", "t_final"])

ppeak = np.nanmax(ii)
tt_peak = np.min(np.where(ii == ppeak))
if ii[-1] == 0:
    tt_final = np.nanmin(np.where(ii == 0))
else:
    tt_final = ddays
ffinal_r = rr[tt_final]
ffinal_p = ii[tt_final]

results = results.append({"Model": "Det. SIR",
                          "KFit": KKFit,
                          "TdFit": TTdFit,
                          "Final p": ffinal_p,
                          "Final r":  ffinal_r,
                          "peak":  ppeak,
                          "t_peak": tt_peak,
                          "t_final": tt_final},
                         ignore_index=True)


peak = np.nanmax(p)
t_peak = np.min(np.where(p == peak))
if pos[-1] == 0:
    t_final = np.nanmin(np.where(pos == 0))
else:
    t_final = days
final_r = r[t_final]
final_p = p[t_final]

results = results.append({"Model": "Det. SEIR",
                          "KFit": KFit,
                          "TdFit": TdFit,
                          "Final p": final_p,
                          "Final r": final_r,
                          "peak": ppeak,
                          "t_peak": t_peak,
                          "t_final": t_final},
                         ignore_index=True)


for net in [rando, latti, watts, barab, holme]:
    peak = np.nanmax(net.pos)
    t_peak = np.min(np.where(net.pos == peak))
    if net.pos[-1] == 0:
        t_final = np.nanmin(np.where(net.pos == 0))
    else:
        t_final = len(net.pos) - 1
    final_r = net.r[t_final]
    final_p = int(net.pos[t_final]/net.N)
    newline = {"Model": net.name,
               "KFit": net.KFit50,
               "TdFit": net.TdFit50,
               "Final p": final_p,
               "Final r": final_r,
               "peak": peak/net.N,
               "t_peak": t_peak,
               "t_final": t_final}
    results = results.append(newline, ignore_index=True)

print(results.round(3))

results.filename = "tex/results.tex"
if not os.path.isfile(results.filename):
    print("Saving TeX file...")
    results.to_latex(buf=results.filename,
                     index=False,
                     caption="Simulations summary.",
                     label="tab:results",
                     escape=False,
                     header=["Model", "$K_0^{Fit}$ $(d^{-1})$",
                             "$T_d^{Fit}$ $(d)$", "$i_{end}$",
                             "$r_{end}$", "End day $(\#)$",
                             "Peak $\%$", "Peak day $(\#)$"],
                     float_format="%.2f")

# %%

# Boxplots for K, r_final
peaks = pd.DataFrame()
times = pd.DataFrame()
rates = pd.DataFrame()
final = pd.DataFrame()

for net in [rando, latti, watts, barab, holme]:
    real_runs = int(len(net.pm)/(net.days+1))
    if real_runs != net.runs:
        print("Warning: different number of runs than expected.")
    matr_p = np.array(net.pm).reshape(real_runs, (net.days+1))

    net.peak = np.max(matr_p, axis=1)
    peaks[net.name] = np.append(net.peak,
                                np.ones(np.abs(real_runs-net.runs))*np.nan)

    net.t_peaks = [np.nanmin(np.where(matr_p[t, :] == net.peak[t])).item()
                   for t in range(real_runs)]
    times[net.name] = np.append(net.t_peaks,
                                np.ones(np.abs(real_runs-net.runs))*np.nan)

    matr_r = np.array(net.rm).reshape(real_runs, (net.days+1))
    net.finals = matr_r[:, -1]
    final[net.name] = np.append(net.finals,
                                np.ones(np.abs(real_runs-net.runs))*np.nan)

    rates[net.name] = np.append(net.parsFitm0,
                                np.ones(np.abs(real_runs-net.runs))*np.nan)


peaks.plot.box(ylabel=r'Maximun number of positives $(individuals)$')
times.plot.box(ylabel=r'Peak day $(d)$')
rates.plot.box(ylabel=r'Initial growth rate $(d^{-1})$')
final.plot.box(ylabel=r'Total affected population $(fraction)$')


# %%
# assortativity (NOT RELEVANT)
sbpx = [221, 222, 223, 224]
nets = [rando, watts, barab, holme]
# nets = latti
fig05 = plt.figure(dpi=300)
for i in range(4):
    net = nets[i]
    sbp = sbpx[i]
    fig05.add_subplot(sbp)
    knn = nx.k_nearest_neighbors(net.G)
    net.G.knn = [knn[i] for i in np.unique(net.G.degree_sequence)]

    x = np.unique(net.G.degree_sequence)
    y = net.G.knn
    r = nx.degree_pearson_correlation_coefficient(net.G)

    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('k')
    plt.ylabel(r'$\langle k_{nn} \rangle$')
    # plt.ylim([net.G.k_min, net.G.k_max])
    plt.text(net.G.k_min, np.min(y),
             'r = ' + str(np.round(r, 2)),
             color='red', alpha=0.7)

    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(x, poly1d_fn(x), 'r--', alpha=0.5)
    plt.title(net.name)
    plt.tight_layout()


# %%
# Peak vs peak Day

fig06 = plt.figure(dpi=300)
for net in [rando, watts, barab, holme, latti]:
    plt.scatter(net.t_peaks, net.peak, alpha=0.5, label=net.name)
plt.legend()
plt.ylabel("Positives peak")
plt.xlabel("Peak day")
plt.tight_layout()

# fine
