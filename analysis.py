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

with open('pickle/simulations_lockHiBC_connected.pkl', 'rb') as f:
    lock = pickle.load(f)


# Add missing information
p = e + i
pos = p*N

Td0 = np.log(2)/K0
TTd0 = np.log(2)/KK0

TdFit = np.log(2)/KFit
TTdFit = np.log(2)/KKFit

network_models = [rando, watts, barab, holme, latti, lock]
# %%%

# Model parameters
print("\nModel parameters:")
parameters = pd.DataFrame(columns=(["N", "perc_inf", "beta",
                                    "gamma", "mu", 'R0', "1/R0"]))
parameters = parameters.append({"N": N,
                                "perc_inf": perc_inf,
                                "beta": beta,
                                "gamma": gamma,
                                "mu": mu,
                                "R0": R0,
                                "1/R0": np.round(1/R0, 2)},
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
                                "$\mu$ $(d^{-1})$",
                                "$R_0$", "$\sfrac{1}{R0}$"],
                        float_format="%.2f")

# %%%
# Derived properties
print("\nModel properties:")
properties = pd.DataFrame(columns=("Model", "K0", "Td0", "tau_s"))
properties = properties.append({"Model": "SIR",
                                "K0": KK0,
                                "Td0": TTd0,
                                "tau_s": tts0},
                               ignore_index=True)

properties = properties.append({"Model": "SEIR",
                                "K0": K0,
                                "Td0": Td0,
                                "tau_s": ts0},
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
                        header=["$K^{SIR}_0$ $(d^{-1})$",
                                "$T^{SIR}_d$ $(d)$",
                                "$K^{SEIR}_0$ $(d^{-1})$",
                                "$T^{SEIR}_d$ $(d)$"],
                        float_format="%.2f")

# %%%
# simulation results 1
print("\nSimulation results without networks:")
models = pd.DataFrame(columns=["Model", "KFit", "TdFit", "Final p",
                               "Final r", "t_final",
                               "peak", "t_peak", "s_peak"])

ppeak = np.nanmax(ii)
tt_peak = np.min(np.where(ii == ppeak))
if ii[-1] < 1/N:
    tt_final = np.nanmin(np.where(ii < 1/N))
else:
    tt_final = ddays
ffinal_r = rr[tt_final]
ffinal_p = ii[tt_final]

models = models.append({"Model": "Det. SIR",
                        "KFit": KKFit,
                        "TdFit": TTdFit,
                        "Final p": ffinal_p,
                        "Final r":  ffinal_r,
                        "t_final": tt_final,
                        "peak":  ppeak,
                        "t_peak": tt_peak,
                        "s_peak": ss[tt_peak]},
                       ignore_index=True)


peak = np.nanmax(p)
t_peak = np.min(np.where(p == peak))
if pos[-1] < 1:
    t_final = np.nanmin(np.where(pos < 1))
else:
    t_final = days
final_r = r[t_final]
final_p = p[t_final]

models = models.append({"Model": "SEIR",
                        "KFit": KFit,
                        "TdFit": TdFit,
                        "Final p": final_p,
                        "Final r": final_r,
                        "t_final": t_final,
                        "peak": ppeak,
                        "t_peak": t_peak,
                        "s_peak": s[t_peak]},
                       ignore_index=True)
print(models.round(2))
models.filename = "tex/models.tex"
if not os.path.isfile(models.filename):
    print("Saving TeX file...")
    models.to_latex(buf=models.filename,
                    index=False,
                    caption="Simulations summary.",
                    label="tab:models",
                    escape=False,
                    header=["Network model", "$K_0^{Fit}$ $(d^{-1})$",
                            "$T_d^{Fit}$ $(d)$", "$i_{end}$",
                            "$r_{end}$", "End day $(\#)$",
                            "Peak $\%$", "Peak day $(\#)$",
                            "$s_{peak}$"],
                    float_format="%.2f")


# %%%
# Networks data
print("\nNetworks data:")
networks = pd.DataFrame(columns=["Net", "E", "N", "<k>", "<C>", "<l>"])

for net in network_models:
    newline = {"Net": net.name,
               # "Size": net.G.size(),
               "E": net.G.number_of_edges(),
               "N": net.G.number_of_nodes(),
               "<k>": net.G.k_avg,
               "<C>": net.G.C_avg,
               "<l>": net.G.L_avg}
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
                              "$<k>$", "$<C>$", "$<l>$"],
                      float_format="%.2f")


# %%%
# simulation results 2
print("\nResults on networks:")
results = pd.DataFrame(columns=["Network model", "KFit", "TdFit", "Final p",
                                "Final r", "t_final",
                                "Peak", "t_Peak", "s_Peak"])


for net in network_models:
    net.Peak = np.nanmax(net.pos)
    net.t_Peak = np.min(np.where(net.pos == net.Peak))
    if net.pos[-1] == 0:
        net.t_final = np.nanmin(np.where(net.pos == 0))
    else:
        net.t_final = len(net.pos) - 1
    net.final_r = net.r[net.t_final]
    net.final_p = int(net.pos[net.t_final]/net.N)
    newline = {"Network model": net.name,
               "KFit": net.KFit50,
               "TdFit": net.TdFit50,
               "Final p": net.final_p,
               "Final r": net.final_r,
               "t_final": net.t_final,
               "Peak": net.Peak/net.N,
               "t_Peak": net.t_Peak,
               "s_Peak": net.s[net.t_Peak]}
    results = results.append(newline, ignore_index=True)

print(results.round(2))

results.filename = "tex/results.tex"
if not os.path.isfile(results.filename):
    print("Saving TeX file...")
    results.to_latex(buf=results.filename,
                     index=False,
                     caption="Simulations summary.",
                     label="tab:results",
                     escape=False,
                     header=["Network model", "$K_0^{Fit}$ $(d^{-1})$",
                             "$T_d^{Fit}$ $(d)$", "$i_{end}$",
                             "$r_{end}$", "End day $(\#)$",
                             "Peak $\%$", "Peak day $(\#)$", 
                             "$s_{peak}$"],
                     float_format="%.2f")


# %%   EXTRACTS QUANTITIES

# Boxplots for K, r_final
peaks = pd.DataFrame()
times = pd.DataFrame()
rates = pd.DataFrame()
final = pd.DataFrame()
clust = np.array([])

for net in network_models:
    real_runs = int(len(net.pm)/len(net.s))
    # if real_runs != net.runs:
    #     print(net.name + " - Warning: different number of runs than expected.")
    #     print("Real runs: " + str(real_runs))
    #     print("expected:  " + str(net.runs))
        
    # matr_p = np.array(net.pm).reshape(net.runs, (net.days+1))
    try:
        matr_p = np.array(net.pm).reshape(net.runs, len(net.s))
    except ValueError:
        print(net.name + str(net.runs) + str(len.net.s))
        continue
    
    net.peak = np.max(matr_p, axis=1)
    peaks[net.name] = np.append(net.peak,
                                np.ones(np.abs(real_runs-net.runs))*np.nan)

    net.t_peaks = [np.nanmin(np.where(matr_p[t, :] == net.peak[t])).item()
                   for t in range(real_runs)]
    net.t_finals = [np.nanmin(np.where(matr_p[t, :] < 8)).item()
                   for t in range(real_runs)]

    times[net.name] = np.append(net.t_peaks,
                                np.ones(np.abs(real_runs-net.runs))*np.nan)

    # matr_r = np.array(net.rm).reshape(real_runs, (net.days+1))
    try:
        matr_r = np.array(net.rm).reshape(net.runs, len(net.s))
    except ValueError:
        print(net.name + str(net.runs) + str(len.net.s))
        continue
    net.finals = matr_r[:, -1]
    final[net.name] = np.append(net.finals,
                                np.ones(np.abs(real_runs-net.runs))*np.nan)

    rates[net.name] = np.append(net.parsFitm0,
                                np.ones(np.abs(real_runs-net.runs))*np.nan)

    clust = np.append(clust, net.G.C_avg)


# peaks.plot.box(ylabel=r'Maximun number of positives $(individuals)$')
# times.plot.box(ylabel=r'Peak day $(d)$')
# rates.plot.box(ylabel=r'Initial growth rate $(d^{-1})$')
# final.plot.box(ylabel=r'Total affected population $(fraction)$',
#                positions=clust)
# plt.violinplot(final,  positions=clust)

# %%
# assortativity (NOT RELEVANT)
sbpx = [221, 222, 223, 224]
nets = [rando, watts, barab, holme]
# nets = latti
fig05 = plt.figure(figsize=(6.4, 4.8), dpi=300)
for i in range(4):
    net = nets[i]
    sbp = sbpx[i]
    fig05.add_subplot(sbp)
    knn = nx.k_nearest_neighbors(net.G)
    net.G.knn = [knn[i] for i in np.unique(net.G.degree_sequence)]

    x = np.unique(net.G.degree_sequence)
    y = np.array(net.G.knn)
    R = nx.degree_assortativity_coefficient(net.G)

    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('k')
    plt.ylabel(r'$\langle k_{nn} \rangle$')
    # plt.ylim([net.G.k_min, net.G.k_max])
    plt.text(net.G.k_min, np.min(y),
             'r = ' + str(np.round(R, 2)),
             color='red', alpha=0.7)

    # coef = np.polyfit(x[x<100], y[x<100], 1)
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(x, poly1d_fn(x), 'r--', alpha=0.5)
    plt.title(net.name)
    plt.tight_layout()
fig05.savefig('immagini/analysis_Assort.png')

# %%
# Peak vs peak Day

fig06 = plt.figure(figsize=(6.4, 4.8), dpi=300)
for net in network_models:
    plt.scatter(net.t_peaks, net.peak, alpha=0.5, label=net.name)
plt.legend()
plt.ylabel("Positives peak")
plt.xlabel("Peak day")
plt.tight_layout()
fig06.savefig('immagini/analysis_Peak.png')

# %%
# # Clustering vs Outbreak size
x = np.array([])
y = np.array([])
fig07 = plt.figure(figsize=(6.4, 4.8), dpi=300)
for net in [rando, watts]: #, latti]:
    xi = np.ones(len(net.t_finals))*net.G.C_avg
    # yi = np.array(net.t_finals)
    yi = np.array(net.finals)
    # yi = np.array(net.peak)
    x = np.append(x, xi)
    y = np.append(y, yi)  # final_r))
    plt.scatter(xi, yi, alpha=0.5, label=net.name)
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.plot(sorted(x), poly1d_fn(sorted(x)), '--', color="grey", alpha=0.5)

x = np.array([])
y = np.array([])
for net in [barab, holme]:  #, lock]:
    xi = np.ones(len(net.t_finals))*net.G.C_avg
    # yi = np.array(net.t_finals)
    yi = np.array(net.finals)
    # # yi = np.array(net.peak)
    x = np.append(x, xi)
    y = np.append(y, yi)  # final_r))
    plt.scatter(xi, yi, alpha=0.5, label=net.name)
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.plot(sorted(x), poly1d_fn(sorted(x)), '--', 
         color="purple", alpha=0.5)

plt.legend()
plt.ylabel("Outbreak size")
plt.xlabel("Average clustering")
plt.tight_layout()

fig07.savefig('immagini/analysis_Size.png')

# fine



# %% DBMF
print("\n\nDBMF results:")

for net in network_models:
    print("\n[ " + net.name + " ]") 
    
    net.beta_n = net.beta/net.G.k_avg
    net.G.Ka  = net.beta_n * net.G.Lma * s[0]
    net.G.Kar = net.beta_n * net.G.Lma * s[0] - net.mu
    
    print("Ksi:  " + str(np.round([net.G.Ka.item(), net.beta_n * net.G.Lm * net.s[0]], 2)))
    print("Ksir: " + str(np.round([net.G.Kar.item(), net.beta_n * net.G.Lm * net.s[0] - net.mu], 2))) 
    
    A1a = np.array([[-net.gamma, net.beta_n * net.G.Lma.item() * net.s[0]], [net.gamma, -net.mu]])
    eigval1a, eigvec1a = np.linalg.eig(A1a)
    net.K1a = eigval1a[0]
    
    A1 = np.array([[-net.gamma, net.beta_n * net.G.Lm * net.s[0]], [net.gamma, -net.mu]])
    eigval1, eigvec1 = np.linalg.eig(A1)
    net.K1 = eigval1[0]
    
    # Growt rates from Lma (uncorrelated approx with k moments), Lm (actual bigger eigenvalue of the corrected connectivity matrix) and from the actual Fit:
    print("Kseir: " + str(np.round([net.K1a, net.K1, net.KFit], 2)))
   
