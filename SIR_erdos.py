#!/usr/bin/env python
# coding: utf-8

# ======================================================
#    SIR epidemic model on a Barabasi Albert network
# ======================================================

# Setup:
# get_ipython().run_line_magic('matplotlib', 'inline')

import jack as jk

import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc

from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
# from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence

np.seterr(divide='ignore', invalid='ignore')

# %% NETWORK
N = 1e4
# G = nx.barabasi_albert_graph(N,  30, seed=1234)
G = nx.erdos_renyi_graph(int(N), 0.005, seed=1234)
nit = 500

# G = jk.graph_tools(G)

# %% Alternativa a grah_tools(G) se si evitano i plots e si va dir. al modello:
k = G.degree()
G.degree_list = [d for n, d in k]
G.k_avg = np.mean(G.degree_list)

# %% GRAPH PLOTS

# jk.graph_plots(G, [1])
# jk.graph_plots(G, [2, 3])
# print(G.k_avg, G.k_min, G.sf_pars)


# %% EPIDEMIC MODEL

# Config:
model = ep.SIRModel(G)

beta = 0.0025  # infection rate
gamma = 0.05  # 0.1  # recovery rate
init = 1e-2  # initial infected population

print(model.parameters)
print(model.available_statuses)

config = mc.Configuration()
config.add_model_parameter('beta',  beta)
config.add_model_parameter('gamma', gamma)
config.add_model_parameter("percentage_infected", init)

model.set_initial_status(config)

# Run:
iterations = model.iteration_bunch(nit, node_status=True)
trends = model.build_trends(iterations)


# Recover status variables:
S = np.array([S for S, I, R in
              [list(it['node_count'].values()) for it in iterations]])
Ii = np.array([I for S, I, R in
              [list(it['node_count'].values()) for it in iterations]])


# Plot:
viz = DiffusionTrend(model, trends)
viz.plot()

# viz = DiffusionPrevalence(model, trends)
# viz.plot()


# %% ANALYSES

# Epidemic indexes computation:
tau = 1/gamma
Ro = beta * tau * G.k_avg
print(Ro)

# a priori reproduction number
Rt = Ro * S/N

# actual reproduction number
rt = Ii[1:]/Ii[:-1]
# rt = Ii[tau:]/Ii[:-tau]/(tau*0.5)        # perche` divido tau per due?
# rt = np.append(np.ones(tau)*np.nan, rt)  # slittamento

# growth rate
Ki = np.diff(np.log(Ii))  # /D

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


Is = pd.Series(list(Ii)).rolling(window=D,
                                 min_periods=1,
                                 center=True).mean().values
xi, pars = jk.growth_fit(Is, 1)


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
plt.ylim([0, int(Ro+1)])
plt.grid()

# %% Indagini extra

# print(pd.DataFrame({"K": K,
#                     "R": R,
#                     "Td": Td,
#                     "rts": rts}))

# print(pd.DataFrame({"S": S[idx],
#                     "Ii": Ii[idx],
#                     "Rt": Rt[idx]}))

# print(pd.DataFrame({"Is": Ii[idx]}))


# # %%  ALTERNATIVE

# Ai = (S[:-D]-S[D:])/D
# As = pd.Series(list(Ai)).rolling(window=D,
#                                  min_periods=1,
#                                  center=True).mean().values

# rt = Ai[1:]/Ai[:-1]
# rts = As[1:]/As[:-1]
# K = np.diff(np.log(As))/D   # growth rate

# R = np.exp(K * D)                 # reproduction number from growth rate
# Td = np.log(2)/K                  # doubling time


# %% ALTR_ALTERNATIVE  Stimo R come casi nuovi/casi di ieri

# rt = (S[:-tau]-S[tau:])/Ii[:-tau]/tau*2
# rt = np.append(np.ones(tau)*np.nan, rt)

# rts = pd.Series(list(rt)).rolling(window=D,
#                                   min_periods=1,
#                                   center=True).mean().values


# # plot
# figR = plt.figure(figsize=(20, 11))
# plt.plot([1 for i in range(500)], 'k--')
# plt.plot(Rt, 'blue', label='R pred')
# plt.plot(rt, 'orange', label='R actual')
# plt.plot(rts, 'red', label='R moving average')
# plt.legend(loc='best')
# plt.xlim([0, len(rts[rts > 0]) + 2*D])
# plt.ylim([0, int(Ro+5)])
# plt.grid()
# plt.title('SPERIMENTALE')
