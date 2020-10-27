#!/usr/bin/env python
# coding: utf-8

# # SIR epidemic model on a Barabasi Albert network
#
#

# Setup:

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')

import jack

import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc

from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
# from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence

np.seterr(divide='ignore', invalid='ignore')

# Rete:

# In[16]:


# %%timeit -r 1

N = 1e3  # 4

# G = nx.barabasi_albert_graph(N,  30, seed=1234)
G = nx.erdos_renyi_graph(int(N), 0.2, seed=1234)
nit = 500

# G = jack.graph_tools(G)


# %% Alternativa a grah_tools(G) se si evitano i plots e si va dir. al modello:

k = G.degree()
G.degree_list = [d for n, d in k]
G.k_avg = np.mean(G.degree_list)


# In[19]:


# jack.graph_plots(G, [1])


# In[4]:


# jack.graph_plots(G, [2, 3])
# print(G.k_avg, G.k_min, G.sf_pars)


# Modello epidemiologico:

# In[5]:


model = ep.SIRModel(G)

beta = 0.0025  # infection rate
gamma = 0.1  # recovery rate
init = 1e-2  # initial infected population

print(model.parameters)
print(model.available_statuses)

config = mc.Configuration()
config.add_model_parameter('beta',  beta)
config.add_model_parameter('gamma', gamma)
config.add_model_parameter("percentage_infected", init)

model.set_initial_status(config)


# In[6]:


iterations = model.iteration_bunch(nit, node_status=True)
trends = model.build_trends(iterations)


# In[154]:


S = np.array([S for S, I, R in
              [list(it['node_count'].values()) for it in iterations]])
Ii = np.array([I for S, I, R in
              [list(it['node_count'].values()) for it in iterations]])

tau = int(1/gamma)
Ro = beta * tau * G.k_avg
print(Ro)

Rt = Ro * S/N

# smoothing in funzione di tau per togliere il rumore stocastico
f = np.exp(1)     # smoothing factor (number of complete infection cycles tau)
D = int(f*tau)    # time interval of the measurements in cycles units

# fix = int((D-1)/2)
# Is = pd.Series(list(Ii)).rolling(window=D).mean().iloc[D-1:].values
# Is = pd.Series(list(Ii)).rolling(window=D).mean().values[D-1:]
# Is = pd.Series(list(Ii)).rolling(window=D).mean().values[fix:-fix]
# Is = pd.Series(list(Ii)).rolling(window=D,
#                                  min_periods=1,
#                                  center=True).mean().values


# # stop when I reaches 0
# idx = Ii > 0
# idxs = Is > 0

# # reproduction number (for each interval and smoothed)
# rt = Ii[idx][1:]/Ii[idx][:-1]
# rts = Is[idxs][1:]/Is[idxs][:-1]
# K = np.diff(np.log(Is[idxs]))/D   # growth rate

# %% Stimo R come casi di oggi / casi di ieri

# reproduction number (for each interval and smoothed)
rt = Ii[1:]/Ii[:-1]
# rts = Is[1:]/Is[:-1]
rts = pd.Series(list(rt)).rolling(window=D,
                                  min_periods=1,
                                  center=True).mean().values

Ki = np.diff(np.log(Ii))/D   # growth rate
Ks = pd.Series(list(Ki)).rolling(window=D,
                                 min_periods=1,
                                 center=True).mean().values

R = np.exp(Ks * D)                 # reproduction number from growth rate
Td = np.log(2)/Ks                  # doubling time


figR = plt.figure(figsize=(20, 11))
plt.plot([1 for i in range(500)], 'k--')
plt.plot(Rt, 'blue', label='R pred')
plt.plot(rt, 'orange', label='R actual')
plt.plot(rts, 'red', label='R moving average')
plt.plot(R, 'grey', alpha=0.5, label='R from K')
plt.legend(loc='best')
plt.xlim([0, len(rts[rts > 0]) + 2*D])
plt.ylim([0, int(Ro+1)])
plt.grid()

# %% Indagini

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


# # %% ALTR_ALTERNATIVE  Stimo R come casi nuovi/casi di ieri

# Ai = (S[:-1]-S[1:])
# As = pd.Series(list(Ai)).rolling(window=D,
#                                  min_periods=1,
#                                  center=True).mean().values

# wtf = np.sqrt(G.k_avg)  # WOW THAT'S FUN

# rt = Ai/Ii[:-1] * wtf
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

# In[93]:


viz = DiffusionTrend(model, trends)
viz.plot()

# viz = DiffusionPrevalence(model, trends)
# viz.plot()


# In[147]:

'''
# **Definisco una soglia di ospedalizzazione massima.**
#
#
# All'approssimarsi della soglia faccio intervenire meccanismi di mitigazione
#   /contenimento del contagio.
#
# 1. beta si riduce (mascherine)
# 2. la connettività media si abbassa (lockdown)
#
# come intervengo sui k?
# * provo a tagliare la coda ad alti k
# * riduco in modo random (una percentuale a tutte le scale)
#
# Salvare entrambi gli scenari (pandas dataset?) e fare un confronto per
valutare la bontà delle misure (strategie di attacco)
#
# Valutare le percentuali finali di popolazione interessate dal contagio
'''
