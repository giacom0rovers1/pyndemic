#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 19:26:21 2020

@author: giacomo
"""
# import ndlib as nd

# to UPDATE NDlib, run:
# $ pip install git+git://github.com/GiulioRossetti/ndlib.git
# inside:
# /home/giacomo/anaconda3/lib/python3.8/site-packages/

# import jack
import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc

from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence

N = 1000
p = 0.075  # 0.1

G = nx.erdos_renyi_graph(N, p, seed=1234)
nit = 500

# G = nx.connected_watts_strogatz_graph(N, 100, 0.1, seed=1234)
# nit = 400

# G = nx.connected_watts_strogatz_graph(N, 60, 0.1, seed=1234)
# nit = 400

''' Inserire le due reti di riferimento equivalenti a ws con
wsr = nx.random_reference(ws)
wsl = nx.lattice_reference(ws)
# ws.sigma = nx.sigma(ws)  # heavy  # small world if >1
# ws.omega = nx.omega(ws)  # heavy
'''

'''Interessante:
    - con 1000,10,0.1 il comportamento e` simile a quello di un grafo random
    - con 1000, 5, 0.1 invece c'e` una sorta di immunita` di gregge
        (nei nodi con alta centralita` presumo)
riflettere su come simulare il lockdown e sui modelli nulli'''


model = ep.SEIRModel(G)


# model = ep.SEIRModel(G)

# continuous time dynamics (sembra molto piu` veloce, 5x)
# model = ep.SEIRctModel(g)

alpha = 0.05  # 0.01  # latent period (# iterations = 1/alpha: 5% = 20 it.)
beta = 0.0025  # infection rate
gamma = 0.1  # recovery rate
init = 5e-3  # initial infected population


print(model.parameters)
print(model.available_statuses)


config = mc.Configuration()
config.add_model_parameter('alpha', alpha)
config.add_model_parameter('beta',  beta)
config.add_model_parameter('gamma', gamma)
# config.add_model_parameter('lambda', 0.01)  # recovery rate
config.add_model_parameter("percentage_infected", init)

model.set_initial_status(config)

iterations = model.iteration_bunch(nit, node_status=True)
trends = model.build_trends(iterations)


# model = ep.ThresholdModel(g)

# config = mc.Configuration()
# config.add_model_parameter('percentage_infected', 0.1)

# threshold = 0.25
# for i in g.nodes():
#     config.add_node_configuration("threshold", i, threshold)
# node attribute setting

# model.set_initial_status(config)


# %% Studio del network
# G = jack.graph_tools(G)
# jack.graph_plots(G)
k = G.degree()
G.degree_list = [d for n, d in k]
G.k_avg = np.mean(G.degree_list)

# %% Studio delle diverse fasi del contagio

# Fare un np.array o un data frame di pandas
# [[S,E,I,R] for S, E, I, R in
#  [list(it['node_count'].values()) for it in iterations]]

# Calcolo R0 e R(t) e plotto

# a priori
S = np.array([S for S, E, I, R in
              [list(it['node_count'].values()) for it in iterations]])
# Ii = np.array([I for S, I, R in
#               [list(it['node_count'].values()) for it in iterations]])

Ii = np.array([E+I for S, E, I, R in
              [list(it['node_count'].values()) for it in iterations]])

''' Controllare la definizione di Ro nel modello SEIR, perche` il valore
effettivo sembra convergere meno alla previsione
'''

D = int(1/gamma)
Ro = beta * D * G.k_avg
Rt = Ro * S/N

# smoothing in funzione di D per togliere il rumore stocastico
Ir = pd.Series(list(Ii)).rolling(window=D).mean().iloc[D-1:].values

# a posteriori
K = np.diff(np.log(Ir/N))
R = np.exp(K * D)


# %% Visual

# %matplotlib inline
viz = DiffusionTrend(model, trends)
viz.plot()


viz = DiffusionPrevalence(model, trends)
viz.plot()


figR = plt.figure(figsize=(7, 5))
plt.plot([1 for i in range(800)], 'k--')
plt.plot(Rt, 'r', label='R pred')
plt.plot(R, 'orange', label='R actual')
plt.legend(loc='best')
plt.grid()


# # differenza ritardata di D
# ini_list = np.log(I/N)  # 1-S/N

# diff_list = []
# for x, y in zip(ini_list[0::], ini_list[D::]):
#     diff_list.append(y-x)

# Kr = np.array(diff_list)
# Rr = np.exp(Kr * D)

# plt.plot(Rr, 'brown', label='R retD')


# Confronto diverse fasi di salita dei soli I


# import pandas as pd
# Ir = pd.Series(list(I)).rolling(window=D).mean().iloc[D-1:].values
# #define array to use and number of previous periods to use in calculation
# x = [50, 55, 36, 49, 84, 75, 101, 86, 80, 104]
# n=3

# #calculate moving average
# pd.Series(x).rolling(window=n).mean().iloc[n-1:].values

# array([47, 46.67, 56.33, 69.33, 86.67, 87.33, 89, 90])
