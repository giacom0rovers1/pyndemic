#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:24:05 2020

@author: giacomo
"""
import time
import pickle
import numpy as np
import networkx as nx

from pyndemic import SEIR_odet, SIR_odet, contagion_metrics, pRandNeTmic
Tic = time.perf_counter()

N = 1e4
perc_inf = 0.1
days = 600
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
p = e + i
mu = 1/tau_r
gamma = 1/tau_i
A = np.array([[-gamma, beta*s[0]], [gamma, -mu]])
eigval, eigvec = np.linalg.eig(A)
K0 = eigval[0]
ts0 = np.log(R0)/K0
pars0 = [K0, p[0]*N]

K, Ki, ts, pars, Rt, Rti, Rts, Td0, Tdi, Tds, fig03, fig04 = \
    contagion_metrics(s, e, i, r, t, K0, ts0, pars0,
                      R0, tau_i, tau_r, N,
                      "SEIR deterministic model")

fig02.savefig('immagini/SEIR_02.png')
fig03.savefig('immagini/SEIR_03.png')
fig04.savefig('immagini/SEIR_04.png')
with open('pickle/SEIR.pkl', 'wb') as f:
    pickle.dump([s, e, i, r, t, K, ts, pars,
                 mu, gamma, R0, K0, ts0, pars0,
                 fig02, fig03, fig04], f)

# =========
# SIR MODEL
# =========
print("\nSIR deterministic model:")
ss, ii, rr, tt, ffig02 = SIR_odet(perc_inf, beta, tau_r, 250,
                                  "SIR deterministic model")
mu = 1/tau_r
KK0 = beta*ss[0]-mu
tts0 = np.log(R0)/KK0
ppars0 = [KK0, ii[0]*N]

KK, KKi, tts, ppars, RRt, RRti, RRts, TTd0, TTdi, TTds, ffig03, ffig04 = \
    contagion_metrics(ss, 0, ii, rr, tt, KK0, tts0, ppars0,
                      R0, 0, tau_r, N,
                      "SIR deterministic model")

ffig02.savefig('immagini/SIR_02.png')
ffig03.savefig('immagini/SIR_03.png')
ffig04.savefig('immagini/SIR_04.png')
with open('pickle/SIR.pkl', 'wb') as f:
    pickle.dump([ss, ii, rr, tt, KK, tts, ppars,
                 mu, R0, KK0, tts0, ppars0,
                 ffig02, ffig03, ffig04], f)

# ss, ii, rr, tt = SIR(perc_inf*0.1, beta/5, tau_r*5, days)
# contagion_metrics(ss, 0, ii, rr, R0, 0, int(tau_r*0.5), N)


# COMPLEX NETWORKS approach

# # # Getting back the objects:
# with open('pickle/all_networks.pkl', 'rb') as f:
#     watts, rando, latti, barab, holme = pickle.load(f)

# erdos = pRandNeTmic('Erdos Renji',
#                   nx.erdos_renyi_graph(int(N), 0.005, seed=1234), #0.0006
#                   'erdos.pkl')
# erdos.run(perc_inf, beta, tau_i, tau_r, days, t)
# erdos.save()


# ==============
# RANDOM NETWORK
# ==============
print("\nSEIR over random network:")
rando = pRandNeTmic('Erdos-Renyi',
                    'random',
                    nx.connected_watts_strogatz_graph(int(N), 12,
                                                      1, seed=1234),
                    nx.connected_watts_strogatz_graph(int(N/100), 12,
                                                      1, seed=1234))
rando.run(perc_inf, beta, tau_i, tau_r, days, t)

# with open('pickle/random.pkl', 'rb') as f:
#     rando = pickle.load(f)
rando.plot(beta, tau_i, tau_r, days, t, K0, ts0, pars0)
rando.save()


# =======
# LATTICE
# =======
print("\nSEIR over lattice network:")

# # special time interval for slow epidemic propagation on lattice network
# days = 4 * 600
# s, e, i, r, t, fig02 = SEIR_odet(perc_inf, beta, tau_i, tau_r, days,
#                                  "SEIR deterministic model")

latti = pRandNeTmic('Ring lattice',
                    'lattice',
                    nx.connected_watts_strogatz_graph(int(N), 12,
                                                      0, seed=1234),
                    nx.connected_watts_strogatz_graph(int(N/100), 12,
                                                      0, seed=1234))
latti.run(perc_inf, beta, tau_i, tau_r, days, t)

# with open('pickle/lattice.pkl', 'rb') as f:
#     latti = pickle.load(f)
latti.plot(beta, tau_i, tau_r, days, t, K0, ts0, pars0)
latti.save()

# # Back to normal
# days = 600
# s, e, i, r, t, fig02 = SEIR_odet(perc_inf, beta, tau_i, tau_r, days,
#                                  "SEIR deterministic model")

# # ===================
# # SMALL-WORLD NETWORK
# # ===================
# print("\nSEIR over small-world network:")

# watts = pRandNeTmic('Watts-Strogatz',
#                     'smallw',
#                     nx.connected_watts_strogatz_graph(int(N), 12,
#                                                       0.1, seed=1234),
#                     nx.connected_watts_strogatz_graph(int(N/100), 12,
#                                                       0.1, seed=1234))
# watts.run(perc_inf, beta, tau_i, tau_r, days, t)

# # with open('pickle/smallw.pkl', 'rb') as f:
# #     watts = pickle.load(f)
# watts.plot(beta, tau_i, tau_r, days, t, K0, ts0, pars0)
# watts.save()


# ==================
# SCALE-FREE NETWORK
# ==================
print("\nSEIR over scale-free network:")
barab = pRandNeTmic('Barabasi-Albert',
                    'scalefree',
                    nx.barabasi_albert_graph(int(N),  6, seed=1234),
                    nx.barabasi_albert_graph(int(N/100),  6, seed=1234))
barab.run(perc_inf, beta, tau_i, tau_r, days, t)

# with open('pickle/scalefree.pkl', 'rb') as f:
#     barab = pickle.load(f)
barab.plot(beta, tau_i, tau_r, days, t, K0, ts0, pars0)
barab.save()


# ==========================
# SCALE-FREE WITH CLUSTERING
# ==========================
print("\nSEIR over clustered scale-free network:")
holme = pRandNeTmic('Holme-Kim',
                    'realw',
                    nx.powerlaw_cluster_graph(int(N), 6, 0.1, seed=1234),
                    nx.powerlaw_cluster_graph(int(N/100), 6, 0.1, seed=1234))
holme.run(perc_inf, beta, tau_i, tau_r, days, t)

# with open('pickle/realw.pkl', 'rb') as f:
#     holme = pickle.load(f)
holme.plot(beta, tau_i, tau_r, days, t, K0, ts0, pars0)
holme.save()


# TODO impostare un processo di ensemble?
# Magari solo per Holme-Kim

Toc = time.perf_counter()
print("All done. [Elapsed: " + str(round(Toc-Tic, 0)) + " seconds]")


for net in [rando, latti, barab, holme]:
    print([net.name,
           net.G.size(),
           net.G.number_of_edges(),
           net.G.number_of_nodes(),
           net.G.k_avg,
           net.G.C_avg])


# Load the last network (days=1600)
with open('watts_long.py') as fd:
    exec(fd.read())

with open('pickle/smallw_long.pkl', 'rb') as f:
    watts_long = pickle.load(f)

# Save all networks together with pickle()
with open('pickle/all_networks.pkl', 'wb') as f:
    pickle.dump([watts_long, rando, latti, barab, holme], f)
