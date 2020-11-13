#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:24:05 2020

@author: giacomo
"""
import pickle
import networkx as nx

from pyndemic import SEIR_odet, SIR_odet, contagion_metrics, pRandNeTmic

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
s, e, i, r, t, fig02 = SEIR_odet(perc_inf, beta, tau_i, tau_r, days,
                                 "SEIR deterministic model")

K0, Ki, Ks, R0_K, Rt, Rti, Rts, Td0, Tdi, Tds, fig03, fig04 = \
    contagion_metrics(s, e, i, r, t, R0, tau_i, tau_r, N,
                      "SEIR deterministic model")
fig02.savefig('immagini/SEIR_02.png')
fig03.savefig('immagini/SEIR_03.png')
fig04.savefig('immagini/SEIR_04.png')


# =========
# SIR MODEL
# =========
ss, ii, rr, tt, ffig02 = SIR_odet(perc_inf, beta, tau_r, 250,
                                  "SIR deterministic model")
KK0, KKi, KKs, RR0_K, RRt, RRti, RRts, TTd0, TTdi, TTds, ffig03, ffig04 = \
    contagion_metrics(ss, 0, ii, rr, tt, R0, 0, tau_r, N,
                      "SIR deterministic model")
ffig02.savefig('immagini/SIR_02.png')
ffig03.savefig('immagini/SIR_03.png')
ffig04.savefig('immagini/SIR_04.png')

# ss, ii, rr, tt = SIR(perc_inf*0.1, beta/5, tau_r*5, days)
# contagion_metrics(ss, 0, ii, rr, R0, 0, int(tau_r*0.5), N)


# COMPLEX NETWORKS approach

# # # Getting back the objects:
# with open('all_networks.pkl', 'rb') as f:
#     latti, erdos, watts, barab, holme = pickle.load(f)

# erdos = pRandNeTmic('Erdos Renji',
#                   nx.erdos_renyi_graph(int(N), 0.005, seed=1234), #0.0006
#                   'erdos.pkl')
# erdos.run(perc_inf, beta, tau_i, tau_r, days, t)
# erdos.save()


# ==============
# RANDOM NETWORK
# ==============
rando = pRandNeTmic('Erdos-Renyi',
                    'random',
                    nx.connected_watts_strogatz_graph(int(N), 50,
                                                      1, seed=1234),
                    nx.connected_watts_strogatz_graph(int(N/100), 50,
                                                      1, seed=1234))
rando.run(perc_inf, beta, tau_i, tau_r, days, t)

# with open('random.pkl', 'rb') as f:
#     rando = pickle.load(f)
rando.plot(beta, tau_i, tau_r, days, t)
rando.save()


# =======
# LATTICE
# =======
latti = pRandNeTmic('Ring lattice',
                    'lattice',
                    nx.connected_watts_strogatz_graph(int(N), 50,
                                                      0, seed=1234),
                    nx.connected_watts_strogatz_graph(int(N/100), 50,
                                                      0, seed=1234))
latti.run(perc_inf, beta, tau_i, tau_r, days, t)

# with open('lattice.pkl', 'rb') as f:
#     latti = pickle.load(f)
latti.plot(beta, tau_i, tau_r, days, t)
latti.save()


# ===================
# SMALL-WORLD NETWORK
# ===================
watts = pRandNeTmic('Watts-Strogatz',
                    'smallw',
                    nx.connected_watts_strogatz_graph(int(N), 50,
                                                      0.1, seed=1234),
                    nx.connected_watts_strogatz_graph(int(N/100), 50,
                                                      0.1, seed=1234))
watts.run(perc_inf, beta, tau_i, tau_r, days, t)

# with open('smallw.pkl', 'rb') as f:
#     watts = pickle.load(f)
watts.plot(beta, tau_i, tau_r, days, t)
watts.save()


# ==================
# SCALE-FREE NETWORK
# ==================
barab = pRandNeTmic('Barabasi-Albert',
                    'scalefree',
                    nx.barabasi_albert_graph(int(N),  3, seed=1234),
                    nx.barabasi_albert_graph(int(N/100),  3, seed=1234))
barab.run(perc_inf, beta, tau_i, tau_r, days, t)

# with open('scalefree.pkl', 'rb') as f:
#     barab = pickle.load(f)
barab.plot(beta, tau_i, tau_r, days, t)
barab.save()


# ==========================
# SCALE-FREE WITH CLUSTERING
# ==========================
holme = pRandNeTmic('Holme-Kim',
                    'realw',
                    nx.powerlaw_cluster_graph(int(N), 3, 0.1, seed=1234),
                    nx.powerlaw_cluster_graph(int(N/100), 3, 0.1, seed=1234))
holme.run(perc_inf, beta, tau_i, tau_r, days, t)

# with open('realw.pkl', 'rb') as f:
#     holme = pickle.load(f)
holme.plot(beta, tau_i, tau_r, days, t)
holme.save()


# Save all networks together with pickle()
with open('pickle/all_networks.pkl', 'wb') as f:
    pickle.dump([watts, rando, latti, barab, holme], f)


# TODO impostare un processo di ensemble?
# Magari solo per Holme-Kim
