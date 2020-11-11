#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:24:05 2020

@author: giacomo
"""
import pickle
import networkx as nx

from jack import SEIR, SIR, contagion_metrics, RandNemic  # SEIR_network,

N = 1e4
perc_inf = 0.1
days = 600
beta = 0.22         # infection probability
tau_i = 20          # incubation time
tau_r = 10          # recovery time
R0 = beta * tau_r   # basic reproduction number


# DETERMINISTIC well-mixed approach

s, e, i, r, t = SEIR(perc_inf, beta, tau_i, tau_r, days)

K0, Ki, Ks, R0_K, Rt, Rti, Rts, Td0, Tdi, Tds = \
    contagion_metrics(s, e, i, r, t, R0, tau_i, tau_r, N)

ss, ii, rr, tt = SIR(perc_inf, beta, tau_r, days)
KK0, KKi, KKs, RR0_K, RRt, RRti, RRts, TTd0, TTdi, TTds = \
    contagion_metrics(ss, 0, ii, rr, tt, R0, 0, tau_r, N)

# ss, ii, rr, tt = SIR(perc_inf*0.1, beta/5, tau_r*5, days)
# contagion_metrics(ss, 0, ii, rr, R0, 0, int(tau_r*0.5), N)


# COMPLEX NETWORKS approach

# # # Getting back the objects:
# with open('all_networks.pkl', 'rb') as f:
#     latti, erdos, watts, barab, holme = pickle.load(f)

# erdos = RandNemic('Erdos Renji',
#                   nx.erdos_renyi_graph(int(N), 0.005, seed=1234), #0.0006
#                   'erdos.pkl')
# erdos.run(perc_inf, beta, tau_i, tau_r, days, t)
# erdos.save()


# ===================
# SMALL-WORLD NETWORK
# ===================
watts = RandNemic('Watts-Strogatz',
                  nx.connected_watts_strogatz_graph(int(N), 50,
                                                    0.1, seed=1234),
                  'watts.pkl')
watts.run(perc_inf, beta, tau_i, tau_r, days, t)
# with open('watts.pkl', 'rb') as f:
#     watts = pickle.load(f)
watts.plot(beta, tau_i, tau_r, days, t)
watts.save()
# ora lattice e erdos dagli equivalenti di small world


# ==============
# RANDOM NETWORK
# ==============
rando = RandNemic('Erdos-Renyi',
                  nx.connected_watts_strogatz_graph(int(N), 50,
                                                    1, seed=1234),
                  'random.pkl')
rando.run(perc_inf, beta, tau_i, tau_r, days, t)
# with open('random.pkl', 'rb') as f:
#     rando = pickle.load(f)
rando.plot(beta, tau_i, tau_r, days, t)
rando.save()


# =======
# LATTICE
# =======
latti = RandNemic('Ring lattice',
                  nx.connected_watts_strogatz_graph(int(N), 50,
                                                    0, seed=1234),
                  'lattice.pkl')
latti.run(perc_inf, beta, tau_i, tau_r, days, t)
# with open('lattice.pkl', 'rb') as f:
#     latti = pickle.load(f)
latti.plot(beta, tau_i, tau_r, days, t)
latti.save()


# ==================
# SCALE-FREE NETWORK
# ==================
barab = RandNemic('Barabasi-Albert',
                  nx.barabasi_albert_graph(int(N),  3, seed=1234),
                  'barabasi.pkl')
barab.run(perc_inf, beta, tau_i, tau_r, days, t)
# with open('barabasi.pkl', 'rb') as f:
#     barab = pickle.load(f)
barab.plot(beta, tau_i, tau_r, days, t)
barab.save()


# ==========================
# SCALE-FREE WITH CLUSTERING
# ==========================
holme = RandNemic('Holme-Kim',
                  nx.powerlaw_cluster_graph(int(N), 3, 0.1, seed=1234),
                  'holme.pkl')
holme.run(perc_inf, beta, tau_i, tau_r, days, t)
# with open('holme.pkl', 'rb') as f:
#     holme = pickle.load(f)
holme.plot(beta, tau_i, tau_r, days, t)
holme.save()


# Save all networks together with pickle()
with open('all_networks.pkl', 'wb') as f:
    pickle.dump([watts, rando, latti, barab, holme], f)


# TODO impostare un processo di ensemble?
