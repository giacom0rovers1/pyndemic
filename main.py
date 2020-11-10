#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:24:05 2020

@author: giacomo
"""
import pickle
import networkx as nx

from jack import SEIR, contagion_metrics, RandNemic  # SEIR_network,

N = 1e4
perc_inf = 0.1
days = 600
beta = 0.22         # infection probability
tau_i = 20          # incubation time
tau_r = 10          # recovery time
R0 = beta * tau_r   # basic reproduction number


# DETERMINISTIC well-mixed approach

s, e, i, r, t = SEIR(perc_inf, beta, tau_i, tau_r, days)

contagion_metrics(s, e, i, r, R0, tau_i, tau_r, N)


# COMPLEX NETWORKS approach

# # # Getting back the objects:
# with open('all_networks.pkl', 'rb') as f:
#     latti, erdos, watts, barab, holme = pickle.load(f)


# erdos = RandNemic('Erdos Renji',
#                   nx.erdos_renyi_graph(int(N), 0.005, seed=1234),
#                   'erdos.pkl')
# erdos.run(perc_inf, beta, tau_i, tau_r, days, t)
# erdos.save()


# TODO lattice e erdos dagli equivalenti di small world

watts = RandNemic('Watts Strogatz',
                  nx.connected_watts_strogatz_graph(int(N), 5, 0.1, seed=1234),
                  'watts.pkl')
watts.run(perc_inf, beta, tau_i, tau_r, days, t)
watts.save()

rando = RandNemic('Random reference',
                  nx.random_reference(watts.G),
                  'rando_ref.pkl')
rando.run(perc_inf, beta, tau_i, tau_r, days, t)
rando.save()

latti = RandNemic('Lattice reference',
                  nx.lattice_reference(watts.G),
                  'latti_ref.pkl')
latti.run(perc_inf, beta, tau_i, tau_r, days, t)
latti.save()


# Scale free senza e con clustering (ok entrambi)

barab = RandNemic('Barabasi Albert',
                  nx.barabasi_albert_graph(int(N),  3, seed=1234),
                  'barabasi.pkl')
barab.run(perc_inf, beta, tau_i, tau_r, days, t)
barab.save()


holme = RandNemic('Holme Kim',
                  nx.powerlaw_cluster_graph(int(N), 3, 0.1, seed=1234),
                  'holme.pkl')
holme.run(perc_inf, beta, tau_i, tau_r, days, t)
holme.save()


# Save all networks together with pickle()
with open('all_networks.pkl', 'wb') as f:
    pickle.dump([watts, rando, latti, barab, holme], f)


# TODO impostare un processo di ensemble?


# codice superato

# s, e, i, r = SEIR_network(latti.G, N, perc_inf, beta, tau_i, tau_r, days, t)
# contagion_metrics(s, e, i, r, R0, tau_i, tau_r, N)

# s, e, i, r = SEIR_network(erdos.G, N, perc_inf, beta, tau_i, tau_r, days, t)
# contagion_metrics(s, e, i, r, R0, tau_i, tau_r, N)

# s, e, i, r = SEIR_network(watts.G, N, perc_inf, beta, tau_i, tau_r, days, t)
# contagion_metrics(s, e, i, r, R0, tau_i, tau_r, N)

# s, e, i, r = SEIR_network(barab.G, N, perc_inf, beta, tau_i, tau_r, days, t)
# contagion_metrics(s, e, i, r, R0, tau_i, tau_r, N)

# s, e, i, r = SEIR_network(holme.G, N, perc_inf, beta, tau_i, tau_r, days, t)
# contagion_metrics(s, e, i, r, R0, tau_i, tau_r, N)
