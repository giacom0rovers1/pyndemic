#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:24:05 2020

@author: giacomo
"""
import os
import time
import pickle
import numpy as np
import networkx as nx

import pyndemic as pn

Tic = time.perf_counter()

N = 1e4
n = N/100
perc_inf = 0.1
days = 150  # 100 too short for Rando
daysl = days*2
daysll = days*3
beta = 0.73         # infection probability
tau_i = 3           # incubation time
tau_r = 3           # recovery time
R0 = beta * tau_r   # basic reproduction number

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# DETERMINISTIC well-mixed approach

# ==========
# SEIR MODEL
# ==========
print("\nSEIR deterministic model:")
s, e, i, r, t = pn.SEIR_odet(perc_inf, beta, tau_i, tau_r, days)

p = e + i
pos = N * p

mu = 1/tau_r
gamma = 1/tau_i
A = np.array([[-gamma, beta*s[0]], [gamma, -mu]])
eigval, eigvec = np.linalg.eig(A)
K0 = eigval[0]
ts0 = np.log(R0)/K0
pars0 = [K0, p[0]*N]
D = int(ts0)

x, xi, yi, KFit, Ki, tsFit, parsFit, \
    Rt, Rti, TdFit, Tdi = \
    pn.contagion_metrics(s, e, i, r, t, K0, ts0, pars0,
                         D, R0, tau_i, tau_r, N)

fig02, fig03, fig04 = pn.SEIR_plot(s, e, i, r, t, R0,
                                   "SEIR deterministic model",
                                   pos, ts0, pars0, x, xi, yi,
                                   parsFit, D, KFit, TdFit, Rt, Rti)

fig02.savefig('immagini/SEIR_02.png')
fig03.savefig('immagini/SEIR_03.png')
fig04.savefig('immagini/SEIR_04.png')
with open('pickle/SEIR.pkl', 'wb') as f:
    pickle.dump([s, e, i, r, t, days, daysl, KFit, tsFit, parsFit,
                 mu, gamma, R0, K0, ts0, pars0,
                 fig02, fig03, fig04], f)


# =========
# SIR MODEL
# =========
ddays = int(0.6 * days)  # int(days/2.3)
print("\nSIR deterministic model:")
ss, ii, rr, tt = pn.SIR_odet(perc_inf, beta, tau_r, ddays)

KK0 = beta*ss[0]-mu
tts0 = np.log(R0)/KK0
ppars0 = [KK0, ii[0]*N]
DD = int(2*tts0)
ppos = N * ii

xx, xxi, yyi, KKFit, KKi, ttsFit, pparsFit, \
    RRt, RRti, TTdFit, TTdi = \
    pn.contagion_metrics(ss, 0, ii, rr, tt, KK0, tts0, ppars0,
                         DD, R0, 0, tau_r, N)

ffig02, ffig03, ffig04 = pn.SIR_plot(ss, ii, rr, tt, R0,
                                     "SIR deterministic model",
                                     ppos, tts0, ppars0, xx, xxi, yyi,
                                     pparsFit, DD, KKFit, TTdFit, RRt, RRti)

ffig02.savefig('immagini/SIR_02.png')
ffig03.savefig('immagini/SIR_03.png')
ffig04.savefig('immagini/SIR_04.png')
with open('pickle/SIR.pkl', 'wb') as f:
    pickle.dump([ss, ii, rr, tt, ddays, KKFit, ttsFit, pparsFit,
                 mu, R0, KK0, tts0, ppars0,
                 ffig02, ffig03, ffig04], f)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# COMPLEX NETWORKS approach

if os.path.isfile('pickle/all_networks.pkl'):
    print("Loading existing networks...")

    # Getting back the objects:
    with open('pickle/all_networks.pkl', 'rb') as f:
        Watts, Rando, Latti, Barab, Holme = pickle.load(f)

else:
    print("No networks found, generating...")
    print("Random [1/5]")
    Rando = pn.randnet('Erdos-Renyi',
                       'random',
                       # nx.erdos_renyi_graph(10000, 12/10000, seed=1234)
                       nx.connected_watts_strogatz_graph(int(N), 12,
                                                         1, seed=1234),
                       nx.connected_watts_strogatz_graph(int(n), 12,
                                                         1, seed=1234))
    print("Lattice [2/5]")
    Latti = pn.randnet('Ring lattice',
                       'lattice',
                       nx.connected_watts_strogatz_graph(int(N), 12,
                                                         0, seed=1234),
                       nx.connected_watts_strogatz_graph(int(n), 12,
                                                         0, seed=1234))
    print("Small world [3/5]")
    Watts = pn.randnet('Watts-Strogatz',
                       'smallw',
                       nx.connected_watts_strogatz_graph(int(N), 12,
                                                         0.1, seed=1234),
                       nx.connected_watts_strogatz_graph(int(n), 12,
                                                         0.1, seed=1234))
    print("Scale free [4/5]")
    Barab = pn.randnet('Barabasi-Albert',
                       'scalefree',
                       nx.barabasi_albert_graph(int(N),  6, seed=1234),
                       nx.barabasi_albert_graph(int(n),  6, seed=1234))

    print("Realistic [5/5]")
    Holme = pn.randnet('Holme-Kim',
                       'realw',
                       nx.powerlaw_cluster_graph(int(N), 6, 1, seed=1234),
                       nx.powerlaw_cluster_graph(int(n), 6, 1, seed=1234))

    # Save all networks together with pickle()
    print('Saving networks...')
    with open('pickle/all_networks.pkl', 'wb') as f:
        pickle.dump([Watts, Rando, Latti, Barab, Holme], f)


if os.path.isfile('pickle/all_models.pkl'):
    print("Loading existing models...")
    # Getting back the objects:
    with open('pickle/all_models.pkl', 'rb') as f:
        watts, rando, latti, barab, holme = pickle.load(f)
else:
    watts = pn.pRandNeTmic(Watts, perc_inf, beta, tau_i, tau_r, daysll)
    rando = pn.pRandNeTmic(Rando, perc_inf, beta, tau_i, tau_r, days)
    latti = pn.pRandNeTmic(Latti, perc_inf, beta, tau_i, tau_r, daysl)
    barab = pn.pRandNeTmic(Barab, perc_inf, beta, tau_i, tau_r, days)
    holme = pn.pRandNeTmic(Holme, perc_inf, beta, tau_i, tau_r, days)

    # Save again all models together with pickle()
    with open('pickle/all_models.pkl', 'wb') as f:
        pickle.dump([watts, rando, latti, barab, holme], f)
# %%
# ==============
# RANDOM NETWORK
# ==============
print("\nSEIR over random network:")
# with open('pickle/network_random.pkl', 'rb') as f:
#     Rando = pickle.load(f)
# rando = pn.pRandNeTmic(Rando, perc_inf, beta, tau_i, tau_r, days)
rando.run(100)
rando.plot()
rando.save()


# %%
# =======
# LATTICE
# =======
print("\nSEIR over lattice network:")
# with open('pickle/network_lattice.pkl', 'rb') as f:
#     Latti = pickle.load(f)
# latti = pn.pRandNeTmic(Latti, perc_inf, beta, tau_i, tau_r, days)
latti.run(100)
latti.plot()
latti.save()


# %%
# # ===================
# # SMALL-WORLD NETWORK
# # ===================
print("\nSEIR over small-world network:")
# with open('pickle/network_smallw.pkl', 'rb') as f:
#     Watts = pickle.load(f)
# watts = pn.pRandNeTmic(Watts, perc_inf, beta, tau_i, tau_r, days)
watts.run(100)
watts.plot()
watts.save()

# %%
# ==================
# SCALE-FREE NETWORK
# ==================
print("\nSEIR over scale-free network:")
# with open('pickle/network_scalefree.pkl', 'rb') as f:
#     Barab = pickle.load(f)
# barab = pn.pRandNeTmic(Barab, perc_inf, beta, tau_i, tau_r, days)
barab.run(100)
barab.plot()
barab.save()


# %%
# ==========================
# SCALE-FREE WITH CLUSTERING
# ==========================
print("\nSEIR over clustered scale-free network:")
# with open('pickle/network_realw.pkl', 'rb') as f:
#     Holme = pickle.load(f)
# holme = pn.pRandNeTmic(Holme, perc_inf, beta, tau_i, tau_r, days)
holme.run(100)
holme.plot()
holme.save()


Toc = time.perf_counter()
print("All done. [Elapsed: " + str(round(Toc-Tic, 0)) + " seconds]")


# %%
for net in [Watts, Rando, Latti, Barab, Holme]:
    print([net.name,
           net.G.size(),
           net.G.number_of_edges(),
           net.G.number_of_nodes(),
           net.G.k_avg,
           net.G.C_avg])


with open('pickle/simulations_random.pkl', 'rb') as f:
    rando = pickle.load(f)

with open('pickle/simulations_smallw.pkl', 'rb') as f:
    watts = pickle.load(f)

with open('pickle/simulations_lattice.pkl', 'rb') as f:
    latti = pickle.load(f)

with open('pickle/simulations_scalefree.pkl', 'rb') as f:
    barab = pickle.load(f)

with open('pickle/simulations_realw.pkl', 'rb') as f:
    holme = pickle.load(f)

# Save again all networks together with pickle()
with open('pickle/all_simulations.pkl', 'wb') as f:
    pickle.dump([watts, rando, latti, barab, holme], f)

# end
