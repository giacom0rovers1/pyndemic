#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 22:35:53 2020

@author: giacomo
"""
import scipy as sp
import datetime as dt
import os
# import copy
import pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

import random

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
# from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend

import networkx as nx
import pyndemic as pn

# import salience_unw as sl

# Reduction factor: voglio la stessa probabilità ma ho <k> inferiore
redfa = 0.6
d0 = 14
# runs = 100


N = 1e4
n = N/100
perc_inf = 0.1
days = 150
beta = 0.061          # infection probability
avgk = 12 * redfa     # corrected average contacts
lmbda = beta * avgk    # infection rate
tau_i = 3             # incubation time
tau_r = 3             # recovery time
R0 = lmbda * tau_r     # basic reproduction number


# %%
# Mitigation based on node metric

def attack_list(graph, ranklist, thr):
    nodes = sorted(graph.nodes(), key=lambda n: ranklist[n])

    while graph.k_avg > thr:
        # remove node with highest rank until reaching an avg degree threshold
        graph.remove_node(nodes.pop())

        # update average connectivity degree
        k = graph.degree()
        graph.degree_list = [d for n, d in k]
        graph.k_avg = np.mean(graph.degree_list)

    return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()


with open('pickle/network_realw.pkl', 'rb') as f:
    Holme = pickle.load(f)

with open('pickle/simulations_realw.pkl', 'rb') as f:
    holme = pickle.load(f)

if os.path.isfile('pickle/network_lockHiBC_connected.pkl'):
    print("Loading existing networks...")

    # Getting back the objects:
    with open('pickle/network_lockHiBC_connected.pkl', 'rb') as f:
        Holme_lbc = pickle.load(f)

else:
    print("No networks found, generating...")
    print("Holme - lock HiBC [1/5]")
    Holme_lbc = pn.randnet("HK lockdown scenario",
                           "lockHiBC_connected",
                           attack_list(Holme.G, Holme.G.BC_list, 12*redfa),
                           attack_list(Holme.Gmini, Holme.G.BC_list, 12*redfa))

G = Holme_lbc.G


# %%

with open('pickle/SEIR.pkl', 'rb') as f:
    s, e, i, r, t, days, daysl, KFit, tsFit, parsFit, \
        mu, gamma, R0, K0, ts0, pars0, \
        fig02, fig03, fig04 = pickle.load(f)

print("\nSEIR deterministic model:")
# calculate constants
frac_inf = perc_inf/100
gamma = 1/tau_i
mu = 1/tau_r

p = e + i
ds0 = np.min(np.where(p > holme.pos[d0]/holme.N))
y0 = np.array([s[ds0], e[ds0], i[ds0], r[ds0]])

y = y0


def dydt(t, y):
    return np.array([-lmbda*y[0]*y[2],                   # ds/dt
                     lmbda*y[0]*y[2] - gamma*y[1],       # de/dt
                     gamma*y[1] - mu*y[2],              # di/dt
                     mu*y[2]])                          # dr/dt


y = sp.integrate.solve_ivp(fun=dydt,
                           t_span=(ds0, days+ds0),
                           y0=y0,
                           t_eval=np.arange(ds0, days+ds0+1))

s2, e2, i2, r2 = [y.y[line, :] for line in range(4)]
t2 = y.t

s = np.append(s[:ds0], s2)
e = np.append(e[:ds0], e2)
i = np.append(i[:ds0], i2)
r = np.append(r[:ds0], r2)
t = np.append(t[:ds0], t2)

p = e + i
pos = N * p

mu = 1/tau_r
gamma = 1/tau_i
A = np.array([[-gamma, lmbda*s[0]], [gamma, -mu]])
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
                                   "SEIR det. with lockdown",
                                   pos, ts0, pars0, x, xi, yi,
                                   parsFit, D, KFit, TdFit, Rt, Rti)


fig02.savefig('immagini/SEIR_02lockdown2.png')
fig03.savefig('immagini/SEIR_03lockdown2.png')
fig04.savefig('immagini/SEIR_04lockdown2.png')
with open('pickle/SEIRlockdown.pkl', 'wb') as f:
    pickle.dump([s, e, i, r, t, days, days, KFit, tsFit, parsFit,
                 mu, gamma, R0, K0, ts0, pars0,
                 fig02, fig03, fig04], f)

# %%

frac_inf = 0.00011  # (one infected just to avoid warnings)
gamma = 1/tau_i
mu = 1/tau_r

# # debug run
# Holme_lbc.nick = "lockHiBC_test"

lock = pn.pRandNeTmic(Holme_lbc, holme.i[d0]*100,
                      beta, tau_i, tau_r, days)

lock.mu = 1/tau_r
lock.gamma = 1/tau_i
A = np.array([[-lock.gamma, lock.lmbda*s[0]], [lock.gamma, -lock.mu]])
eigval, eigvec = np.linalg.eig(A)
lock.K0 = eigval[0]
lock.ts0 = np.log(lock.R0)/lock.K0
lock.pars0 = [lock.K0, p[0]*lock.N]
lock.D = int(lock.ts0)

# %%
# SIMULATION
lock.t = t
lock.runs = holme.runs

lock.sm = pd.Series(data=None, dtype='float64')
lock.em = pd.Series(data=None, dtype='float64')
lock.im = pd.Series(data=None, dtype='float64')
lock.rm = pd.Series(data=None, dtype='float64')
lock.pm = pd.Series(data=None, dtype='float64')

lock.parsFitm0 = pd.Series(data=None, dtype='float64')
lock.parsFitm1 = pd.Series(data=None, dtype='float64')
lock.Rtim = pd.Series(data=None, dtype='float64')

# run n simulations
run = 0
member = lock.copy()
# Config:
model = ep.SEIRModel(G)

# print(model.parameters)
# print(model.available_statuses)

config = mc.Configuration()
config.add_model_parameter('alpha', gamma)
config.add_model_parameter('beta',  beta)
config.add_model_parameter('gamma', mu)
config.add_model_parameter("percentage_infected", 0.00011)

while run < lock.runs:
    print("\n" + str(run+1) + " of " + str(lock.runs))

    model.set_initial_status(config)

    # bruteforce
    init = model.initial_status
    for ind in list(init):
        init[ind] = 0

    i = 0
    while i < int(list(holme.em[d0])[run]*N):
        idx = random.choice(list(init))
        if init[idx] == 0:
            init[idx] = 2
            i += 1
    i = 0
    while i < int(list(holme.im[d0])[run]*N):
        idx = random.choice(list(init))
        if init[idx] == 0:
            init[idx] = 1
            i += 1
    i = 0
    while i < int(list(holme.rm[d0])[run]*N):
        idx = random.choice(list(init))
        if init[idx] == 0:
            init[idx] = 3
            i += 1

    model.initial_status = init

    # Run:
    iterations = model.iteration_bunch(days, node_status=True)
    # trends = model.build_trends(iterations)

    # viz = DiffusionTrend(model, trends)
    # viz.plot()

    # Recover status variables:
    s = np.array([S for S, E, I, R in
                  [list(it['node_count'].values()) for it in iterations]])/N
    e = np.array([E for S, E, I, R in
                  [list(it['node_count'].values()) for it in iterations]])/N
    i = np.array([I for S, E, I, R in
                  [list(it['node_count'].values()) for it in iterations]])/N
    r = np.array([R for S, E, I, R in
                  [list(it['node_count'].values()) for it in iterations]])/N

    sel = (holme.days+1)*run
    s = np.append(np.array(list(holme.sm[sel:(sel+d0)])), s)
    e = np.append(np.array(list(holme.em[sel:(sel+d0)])), e)
    i = np.append(np.array(list(holme.im[sel:(sel+d0)])), i)
    r = np.append(np.array(list(holme.rm[sel:(sel+d0)])), r)

    # resampling through t (variable spacing decided by the ODE solver)
    member.s = np.interp(t, np.arange(0, len(s)), s)
    member.e = np.interp(t, np.arange(0, len(e)), e)
    member.i = np.interp(t, np.arange(0, len(i)), i)
    member.r = np.interp(t, np.arange(0, len(r)), r)
    member.t = lock.t
    member.pos = np.array((member.e + member.i) * lock.N)

    try:
        member.x, member.xi, member.yi, \
            member.KFit, member.Ki, member.tsFit, member.parsFit, \
            member.Rt, member.Rti, \
            member.TdFit, member.Tdi = \
            pn.contagion_metrics(s=member.s, e=member.e, i=member.i,
                                 r=member.r, t=lock.t,
                                 K0=lock.K0, ts0=lock.ts0,
                                 pars0=lock.pars0,
                                 D=lock.D, R0=lock.R0,
                                 tau_i=lock.tau_i,
                                 tau_r=lock.tau_r, N=lock.N)
        run += 1

    except ValueError:
        now = dt.datetime.now()
        print("\nVALUE ERROR OCCURRED in contagion_metrics()")
        print(now)
        logname = 'pickle/lock_valerror_' + \
            now.strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'
        with open(logname, 'wb') as f:
            pickle.dump([member, run, now], f)
        print("Error log saved. Repeating run " + str(run+1))
        run += 1
        # altrimenti se il problema e` contenuto nell'ensemble di
        # partenza, l'errore si ripete all'infinito. (run 63)
        continue

    lock.sm = lock.sm.append(pd.Series(member.s))
    lock.em = lock.em.append(pd.Series(member.e))
    lock.im = lock.im.append(pd.Series(member.i))
    lock.rm = lock.rm.append(pd.Series(member.r))
    lock.pm = lock.pm.append(pd.Series(member.pos))

    lock.parsFitm0 = \
        lock.parsFitm0.append(pd.Series(member.parsFit[0]))
    lock.parsFitm1 = \
        lock.parsFitm1.append(pd.Series(member.parsFit[1]))

    lock.Rtim = lock.Rtim.append(pd.Series(member.Rti))

lock.s = np.array([lock.sm[i].median() for i in lock.t])
lock.e = np.array([lock.em[i].median() for i in lock.t])
lock.i = np.array([lock.im[i].median() for i in lock.t])
lock.r = np.array([lock.rm[i].median() for i in lock.t])
lock.pos = np.array([lock.pm[i].median() for i in lock.t])

lock.s05 = np.array([lock.sm[i].quantile(0.05) for i in lock.t])
lock.e05 = np.array([lock.em[i].quantile(0.05) for i in lock.t])
lock.i05 = np.array([lock.im[i].quantile(0.05) for i in lock.t])
lock.r05 = np.array([lock.rm[i].quantile(0.05) for i in lock.t])
lock.p05 = np.array([lock.pm[i].quantile(0.05) for i in lock.t])

lock.s95 = np.array([lock.sm[i].quantile(0.95) for i in lock.t])
lock.e95 = np.array([lock.em[i].quantile(0.95) for i in lock.t])
lock.i95 = np.array([lock.im[i].quantile(0.95) for i in lock.t])
lock.r95 = np.array([lock.rm[i].quantile(0.95) for i in lock.t])
lock.p95 = np.array([lock.pm[i].quantile(0.95) for i in lock.t])

# Contagion metrics of the median scenario
lock.x, lock.xi, lock.yi, \
    lock.KFit, lock.Ki, lock.tsFit, lock.parsFit, \
    lock.Rt, lock.Rti, \
    lock.TdFit, lock.Tdi = \
    pn.contagion_metrics(s=lock.s, e=lock.e, i=lock.i,
                         r=lock.r, t=lock.t,
                         K0=lock.K0, ts0=lock.ts0,
                         pars0=lock.pars0,
                         D=lock.D, R0=lock.R0, tau_i=lock.tau_i,
                         tau_r=lock.tau_r, N=lock.N)

lock.Rt05 = lock.R0 * lock.s05
lock.Rt95 = lock.R0 * lock.s95

lock.parsFit50 = [lock.parsFitm0.median(), lock.parsFitm1.median()]
lock.parsFit05 = [lock.parsFitm0.quantile(0.05),
                  lock.parsFitm1.quantile(0.05)]
lock.parsFit95 = [lock.parsFitm0.quantile(0.95),
                  lock.parsFitm1.quantile(0.95)]

lock.KFit50 = lock.parsFit50[0]
lock.TdFit50 = np.log(2)/lock.KFit50

lock.Rti50 = np.array([lock.Rtim[i].median() for i in lock.t])
lock.Rti05 = np.array([lock.Rtim[i].quantile(0.05)
                       for i in lock.t])
lock.Rti95 = np.array([lock.Rtim[i].quantile(0.95)
                       for i in lock.t])


# %%

lock.plot()
lock.save()


# %% fix

# with open('pickle/simulations_lockHighBC.pkl', 'rb') as f:
#     lock = pickle.load(f)

# lock.name = "HK lockdown scenario"

# lock.Gmini = pn.graph_plots(lock.Gmini, lock.name, [0])
# lock.fig00 = lock.Gmini.fig0

# lock.G = pn.graph_plots(lock.G, lock.name, [1])
# lock.fig01 = lock.G.fig1


# %% Late hosptalization scenario

with open('pickle/all_networks.pkl', 'rb') as f:
    Watts, Rando, Latti, Barab, Holme = pickle.load(f)
N = 1e4
n = N/100
perc_inf = 0.1
days = 150  # 100 too short for Rando
daysl = days*2
daysll = days*3
avgk = 12
beta = 0.061        # infection probability
lmbda = beta * avgk  # infection rate
tau_i = 3           # incubation time
tau_r = 7
R0 = lmbda * tau_r   # basic reproduction number

nawarHK = pn.pRandNeTmic(Holme, perc_inf, beta, tau_i, tau_r, days)
nawarHK.name = "HK late hospit. scenario"
nawarHK.nick = "HK_hiTau"

nawarHK.run(100)
nawarHK.plot()
nawarHK.save()
