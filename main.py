#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:24:05 2020

@author: giacomo
"""
from jack import SEIR, contagion_metrics
import networkx as nx

N = 1e4
perc_inf = 0.1
days = 600
beta = 0.22
tau = 20
tar = 10

# DETERMINISTIC well-mixed approach

s, e, i, r, t = SEIR(perc_inf, beta, tau, tar, days)

R0 = beta * tar

contagion_metrics(s, e, i, r, R0, tau, tar, N)

# COMPLEX NETWORKS


