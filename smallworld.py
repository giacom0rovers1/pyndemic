#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 18:46:59 2020

@author: giacomo
"""
import jack
import networkx as nx

G = nx.connected_watts_strogatz_graph(125, 10, 0.1, seed=1234)
# aumentare il numero di nodi!

G = jack.graph_tools(G)

jack.graph_plots(G)  # , [1, 2])

# G.sigma = nx.sigma(G)  # heavy  # small world if >1
# G.omega = nx.omega(G)  # heavy

wsR = jack.graph_tools(nx.random_reference(G))
wsL = jack.graph_tools(nx.lattice_reference(G))


jack.graph_plots(wsR)
jack.graph_plots(wsL)
