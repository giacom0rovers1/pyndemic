#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:12:47 2020

@author: giacomo
"""
import jack
import networkx as nx

G = nx.erdos_renyi_graph(125, 0.1, seed = 1234)

jack.graph_plots(G)