#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 18:46:59 2020

@author: giacomo
"""
import jack
import networkx as nx

G = nx.connected_watts_strogatz_graph(125, 4, 0.05, seed=1234)

G = jack.graph_tools(G)

jack.graph_plots(G, [1, 2])
