#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 18:37:22 2020

@author: giacomo
"""
import jack
import networkx as nx

G = nx.barabasi_albert_graph( 125,  3, seed = 1234)

jack.graph_plots(G)