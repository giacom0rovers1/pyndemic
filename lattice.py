#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 18:46:24 2020

@author: giacomo
"""
import jack
import networkx as nx
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import probplot 

G = nx.grid_graph([5, 5, 5])

jack.graph_plots(G)

