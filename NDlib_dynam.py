#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 20:01:30 2020

@author: giacomo
"""
import networkx as nx
import dynetx as dn
import ndlib.models.dynamic as dm
import ndlib.models.ModelConfig as mc
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
# from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence

# Dynamic Network topology
dg = dn.DynGraph()

# Naive synthetic dynamic graph
# At each timestep t a new graph having the same set of node ids is created
for t in range(0, 30):
    # g = nx.erdos_renyi_graph(1000, 0.1)
    g = nx.connected_watts_strogatz_graph(200, 10, 0.1)
    dg.add_interactions_from(g.edges(), t)

# Model selection
model = dm.DynSIRModel(dg)

# Model Configuration
config = mc.Configuration()
config.add_model_parameter('beta',  0.01)  # infection rate
config.add_model_parameter('gamma', 0.01)  # recovery rate
# config.add_model_parameter('lambda', 0.01)  # recovery rate
config.add_model_parameter('alpha', 0.01)  # latent period (units?)
config.add_model_parameter("percentage_infected", 0.1)

model.set_initial_status(config)


# Simulate snapshot based execution
snapshots = model.execute_snapshots()

# Simulation interaction graph based execution
interactions = model.execute_iterations()

trends = model.build_trends(interactions)

viz = DiffusionTrend(model, trends)
viz.plot()
