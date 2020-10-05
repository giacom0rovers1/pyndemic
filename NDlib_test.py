#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 19:26:21 2020

@author: giacomo
"""
# import ndlib as nd

# to UPDATE NDlib, run: 
# $ pip install git+git://github.com/GiulioRossetti/ndlib.git
# inside:
# /home/giacomo/anaconda3/lib/python3.8/site-packages/


import networkx as nx

# g = nx.erdos_renyi_graph(1000, 0.1)
g = nx.connected_watts_strogatz_graph(200,5,0.1)


# interessante: con 1000,10,0.1 il comportamento e` simile a quello di un grafo random connesso
#               con 1000, 5, 0.1 invece c'e` una sorta di immunita` di gregge nei nodi con alta centralita` presumo
# riflettere su come simulare il lockdown e sui modelli nulli

import ndlib.models.epidemics as ep

model = ep.SEIRModel(g)
print(model.parameters)
print(model.available_statuses)



import ndlib.models.ModelConfig as mc

config = mc.Configuration()
config.add_model_parameter('beta',  0.01) # infection rate
config.add_model_parameter('gamma', 0.01) # recovery rate
# config.add_model_parameter('lambda', 0.01) # recovery rate
config.add_model_parameter('alpha', 0.01)  # latent period (units?)
config.add_model_parameter("percentage_infected", 0.01)
model.set_initial_status(config)


iterations = model.iteration_bunch(2000, node_status=True)
trends    = model.build_trends(iterations)


# %matplotlib inline
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
viz = DiffusionTrend(model, trends)
viz.plot()


# from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence
# viz = DiffusionPrevalence(model, trends)
# viz.plot()



# model = ep.ThresholdModel(g)

# config = mc.Configuration()
# config.add_model_parameter('percentage_infected', 0.1)

# threshold = 0.25
# for i in g.nodes():
#     config.add_node_configuration("threshold", i, threshold) # node attribute setting

# model.set_initial_status(config)