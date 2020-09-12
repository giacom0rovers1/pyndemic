#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 12:00:07 2020

@author: giacomo
"""
#import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

n = 10
k = 4
p = 0.15



# empty graph
#G = nx.Graph()
#G = nx.petersen_graph()
#
#plt.figure()
#plt.subplot(121)
#nx.draw_shell(G, with_labels=True)
#plt.title("Petersen")
#plt.subplot(122)
#nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True)
#
#list(G.degree)

#%% classic graphs
K_5 = nx.complete_graph(n)

plt.figure()
plt.subplot(121)
nx.draw_shell(K_5, with_labels=True)
plt.title("Classic complete")

K_3_5 = nx.complete_bipartite_graph(n, k)

plt.subplot(122)
nx.draw_shell(K_3_5, with_labels=True)
plt.title("Classic bipartite")


#%% random graph
er = nx.erdos_renyi_graph(n, p)

plt.figure()
plt.subplot(121)
nx.draw(er, with_labels=True)
plt.title("Erdos-Renyi")
plt.subplot(122)
nx.draw_shell(er, with_labels=True) 
dict(nx.clustering(er))
list(nx.connected_components(er))

plt.figure()
plt.scatter(dict(er.degree).values(), dict(nx.clustering(er)).values())
plt.xlabel("Degree")
plt.ylabel("Clustering coeff.")


ba = nx.barabasi_albert_graph(n, k)

plt.figure()
plt.subplot(121)
nx.draw(ba, with_labels=True)
plt.title("Barabasi-Albert")
plt.subplot(122)
nx.draw_shell(ba, with_labels=True) 
dict(nx.clustering(ba))
list(nx.connected_components(ba))

plt.figure()
plt.scatter(dict(ba.degree).values(), dict(nx.clustering(ba)).values())
plt.xlabel("Degree")
plt.ylabel("Clustering coeff.")


#%% small-world random graph
ws = nx.watts_strogatz_graph( n, k, p)

plt.figure()
plt.subplot(121)
nx.draw(ws, with_labels=True)
plt.title("Small word")
plt.subplot(122)
nx.draw_shell(ws, with_labels=True) 
dict(nx.clustering(ws))
list(nx.connected_components(ws))

plt.figure()
plt.scatter(dict(ws.degree).values(), dict(nx.clustering(ws)).values())
plt.xlabel("Degree")
plt.ylabel("Clustering coeff.")


#nx.random_reference()
#nx.lattice_reference()
#nx.sigma(ws)  # heavy  # small world if >1
#nx.omega(ws)  # heavy


