#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:48:21 2020

@author: giacomo
"""
import networkx as nx
# import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

# G = nx.DiGraph()
G = nx.Graph()

G.add_nodes_from(range(4))

G.add_edges_from([(0,1), (1,2), (1,3), (2,3)])

while G.size() < 200:
    edges = list(G.edges())
    shuffle(edges)
    l = G.size()
    
    new_nodes = list()
    new_edges = list()
    del_edges = list()
    
    for i,j in edges:
        for ii,k in edges:
            if i == ii & j != k:
                new_nodes.append(l)
                del_edges.append((i,j))
                new_edges.append((i,l))
                new_edges.append((j,l))
                new_edges.append((k,l))
                # l +=1
                break

                
    G.add_nodes_from(new_nodes)
    G.add_edges_from(new_edges)
    G.remove_edges_from(del_edges)

G.degseq = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    
# fig = plt.figure()
# ax = fig.add_subplot(121)
# nx.draw(G)
# ax = fig.add_subplot(122)
# nx.draw_shell(G)

fig = plt.figure()
ax = fig.add_subplot(121)
nx.draw(G, pos=nx.kamada_kawai_layout(G), ax=ax, node_size=80)
ax = fig.add_subplot(122)
ax.hist(G.degseq, density=True)


# plt.figure(figsize=(8, 8))
# nx.draw_networkx_edges(G, pos = nx.spectral_layout(G), alpha=0.4)
# nx.draw_networkx_nodes(
#     G,
#     pos = nx.spectral_layout(G),
#     node_size=80,
#     node_color= [d for n, d in list(G.degree())],
#     cmap=plt.cm.Reds_r,
# )

deg_list = [d for n, d in list(G.degree())]
colormap = plt.cm.Reds

plt.figure(figsize=(7,5.5))
nx.draw_networkx_edges(G, pos = nx.kamada_kawai_layout(G), alpha=0.4)
nx.draw_networkx_nodes(
    G,
    pos = nx.kamada_kawai_layout(G),
    node_size=80,
    node_color= deg_list,
    cmap=colormap,
    label='degree'
)
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin = min(deg_list), vmax=max(deg_list)))
plt.colorbar(sm)

# plt.figure()
# S=nx.star_graph(20)
# pos=nx.spring_layout(S)
# colors=range(20)
# cmap=plt.cm.Blues
# nx.draw(S, pos, node_color='#A0CBE2', edge_color=colors, width=4, edge_cmap=cmap,
#            with_labels=False, vmin=vmin, vmax=vmax)



