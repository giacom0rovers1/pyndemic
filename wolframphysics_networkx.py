#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:48:21 2020

@author: giacomo
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from scipy.stats import probplot  #poisson, norm, 

# G = nx.DiGraph()
G = nx.Graph()

G.add_nodes_from(range(4))

G.add_edges_from([(0,1), (1,2), (1,3), (2,3)])

run = 0
l = 0
while l < 100:  # keeps an eye on memory and visual repr.
# while run < 100:
    run +=1
    print('Run =',run)
    
    edges = list(G.edges())
    shuffle(edges)
    l = G.size()
    print('Size =', l)
    
    new_nodes = list()
    new_edges = list()
    del_edges = list()
    
    for i,j in edges:
        # print(i,j)
        for ii,k in edges:
            # print((i,j), (ii, k))
            if (i == ii and j != k):
                print('Match =', (i,j), (ii,k))
                
                new_nodes.append(l)
                del_edges.append((i,j))
                new_edges.append((i,l))
                new_edges.append((j,l))
                new_edges.append((k,l))
                l +=1
                # break
                # continue
    print('')
    G.add_nodes_from(new_nodes)
    G.add_edges_from(new_edges)
    G.remove_edges_from(del_edges)

G.degseq = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    
# fig = plt.figure()
# ax = fig.add_subplot(121)
# nx.draw(G)
# ax = fig.add_subplot(122)
# nx.draw_shell(G)

fig1 = plt.figure(figsize=(9,4))
ax = fig1.add_subplot(121)
nx.draw(G, pos=nx.kamada_kawai_layout(G), ax=ax, node_size=80)
ax = fig1.add_subplot(122)
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

degree_list = [d for n, d in list(G.degree())]
colormap = plt.cm.Reds

plt.figure(figsize=(7,5.5))
nx.draw_networkx_edges(G, pos = nx.kamada_kawai_layout(G), alpha=0.4)
nx.draw_networkx_nodes(
    G,
    pos = nx.kamada_kawai_layout(G),
    node_size=80,
    node_color= degree_list,
    cmap=colormap,
)
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin = min(degree_list), vmax=max(degree_list)))
plt.colorbar(sm)

# plt.figure()
# S=nx.star_graph(20)
# pos=nx.spring_layout(S)
# colors=range(20)
# cmap=plt.cm.Blues
# nx.draw(S, pos, node_color='#A0CBE2', edge_color=colors, width=4, edge_cmap=cmap,
#            with_labels=False, vmin=vmin, vmax=vmax)




# G = nx.gnp_random_graph(100, 0.02)
# Degree Rank plot from: 
# https://networkx.github.io/documentation/latest/auto_examples/drawing/plot_degree_rank.html#sphx-glr-auto-examples-drawing-plot-degree-rank-py

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
dmax = max(degree_sequence)

plt.figure(figsize=(7,5.5))
plt.loglog(degree_sequence, "b-", marker="o")
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")

# draw graph in inset
plt.axes([0.45, 0.45, 0.45, 0.45])
Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
pos = nx.spring_layout(Gcc)
plt.axis("off")
nx.draw_networkx_nodes(Gcc, pos, node_size=20)
nx.draw_networkx_edges(Gcc, pos, alpha=0.4)
plt.show()


fig2 = plt.figure(figsize=(9,4))
ax = fig2.add_subplot(121)
# ax.bar(range(dmax+1), nx.degree_histogram(G))
plt.loglog(degree_sequence, "b-", marker="o")
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")

ax = fig2.add_subplot(122)
probplot(degree_list, dist="norm", plot = plt)



# NOTABLE FUNCTIONS (spostare da un'altra parte)

nx.info(G)
nx.degree(G)
np.mean(G.degree())
nx.average_shortest_path_length(G)
nx.average_clustering(G)

BC = nx.betweenness_centrality(G)
Cl = nx.closeness_centrality(G)

nx.degree_centrality(G)                 # same as degree_list but normalized
# plt.scatter(degree_list, nx.degree_centrality(G).values())

nx.eigenvector_centrality(G)
# nx.percolation_centrality(G)   # returns an error
nx.global_efficiency(G)

nx.diameter(G)


# NOTABLE PLOTS

plt.scatter(Cl.values(), BC.values()) #, cmap=plt.cm.Blues, c = degree_list)
plt.xlabel("Closeness")
plt.ylabel("Betweenness")

plt.scatter(degree_list, BC.values())
plt.xlabel("Connectivity")
plt.ylabel("Betweenness")
