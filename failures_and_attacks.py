#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:30:02 2020

@author: giacomo
"""
import random
import matplotlib.pyplot as plt
import networkx as nx


def save_graph(graph, pos, file_name):
    # initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.savefig(file_name, bbox_inches="tight")
    plt.close()
    del fig


def attack(graph, centrality_metric):
    graph = graph.copy()
    steps = 0
    ranks = centrality_metric(graph)
    nodes = sorted(graph.nodes(), key=lambda n: ranks[n])

    # Generate spring layout
    pos = nx.spring_layout(graph)
    while nx.is_connected(graph):
        graph.remove_node(nodes.pop())
        file_name = './attack/'+str(steps)+'.png'
        save_graph(graph, pos, file_name)
        steps += 1
    else:
        return steps


def failure(graph):
    graph = graph.copy()
    steps = 0

    while nx.is_connected(graph):
        node = random.choice(list(graph.nodes()))
        graph.remove_node(node)
        steps += 1
    else:
        return steps


NETWORK_SIZE = 1000
print('Creating powerlaw cluster with %d Nodes.' % NETWORK_SIZE)
K = 4
P = 0.1
HK = nx.powerlaw_cluster_graph(NETWORK_SIZE, K, 0.1)


print('Starting attacks...')

print('Network with  Scale-free Model broke after',
      failure(HK),
      'steps with random failures.')

print('Network with Scale-free Model broke after',
      attack(HK, nx.betweenness_centrality),
      'steps with Targeted Attacks.')
