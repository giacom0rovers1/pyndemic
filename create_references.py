#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:30:22 2020

@author: giacomo
"""
import time
import pickle
import networkx as nx

from jack import RandNemic

tic = time.perf_counter()
print('Execution started')

with open('watts.pkl', 'rb') as f:
    watts = pickle.load(f)

print('Small world graph loaded')

rando = RandNemic('Random reference',    # TAKES SOME TIME (> 150 h !!)
                  nx.random_reference(watts.G),
                  'rando_ref.pkl')

toc = time.perf_counter()
print(f'Random reference found in {toc - tic:0.4f} seconds')

latti = RandNemic('Lattice reference',
                  nx.lattice_reference(watts.G),
                  'latti_ref.pkl')

tac = time.perf_counter()
print(f'Lattice reference found in {tac - toc:0.4f} seconds')
print(f'Total elapsed time: {tac - tic:0.4f} seconds')

rando.save()
latti.save()

print('Networks saved. All done.')


# tic = time.perf_counter()
# BC = nx.betweenness_centrality(watts.G)
# toc = time.perf_counter()
# print(f'Random reference found in {toc - tic:0.4f} seconds')
