#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:30:22 2020

@author: giacomo
"""
import time
# import pickle
import networkx as nx

from jack import RandNemic

tic = time.perf_counter()
print('Execution started')

# with open('watts.pkl', 'rb') as f:
#     watts = pickle.load(f)

watts = RandNemic('Watts Strogatz',
                  nx.connected_watts_strogatz_graph(int(1e3), 50,
                                                    0.1, seed=1234),
                  'watts_1e3.pkl')
print('Small world graph loaded')

rando = RandNemic('Random reference',    # TAKES SOME TIME (!!)
                  nx.random_reference(watts.G),
                  'rando_ref_1e3.pkl')

toc = time.perf_counter()
print(f'Random reference found in {toc - tic:0.4f} seconds')

latti = RandNemic('Lattice reference',
                  nx.lattice_reference(watts.G),
                  'latti_ref_1e3.pkl')

tac = time.perf_counter()
print(f'Lattice reference found in {tac - toc:0.4f} seconds')
print(f'Total elapsed time: {tac - tic:0.4f} seconds')

rando.save()
latti.save()

print('Networks saved. All done.')
