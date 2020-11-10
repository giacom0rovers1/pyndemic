#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:30:22 2020

@author: giacomo
"""
import pickle
import networkx as nx

from jack import RandNemic

with open('watts.pkl', 'rb') as f:
    watts = pickle.load(f)

rando = RandNemic('Random reference',               # TAKES SOME TIME (!!)
                  nx.random_reference(watts.G),
                  'rando_ref.pkl')

latti = RandNemic('Lattice reference',
                  nx.lattice_reference(watts.G),
                  'latti_ref.pkl')

rando.save()
latti.save()
