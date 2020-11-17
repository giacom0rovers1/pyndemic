#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:07:44 2020

@author: Giacomo Roversi
"""

import pickle


with open('pickle/random.pkl', 'rb') as f:
    rando = pickle.load(f)
with open('pickle/lattice.pkl', 'rb') as f:
    latti = pickle.load(f)
with open('pickle/smallw.pkl', 'rb') as f:
    watts = pickle.load(f)
with open('pickle/scalefree.pkl', 'rb') as f:
    barab = pickle.load(f)
with open('pickle/realw.pkl', 'rb') as f:
    holme = pickle.load(f)

with open('pickle/all_networks.pkl', 'wb') as f:
    pickle.dump([watts, rando, latti, barab, holme], f)
