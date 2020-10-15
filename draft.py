#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 12:00:07 2020

@author: giacomo
"""

import numpy as np
# from scipy import stats, sparse, linalg #, integrate, interpolate
import networkx as nx
import salience_unw as sl
import matplotlib.pyplot as plt
# import collections
# import pandas as pd
# import seaborn as sns
from scipy.stats import poisson, norm, probplot
from scipy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds, eigs


measurements = np.random.normal(loc=20, scale=5, size=100)
probplot(measurements, dist="norm", plot=plt)
plt.show()


N = 1000
k = 3
p = 0.1
seed = 3

print(N, k, p, seed)


# %% First graph

# empty graph
# G = nx.Graph()

G = nx.petersen_graph()

plt.figure()
A = nx.adj_matrix(G)
A = A.asfptype()
plt.spy(A)

U, s, V = svds(A)
plt.hist(eigs(A)[0])

S = sl.salience(G)

for i, j in G.edges:
    print(S[i, j])
    G.edges[i, j]['salience'] = S[i, j]


ax = plt.figure()
ax.add_subplot(111)
ax.show()
nx.draw_shell(G, nlist=[[9, 5, 6, 7, 8], range(5)],
              with_labels=True,
              edge_color=[sal for i, j, sal in G.edges.data('salience')])

plt.show()


#
# plt.figure()
# plt.subplot(121)
# nx.draw_shell(G, with_labels=True)
# plt.title("Petersen")
# plt.subplot(122)
# nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True)
#
# list(G.degree)

# %% classic graphs
K = nx.complete_graph(10)

plt.figure()
plt.subplot(121)
nx.draw_shell(K, with_labels=True)
plt.title("Classic complete")

K_3_5 = nx.complete_bipartite_graph(3, 5)

plt.subplot(122)
nx.draw_shell(K_3_5, with_labels=True)
plt.title("Classic bipartite")


# %% random graph
er = nx.erdos_renyi_graph(N, p, seed)
er.degseq = sorted([d for n, d in er.degree()], reverse=True)
plt.hist(er.degseq, density=True)


ba = nx.barabasi_albert_graph(N, k, seed)
ba.degseq = sorted([d for n, d in ba.degree()], reverse=True)
plt.hist(ba.degseq, density=True, log=True)

# %%
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


# %% small-world random graph
ws = nx.watts_strogatz_graph(N, k, p, seed=3)
ws.degseq = sorted([d for n, d in ws.degree()], reverse=True)
# plt.hist(ws.degseq, density=True)
# figura di test
plt.figure()
# deg = list(dict(ws.degree).values())
# sns.distplot(deg, axlabel="Degree k")


mu, sigma = norm.fit(ws.degseq)
mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
x = np.linspace(poisson.ppf(0.01, mu),
                poisson.ppf(0.99, mu), num=100)
x_int = np.arange(poisson.ppf(0.01, mu),
                  poisson.ppf(0.99, mu))

plt.hist(ws.degseq,
         align="left",
         bins=x_int,
         color="lightsteelblue",
         label="Degree (k)",
         density=True)
plt.plot(x_int, poisson.pmf(x_int, mu), label="Poisson")
plt.plot(x, norm.pdf(x, mu, sigma), label="Gauss")
plt.legend(loc="best")

plt.figure()
plt.spy(nx.adj_matrix(ws))


# %% Salience (links)

S = sl.salience(ws)
ws.salseq = sorted(S.flatten(), reverse=True)
plt.hist(ws.salseq, log=True)

plt.figure()
plt.hist(eig(S)[0], log=True)

spS = csr_matrix(S)


# %%
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

# %%
wsr = nx.random_reference(ws)
wsl = nx.lattice_reference(ws)
ws.sigma = nx.sigma(ws)  # heavy  # small world if >1
ws.omega = nx.omega(ws)  # heavy
