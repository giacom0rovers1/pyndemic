#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:44:38 2020

@author: giacomo
"""
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.stats import probplot, norm, poisson
from scipy.optimize import curve_fit

# to avoid white dots on white bg
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



# all in one call, refined
def graph_tools(G):
    G.n =  len(list(G.nodes))
    G.l =  len(list(G.edges))

    k = G.degree()
    G.degree_list = [d for n, d in k]
    G.k_avg = np.mean(G.degree_list)
    G.k_std = np.std(G.degree_list)
    G.degree_sequence = sorted(G.degree_list, reverse=True)
    G.k_max = G.degree_sequence[0]
    G.k_min = G.degree_sequence[-1]
    G.k_histD = np.array(nx.degree_histogram(G))/G.n

    BC = nx.betweenness_centrality(G) #, normalized = False)
    G.BC_list = [bc for bc in BC.values()]

    # Cl = nx.closeness_centrality(G)
    # Cl_list = [cl for cl in Cl.values()]

    G.eig_list = nx.adjacency_spectrum(G)
    # eig_sequence = sorted(eig_list, reverse=True)

    G.A = nx.adj_matrix(G)
    G.A = G.A.asfptype()
    # plt.spy(A)
    G.eig_val, G.eig_vec = sp.sparse.linalg.eigs(G.A)


    C = nx.clustering(G)
    G.C_list = [c for c in C.values()]
    G.C_avg = nx.average_clustering(G)


    G.L = nx.average_shortest_path_length(G)

    return G


def scale_free(x, a, b):
    return a*x**-b


def graph_plots(G,  plots_to_print = [1,2,3,4], cmap = plt.cm.Blues):
    ''' provide 'plots_to_print' as a list '''
    colormap = truncate_colormap(cmap, 0.2, 0.8)
    plots_to_print = list(plots_to_print)

    if 1 in plots_to_print:
        ## Network visualization
        fig1 = plt.figure(figsize=(7,5.5))

        fig1.add_subplot(111)
        nx.draw_networkx_edges(G,pos = nx.kamada_kawai_layout(G), alpha=0.4)
        nx.draw_networkx_nodes(G,pos = nx.kamada_kawai_layout(G), node_size=80, node_color= G.degree_list, cmap=colormap)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin = min(G.degree_list), vmax=max(G.degree_list)))
        plt.colorbar(sm)


    if 2 in plots_to_print:
        ## Degree analysis
        fig2 = plt.figure(figsize=(11,12))
        x = np.linspace(0, G.k_max, 100)
        pars, cov = curve_fit(f=scale_free, xdata=np.arange(G.k_max+1), ydata=G.k_histD, p0=[1, 1], bounds=(-np.inf, np.inf))

        # Degree histogram
        fig2.add_subplot(221)
        # plt.hist(degree_list, density=True)
        plt.scatter(np.arange(G.k_max+1), G.k_histD, alpha=0.75)
        plt.xlabel('k')
        plt.ylabel('p(k)')
        plt.title(r'$< k > =$' + str(np.round(G.k_avg, 2)))
        plt.plot(x, norm.pdf(x,G.k_avg, G.k_std), 'r--', label = 'Normal')
        plt.plot(np.arange(G.k_max+1), poisson.pmf(np.arange(G.k_max+1), mu = G.k_avg), 'y--', label = 'Poisson')
        plt.plot(x, scale_free(x, *pars), 'b--', label = 'Scale free')
        plt.ylim([0,max(G.k_histD+0.1)])
        plt.legend(loc = 'best')

        fig2.add_subplot(222)
        # plt.hist(degree_list, density=True)
        plt.loglog(np.arange(G.k_max+1), G.k_histD, '.', ms=12, alpha=0.75)
        plt.xlabel('k')
        plt.ylabel('p(k)')

        plt.loglog(x, norm.pdf(x,G.k_avg, G.k_std), 'r--', label = 'Normal')
        plt.loglog(np.arange(G.k_max+1), poisson.pmf(np.arange(G.k_max+1), mu = G.k_avg), 'y--', label = 'Poisson')
        plt.loglog(x, scale_free(x, *pars), 'b--', label = 'Scale free')

        plt.legend(loc = 'best')
        plt.xlim([1,G.k_max+1])
        # TODO plot distribuzioni di confronto (power law. gauss, poisson)

        # Dergee rank plot
        fig2.add_subplot(223)
        plt.loglog(G.degree_sequence, "b-", marker="o")
        plt.title("Degree rank plot")
        plt.ylabel("Degree")
        plt.xlabel("Rank")

        # Degree probability plot
        fig2.add_subplot(224)
        probplot(G.degree_list, dist="norm", plot = plt)


    if 3 in plots_to_print:
        ## More insight
        fig3 = plt.figure(figsize=(11,5))

        # BC vs K
        fig3.add_subplot(121)
        plt.scatter(G.degree_list, G.BC_list, alpha=0.75)
        plt.title('Average Shortest-path length = ' + str(round(G.L, 2)))
        plt.xlabel("Connectivity degree k")
        plt.ylabel("Betweenness centrality BC")
        plt.xlim(0.5, G.k_max+0.5)

        # x = np.linspace(0,dmax+1,100)
        # y = x**2
        # plt.plot(x,y,'r')

        # Clustering
        fig3.add_subplot(122)
        plt.hist(G.C_list, density=True, alpha=0.75)
        plt.xlabel("Clustering coefficient")
        plt.ylabel("Density")
        plt.title('Average clustering coeff =' + str(np.round(G.C_avg, 2)))

        # Closeness

        # plt.scatter(Cl_list, BC_list, cmap = colormap, c = degree_list)
        # plt.xlabel("Closeness")
        # plt.ylabel("Betweenness")
        # sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin = min(degree_list), vmax=max(degree_list)))
        # plt.colorbar(sm)


    if 4 in plots_to_print:
        # Adjacency spectrum and base vectors
        fig4 = plt.figure(figsize=(11,9))

        # Spy plot
        fig4.add_subplot(221)
        plt.spy(G.A, markersize=3)
        plt.title('A')

        # Spectrum (eigenvalues histogram)
        fig4.add_subplot(222)
        plt.hist(G.eig_list, density=True, alpha=0.75)
        plt.xlabel(r"Eigenvalue $\lambda$")
        plt.ylabel("Density")
        plt.title("Adjacency spectrum")

        # Eigenvector functions
        fig4.add_subplot(212)
        for i in range(G.eig_val.size):
            plt.plot(G.eig_vec[:, i], alpha=0.5,
                     label=r'$\lambda =$' + str(np.round(G.eig_val.real, 2)[i]))
        plt.legend(loc='best')



    #TODO Laplacian matrix and spectrum



    return 0


