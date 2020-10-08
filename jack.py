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

from scipy.stats import probplot


# to avoid white dots on white bg
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# all in one call, refined
def graph_plots(G, cmap = plt.cm.Blues):
    colormap = truncate_colormap(cmap, 0.2, 0.8)
    
    K = G.degree()
    degree_list = [d for n, d in K]
    degree_sequence = sorted(degree_list, reverse=True)
    
    BC = nx.betweenness_centrality(G) #, normalized = False)    
    BC_list = [bc for bc in BC.values()]
    
    # Cl = nx.closeness_centrality(G)
    # Cl_list = [cl for cl in Cl.values()]
    
    eig_list = nx.adjacency_spectrum(G)
    # eig_sequence = sorted(eig_list, reverse=True)
    
    A = nx.adj_matrix(G)
    A = A.asfptype()
    # plt.spy(A)
    values, vectors = sp.sparse.linalg.eigs(A) 
    
    
    C = nx.clustering(G)
    C_list = [c for c in C.values()]
    C_avg = nx.average_clustering(G)

       
    L = nx.average_shortest_path_length(G)

    
    
    ## Graph visualization
    fig1 = plt.figure(figsize=(7,5.5))
    
    fig1.add_subplot(111)
    nx.draw_networkx_edges(G,pos = nx.kamada_kawai_layout(G), alpha=0.4)
    nx.draw_networkx_nodes(G,pos = nx.kamada_kawai_layout(G), node_size=80, node_color= degree_list, cmap=colormap)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin = min(degree_list), vmax=max(degree_list)))
    plt.colorbar(sm)

    
    ## Degree analysis
    fig2 = plt.figure(figsize=(11,12))
    
    # Degree histogram
    fig2.add_subplot(221)
    # plt.hist(degree_list, density=True)
    plt.scatter(range(degree_sequence[0]+1), nx.degree_histogram(G), alpha=0.75)
    plt.xlabel('k')
    plt.ylabel('p(k)')    
    plt.title(r'$< k > =$' + str(np.round(np.mean(degree_list), 2)))

    fig2.add_subplot(222)
    # plt.hist(degree_list, density=True)
    plt.loglog(range(degree_sequence[0]+1), nx.degree_histogram(G), '.', ms=12, alpha=0.75)
    plt.xlabel('k')
    plt.ylabel('p(k)')    
    
    #TODO plot distribuzioni di confronto (power law. gauss, poisson)
    

    # Dergee rank plot
    fig2.add_subplot(223)
    plt.loglog(degree_sequence, "b-", marker="o")       
    plt.title("Degree rank plot")
    plt.ylabel("Degree")
    plt.xlabel("Rank")
    
    # Degree probability plot
    fig2.add_subplot(224)
    probplot(degree_list, dist="norm", plot = plt)
    
    
    ## More insight
    fig3 = plt.figure(figsize=(11,5))
    
    # BC vs K
    fig3.add_subplot(121)
    plt.scatter(degree_list, BC_list, alpha=0.75)
    plt.title('Average Shortest-path length = ' + str(round(L, 2)))
    plt.xlabel("Connectivity degree k")
    plt.ylabel("Betweenness centrality BC")
    plt.xlim(0.5, degree_sequence[0]+0.5)
    
    # x = np.linspace(0,dmax+1,100)
    # y = x**2
    # plt.plot(x,y,'r')
    
    # Clustering
    fig3.add_subplot(122)
    plt.hist(C_list, density=True, alpha=0.75)
    plt.xlabel("Clustering coefficient")
    plt.ylabel("Density")
    plt.title('Average clustering coeff =' + str(np.round(C_avg, 2)))
    
    # Closeness
    
    # plt.scatter(Cl_list, BC_list, cmap = colormap, c = degree_list)
    # plt.xlabel("Closeness")
    # plt.ylabel("Betweenness")
    # sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin = min(degree_list), vmax=max(degree_list)))
    # plt.colorbar(sm)
    
    
    # Adjacency spectrum and base vectors
    fig4 = plt.figure(figsize=(11,9))
    
    # Spy plot
    fig4.add_subplot(221)
    plt.spy(A, markersize=3)
    plt.title('A')
    
    # Spectrum (eigenvalues histogram)
    fig4.add_subplot(222)
    plt.hist(eig_list, density=True, alpha=0.75)
    plt.xlabel(r"Eigenvalue $\lambda$")
    plt.ylabel("Density")
    plt.title("Adjacency spectrum")
    
    # Eigenvector functions
    fig4.add_subplot(212)
    for i in range(values.size):
        plt.plot(vectors[:,i], alpha = 0.5, label = r'$\lambda =$' + str(np.round_(values.real, 2)[i]))
    plt.legend(loc = 'best')    



    #TODO Laplacian matrix and its spectrum



    return 0


