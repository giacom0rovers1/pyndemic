#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:44:38 2020

@author: giacomo
"""
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.stats import probplot, norm, poisson
from scipy.optimize import curve_fit

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc

from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend

# SUPPORT to APPEARENCE


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    To avoid white dots on white background


    Parameters
    ----------
    cmap : TYPE
        DESCRIPTION.
    minval : TYPE, optional
        DESCRIPTION. The default is 0.0.
    maxval : TYPE, optional
        DESCRIPTION. The default is 1.0.
    n : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    new_cmap : TYPE
        DESCRIPTION.

    '''
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


# BASIC FUNCTIONS

def scale_free(x, a, b):
    return a*(x)**-b


def exponential(x, a, b, c):
    return a*np.exp(b*(x+c))


# COMPLEX NETWORKS ANALYSIS AND VISUALIZATION

def graph_tools(G):
    G.nn = len(list(G.nodes))
    G.nl = len(list(G.edges))

    k = G.degree()
    G.degree_list = [d for n, d in k]
    G.k_avg = np.mean(G.degree_list)
    G.k_std = np.std(G.degree_list)
    G.degree_sequence = sorted(G.degree_list, reverse=True)
    G.k_max = G.degree_sequence[0]
    G.k_min = G.degree_sequence[-1]
    G.k_histD = np.array(nx.degree_histogram(G))/G.nn

    BC = nx.betweenness_centrality(G)  # , normalized = False)
    G.BC_list = [bc for bc in BC.values()]

    # Cl = nx.closeness_centrality(G)
    # Cl_list = [cl for cl in Cl.values()]

    G.eig_list = nx.adjacency_spectrum(G)
    # eig_sequence = sorted(eig_list, reverse=True)

    G.A = nx.adj_matrix(G)  # create a sparse adjacency matrix
    G.A = G.A.asfptype()  # convert the sparse values from int to float
    # plt.spy(A)
    G.eig_val, G.eig_vec = sp.sparse.linalg.eigs(G.A)

    C = nx.clustering(G)
    G.C_list = [c for c in C.values()]
    G.C_avg = nx.average_clustering(G)

    G.L = nx.average_shortest_path_length(G)

    return G


def graph_plots(G,  plots_to_print=[1, 2, 3, 4], cmap=plt.cm.Blues):
    ''' provide 'plots_to_print' as a list '''
    colormap = truncate_colormap(cmap, 0.2, 0.8)
    plots_to_print = list(plots_to_print)

    if 1 in plots_to_print:
        # Network visualization
        fig1 = plt.figure(figsize=(7, 5.5))

        fig1.add_subplot(111)
        nx.draw_networkx_edges(G, pos=nx.kamada_kawai_layout(G),
                               alpha=0.4)
        nx.draw_networkx_nodes(G, pos=nx.kamada_kawai_layout(G),
                               node_size=80,
                               node_color=G.degree_list, cmap=colormap)

        sm = plt.cm.ScalarMappable(cmap=colormap,
                                   norm=plt.Normalize(vmin=min(G.degree_list),
                                                      vmax=max(G.degree_list)))
        plt.colorbar(sm)

    if 2 in plots_to_print:
        # Degree analysis
        fig2 = plt.figure(figsize=(11, 12))
        x = np.linspace(0, G.k_max, 100)
        xi = np.arange(G.k_min, G.k_max+1)
        yi = G.k_histD[G.k_min:]

        G.gauss = norm.pdf(x, G.k_avg, G.k_std)
        G.poiss = poisson.pmf(xi, mu=G.k_avg)
        G.sf_pars, G.sf_cov = curve_fit(f=scale_free,
                                        xdata=xi,
                                        ydata=yi,
                                        p0=[1, 1],
                                        bounds=(1e-10, np.inf))

        # Degree histogram
        fig2.add_subplot(221)
        # plt.hist(degree_list, density=True)
        plt.scatter(xi, yi, alpha=0.75)
        plt.xlabel('k')
        plt.ylabel('p(k)')
        plt.title(r'$< k > =$' + str(np.round(G.k_avg, 2)))
        plt.plot(x, G.gauss, 'r--', label='Normal')
        plt.plot(xi, G.poiss, 'y--', label='Poisson')
        plt.plot(x, scale_free(x, *G.sf_pars), 'b--', label='Scale free')
        plt.ylim([0, max(yi+0.1)])
        plt.legend(loc='best')

        fig2.add_subplot(222)
        # plt.hist(degree_list, density=True)
        plt.loglog(xi, yi, '.', ms=12, alpha=0.75)
        plt.xlabel('k')
        plt.ylabel('p(k)')

        plt.loglog(x, G.gauss, 'r--', label='Normal')
        plt.loglog(xi, G.poiss, 'y--', label='Poisson')
        plt.loglog(x, scale_free(x, *G.sf_pars), 'b--', label='Scale free')

        plt.legend(loc='best')
        plt.xlim([1, G.k_max+1])

        # Dergee rank plot
        fig2.add_subplot(223)
        plt.loglog(G.degree_sequence, "b-", marker="o")
        plt.title("Degree rank plot")
        plt.ylabel("Degree")
        plt.xlabel("Rank")

        # Degree probability plot
        fig2.add_subplot(224)
        probplot(G.degree_list, dist="norm", plot=plt)

    if 3 in plots_to_print:
        # More insight
        fig3 = plt.figure(figsize=(11, 5))

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
        # sm = plt.cm.ScalarMappable(cmap=colormap,
        #                            norm=plt.Normalize(vmin=min(degree_list),
        #                                               vmax=max(degree_list)))
        # plt.colorbar(sm)

    if 4 in plots_to_print:
        # Adjacency spectrum and base vectors
        fig4 = plt.figure(figsize=(11, 9))

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
            lab = r'$\lambda =$' + str(np.round(G.eig_val.real, 2)[i])
            plt.plot(G.eig_vec[:, i], alpha=0.5,
                     label=lab)
        plt.legend(loc='best')

    # TODO Laplacian matrix and spectrum

    return 0


# EPIDEMIC PROCESSES


def SEIR(perc_inf, beta, tau_i, tau_r, days):
    '''
    SEIR Epidemic Model

    Parameters
    ----------
    perc_inf : float
        Initially infected percentage of the population [0, 100].
    beta : float
        Probability of transmission in one day [0, 1].
    tau_i : int
        average incubation time [days].
    tau_r : int
        average recovery time [days].
    days : int
        Total number of simulated days.

    Returns
    -------
    s, e, i, r : floats
        Relative populations
    t : float
        Time array

    '''
    # calculate constants
    frac_inf = perc_inf/100
    gamma = 1/tau_i
    mu = 1/tau_r
    R0 = beta/mu

    y0 = np.array([(1-frac_inf), 0, frac_inf, 0])
    y = y0

    def dydt(t, y):
        return np.array([-beta*y[0]*y[2],                   # ds/dt
                         beta*y[0]*y[2] - gamma*y[1],       # de/dt
                         gamma*y[1] - mu*y[2],              # di/dt
                         mu*y[2]])                          # dr/dt

    y = sp.integrate.solve_ivp(fun=dydt, t_span=(0, days), y0=y0)

    plt.figure()
    plt.plot(y.t, y.y.T)
    # plt.legend(["s", "e", "i", "r"])
    plt.legend(["Susceptible", "Exposed", "Infected", "Removed"])
    plt.text(0.8*days, 0.9, r'$R_{0}$ ='+str(np.round(R0, 2)))
    plt.xlabel('Days')
    plt.ylabel('Relative population')

    s, e, i, r = [y.y[line, :] for line in range(4)]
    t = y.t
    return s, e, i, r, t


def growth_fit(pos, tau_r):
    # Locates and analyses the first phase of an epidemic spreading

    # Growth flex
    incr = np.gradient(np.gradient(pos))

    idx = tau_r
    while incr[idx] > 0:
        idx += 1

    # formulazione semplice: mi fermo al flesso
    growth = pos[:idx]

    xi = np.arange(0, len(growth))
    yi = growth

    pars, cov = curve_fit(f=exponential,
                          xdata=xi,
                          ydata=yi,
                          bounds=(-100, 100))
    K0 = pars[1]
    Td0 = np.log(2)/K0

    x = np.linspace(0, 1.3*len(growth), 100)

    plt.figure()
    plt.plot(pos, label="Positives (E+I)")
    plt.plot(x, exponential(x, *pars), 'g--', label="Exponential fit")
    plt.xlim([0, 2*len(growth)])
    plt.ylim([0, 1.3*max(pos)])
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.text(1, max(pos),
             r'$K$ =' + str(np.round(K0, 2)) +
             r'; $T_{d}$ =' + str(np.round(Td0, 2)))
    plt.legend(loc='best')

    return xi, pars, K0, Td0


def contagion_metrics(s, e, i, r, R0, tau_i, tau_r, N):
    '''
    Calculates the reproduction number in time, the growth factor and the
    doubling time. Produces two graphs: the intial exponential growth and the
    variation of Rt throughout the epidemic evolution.

    Parameters
    ----------
    s : float
        Susceptible fraction.
    e : float
        Exposed fraction.
    i : float
        Infected (contagious) fraction.
    r : float
        Recovered/removed fraction.
    R0 : float
        Basic reproduction number.
    tau_i : int
        Incubation time scale.
    tau_r : int
        Recovery time scale.
    N : int
        Total number of individuals of the population.

    Returns
    -------
    Td : np.array
        Doubling time.
    R : np.array
        Reproduction number.
    rt : np.array
        DESCRIPTION.
    rts : np.array
        DESCRIPTION.
    Rt : np.array
        DESCRIPTION.
    Ki : np.array
        DESCRIPTION.
    Ks : np.array
        DESCRIPTION.

    '''
    # total positives (exposed + infected)
    pos = N * (e+i)

    # a priori Rt
    Rt = R0 * s

    # actual reproduction number (from increments of positives)
    rt = pos[1:]/pos[:-1]

    # actual growth rate
    Ki = np.diff(np.log(pos))

    # smoothing based on "tau_r"
    f = 0.5               # fraction of tau_r to use as smoothing window
    D = int(f*(tau_r))    # time interval of the measurements in cycles units

    rts = pd.Series(list(rt)).rolling(window=D,
                                      min_periods=D,
                                      center=True).mean().values

    Ks = pd.Series(list(Ki)).rolling(window=D,
                                     min_periods=D,
                                     center=True).mean().values

    # Reproduction number from smoothed growing rate
    R = np.exp(Ks)

    # Doubling time from the smoothed growing rate
    Td = np.log(2)/Ks

    # Es = pd.Series(list(N*e)).rolling(window=D,
    #                                   min_periods=1,
    #                                   center=True).mean().values
    # xi, pars = growth_fit(Es, tau_r)

    # Initial exponential growth
    xi, pars, K0, Td0 = growth_fit(pos, tau_r)

    plt.figure()
    plt.plot([1 for i in range(500)], 'k--', linewidth=0.7)
    plt.plot(Rt, 'b', label='Rt predicted')
    plt.plot(rt, 'orange', label='Rt from actual increments')
    plt.plot(rts, 'r', alpha=0.5, label='Rt moving average')
    plt.plot(R, 'grey', alpha=0.5, label='Rt derived from Ks')
    plt.plot(xi, np.ones(len(xi)) * np.exp(pars[1]), 'g--',
             label='R0 from the exponential fit')

    plt.legend(loc='best')
    plt.xlim([0, len(rts[rts > 0]) + 2*D])
    plt.ylim([0, int(R0+1)])
    # plt.grid()

    return Td0, Td, R, rt, rts, Rt, K0, Ki, Ks


def SEIR_network(G, N, perc_inf, beta, tau_i, tau_r, days, t):
    # G = jk.graph_tools(G)

    # Alternativa a grah_tools(G) per il solo degree
    k = G.degree()
    G.degree_list = [d for n, d in k]
    G.k_avg = np.mean(G.degree_list)

    print(G.k_avg)
    # # GRAPH PLOTS
    # jk.graph_plots(G, [1])
    # jk.graph_plots(G, [2, 3])
    # print(G.k_avg, G.k_min, G.sf_pars)

    # EPIDEMIC MODEL
    frac_inf = perc_inf/100
    beta_n = beta/G.k_avg  # infection rate
    gamma = 1/tau_i
    mu = 1/tau_r

    # Config:
    model = ep.SEIRModel(G)

    # print(model.parameters)
    # print(model.available_statuses)

    config = mc.Configuration()
    config.add_model_parameter('alpha', gamma)
    config.add_model_parameter('beta',  beta_n)
    config.add_model_parameter('gamma', mu)
    config.add_model_parameter("percentage_infected", frac_inf)

    model.set_initial_status(config)

    # Run:
    iterations = model.iteration_bunch(days, node_status=True)
    trends = model.build_trends(iterations)

    # Recover status variables:
    s = np.array([S for S, E, I, R in
                  [list(it['node_count'].values()) for it in iterations]])/N
    e = np.array([E for S, E, I, R in
                  [list(it['node_count'].values()) for it in iterations]])/N
    i = np.array([I for S, E, I, R in
                  [list(it['node_count'].values()) for it in iterations]])/N
    r = np.array([R for S, E, I, R in
                  [list(it['node_count'].values()) for it in iterations]])/N

    # resampling through t (variable spacing decided by the ODE solver)
    s = np.interp(t, np.arange(0, len(s)), s)
    e = np.interp(t, np.arange(0, len(e)), e)
    i = np.interp(t, np.arange(0, len(i)), i)
    r = np.interp(t, np.arange(0, len(r)), r)

    # Plot:
    viz = DiffusionTrend(model, trends)
    viz.plot()
    return s, e, i, r


class RandNemic:

    def __init__(self, name, graph, location):
        self.name = name
        self.G = graph
        self.loc = location
        self.N = self.G.number_of_nodes()

    def save(self):
        with open(self.loc, 'wb') as f:
            pickle.dump(self, f)

    def run(self, perc_inf, beta, tau_i, tau_r, days, t):

        self.s, self.e, self.i, self.r = SEIR_network(self.G, self.N,
                                                      perc_inf, beta,
                                                      tau_i, tau_r,
                                                      days, t)

        # self.s, self.e, self.i, self.r = s, e, i, r

        contagion_metrics(self.s, self.e,
                          self.i, self.r,
                          beta*tau_r, tau_i,
                          tau_r, self.N)
