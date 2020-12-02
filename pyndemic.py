#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:44:38 2020

@author: giacomo
"""
import time
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

# from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend

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


# def exponential(x, b):
#     return 10*np.exp(b*x)


def exponential(x, a, b):
    return b*np.exp(a*x)


# def exponential(x, a, b, c):
#     return a*np.exp(b*(x+c))


# COMPLEX NETWORKS ANALYSIS AND VISUALIZATION

def graph_tools(G):
    tic = time.perf_counter()
    print("Graph analysis started...")

    G.nn = len(list(G.nodes))
    G.nl = len(list(G.edges))

    k = G.degree()
    G.degree_list = [d for n, d in k]
    G.k_avg = np.mean(G.degree_list)
    G.k_std = np.std(G.degree_list)
    G.degree_sequence = sorted(G.degree_list, reverse=True)
    G.k_max = np.max(G.degree_list)  # G.degree_sequence[0]
    G.k_min = np.min(G.degree_list)  # G.degree_sequence[-1]
    G.k_histD = np.array(nx.degree_histogram(G))/G.nn
    print("Connectivity degree histogram completed.")

    BC = nx.betweenness_centrality(G)  # , normalized = False)
    G.BC_list = [bc for bc in BC.values()]
    print("Betweenness centrality list completed.")

    # Cl = nx.closeness_centrality(G)
    # Cl_list = [cl for cl in Cl.values()]

    # G.eig_list = nx.adjacency_spectrum(G)
    # # eig_sequence = sorted(eig_list, reverse=True)

    # G.A = nx.adj_matrix(G)  # create a sparse adjacency matrix
    # G.A = G.A.asfptype()  # convert the sparse values from int to float
    # # plt.spy(A)
    # G.eig_val, G.eig_vec = sp.sparse.linalg.eigs(G.A)

    C = nx.clustering(G)
    G.C_list = [c for c in C.values()]
    G.C_avg = nx.average_clustering(G)
    print("Clustering list completed.")

    # G.L = nx.average_shortest_path_length(G)
    toc = time.perf_counter()
    print(f'--> graph_tools() completed in {toc - tic:0.0f} seconds.\n')

    return G


def graph_plots(G,  net_name, plots_to_print=[0, 1], cmap=plt.cm.Blues):
    ''' provide 'plots_to_print' as a list '''
    colormap = truncate_colormap(cmap, 0.2, 0.8)
    plots_to_print = list(plots_to_print)

    if 0 in plots_to_print:
        # Network visualization
        G.fig0 = plt.figure(figsize=(7, 5.5))

        G.fig0.add_subplot(111)
        nx.draw_networkx_edges(G, pos=nx.kamada_kawai_layout(G),
                               alpha=0.4)
        nx.draw_networkx_nodes(G, pos=nx.kamada_kawai_layout(G),
                               node_size=80,
                               node_color=G.degree_list, cmap=colormap)

        sm = plt.cm.ScalarMappable(cmap=colormap,
                                   norm=plt.Normalize(vmin=min(G.degree_list),
                                                      vmax=max(G.degree_list)))
        # cbar = plt.colorbar(sm)
        plt.colorbar(sm, label='Node connectivity degree k')
        # cbar.ax.get_yaxis().labelpad = 15
        # cbar.ax.set_ylabel("Node connectivity degree", rotation=270, )
        plt.title(net_name + " - 1:100 scale model")
        plt.tight_layout()

    if 1 in plots_to_print:
        # Degree and BC analysis
        G.fig1 = plt.figure(figsize=(11, 5))
        # Axes
        x = np.linspace(0, G.k_max, 100)
        if net_name == "Holme-Kim":
            xi = np.arange((G.k_min+1), (G.k_max+1))
            yi = G.k_histD[(G.k_min+1):]
        else:
            xi = np.arange((G.k_min), (G.k_max+1))
            yi = G.k_histD[(G.k_min):]

        # Fits
        G.gauss = norm.pdf(x, G.k_avg, G.k_std)
        G.poiss = poisson.pmf(xi, mu=G.k_avg)

        if len(yi) > 0:
            G.sf_pars, G.sf_cov = curve_fit(f=scale_free,
                                            xdata=xi,
                                            ydata=yi,
                                            p0=[1, 1],
                                            bounds=(1e-10, np.inf))
        else:
            G.sf_pars = [0, -1]

        # Degree histogram
        G.fig1.add_subplot(121)
        plt.scatter(xi, yi, alpha=0.75)
        plt.xlabel('Node connectivity degree k')
        plt.ylabel('Degree distribution p(k)')
        plt.title('Average connectivity degree = ' + str(np.round(G.k_avg, 1)))
        plt.plot(x, G.gauss, 'r--', alpha=0.5, label='Normal distr.')
        # plt.plot(xi, G.poiss, 'y--', label='Poisson')
        plt.plot(x, scale_free(x, *G.sf_pars), 'b--',
                 alpha=0.5, label='Power law')
        plt.ylim([0, max(yi+0.1)])
        plt.legend(loc='best')

        # BC vs K
        G.fig1.add_subplot(122)
        plt.scatter(G.degree_list, G.BC_list, alpha=0.75)
        plt.title('Average clustering coeff. = ' + str(np.round(G.C_avg, 3)))
        plt.xlabel("Node connectivity degree k")
        plt.ylabel("Node betweenness centrality BC")
        plt.xlim(0.5, G.k_max+0.5)
        plt.tight_layout()

    if 2 in plots_to_print:
        # Degree analysis
        G.fig2 = plt.figure(figsize=(11, 12))
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
        G.fig2.add_subplot(221)
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

        G.fig2.add_subplot(222)
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
        G.fig2.add_subplot(223)
        plt.loglog(G.degree_sequence, "b-", marker="o")
        plt.title("Degree rank plot")
        plt.ylabel("Degree")
        plt.xlabel("Rank")

        # Degree probability plot
        G.fig2.add_subplot(224)
        probplot(G.degree_list, dist="norm", plot=plt)
        plt.tight_layout()

    if 3 in plots_to_print:
        # More insight
        G.fig3 = plt.figure(figsize=(11, 5))

        # BC vs K
        G.fig3.add_subplot(121)
        plt.scatter(G.degree_list, G.BC_list, alpha=0.75)
        plt.title('Average Shortest-path length = ' + str(round(G.L, 2)))
        plt.xlabel("Connectivity degree k")
        plt.ylabel("Betweenness centrality BC")
        plt.xlim(0.5, G.k_max+0.5)

        # x = np.linspace(0,dmax+1,100)
        # y = x**2
        # plt.plot(x,y,'r')

        # Clustering
        G.fig3.add_subplot(122)
        plt.hist(G.C_list, density=True, alpha=0.75)
        plt.xlabel("Clustering coefficient")
        plt.ylabel("Density")
        plt.title('Average clustering coeff = ' + str(np.round(G.C_avg, 2)))
        plt.tight_layout()
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
        G.fig4 = plt.figure(figsize=(11, 9))

        # Spy plot
        G.fig4.add_subplot(221)
        plt.spy(G.A, markersize=3)
        plt.title('A')

        # Spectrum (eigenvalues histogram)
        G.fig4.add_subplot(222)
        plt.hist(G.eig_list, density=True, alpha=0.75)
        plt.xlabel(r"Eigenvalue $\lambda$")
        plt.ylabel("Density")
        plt.title("Adjacency spectrum")

        # Eigenvector functions
        G.fig4.add_subplot(212)
        for i in range(G.eig_val.size):
            lab = r'$\lambda =$' + str(np.round(G.eig_val.real, 2)[i])
            plt.plot(G.eig_vec[:, i], alpha=0.5,
                     label=lab)
        plt.legend(loc='best')
        plt.tight_layout()

    # TODO Laplacian matrix and spectrum

    return G


# EPIDEMIC PROCESSES


def SEIR_odet(perc_inf, beta, tau_i, tau_r, days, title):
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

    y0 = np.array([(1-frac_inf), frac_inf*(1-beta), frac_inf*beta, 0])
    y = y0

    def dydt(t, y):
        return np.array([-beta*y[0]*y[2],                   # ds/dt
                         beta*y[0]*y[2] - gamma*y[1],       # de/dt
                         gamma*y[1] - mu*y[2],              # di/dt
                         mu*y[2]])                          # dr/dt

    y = sp.integrate.solve_ivp(fun=dydt,
                               t_span=(0, days),
                               y0=y0,
                               t_eval=np.arange(0, days+1))

    # fig02 = plt.figure()
    # plt.plot(y.t, y.y.T)
    # # plt.legend(["s", "e", "i", "r"])
    # plt.legend(["Susceptible", "Exposed", "Infected", "Removed"])
    # plt.text(0.8*days, 0.9, r'$R_{0}$ ='+str(np.round(R0, 2)))
    # plt.xlabel('t (days)')
    # plt.ylabel('Relative population')
    # plt.title(title)
    # plt.xlim([0, days])
    # plt.ylim([0, 1])
    # plt.grid(axis='y')
    # plt.tight_layout()

    s, e, i, r = [y.y[line, :] for line in range(4)]
    t = y.t
    return s, e, i, r, t


# def SEIR_plot(s, e, i, r, t, R0, title, pos, ts0, pars0):
#     y = np.array([s, e, i, r])
    
#     fig02 = plt.figure()
#     plt.plot(y.t, y.y.T)
#     # plt.legend(["s", "e", "i", "r"])
#     plt.legend(["Susceptible", "Exposed", "Infected", "Removed"])
#     plt.text(0.8*(len(t)-1), 0.9, r'$R_{0}$ ='+str(np.round(R0, 2)))
#     plt.xlabel('t (days)')
#     plt.ylabel('Relative population')
#     plt.title(title)
#     plt.xlim([0, (len(t)-1)])
#     plt.ylim([0, 1])
#     plt.grid(axis='y')
#     plt.tight_layout()
    
#     fig03 = plt.figure()
#     plt.plot(t, pos, label="Positives")

#     if ts0 != 0:
#         plt.plot(x, exponential(x, *pars0), 'r--', alpha=0.4,
#                   label="Deterministic exp. growth")

#     plt.plot(xi, yi, label="Initial growth", linewidth=2)
#     plt.plot(x, exponential(x, *pars), 'k--',
#               label="Exponential fit", alpha=0.8)

#     plt.xlim([0, 3*max(xi)])
#     plt.ylim([0, 1.4*np.nanmax(pos)])
#     plt.xlabel('t (days)')
#     plt.ylabel('Individuals')
#     plt.text(D*0.5, np.nanmax(pos,)*0.75,
#               r'$K$ =' + str(np.round(K0, 2)) +
#               r'; $T_{d}$ =' + str(np.round(Td0, 2)))  # +
#     # r'; $\tau_{s}$ =' + str(np.round(ts0, 2)))
#     plt.legend(loc='best')
#     plt.title(title + " - initial phase of the epidemic")
#     plt.grid(axis='y')
#     plt.tight_layout()
    
#     fig04 = plt.figure()
#     plt.plot(np.arange(np.min(t)-50, np.max(t)+50),  # red line at Rt == 1
#               [1 for i in np.arange(1, len(t)+100)],
#               'r--', linewidth=2, alpha=0.4)

#     plt.plot(t, Rt, alpha=0.8,
#               label='R(t) as R0 times s(t)')

#     plt.plot(t, Rti, 'grey', alpha=0.2,
#               label='R(t) from instant. K(t)')

#     plt.plot(t, Rts, 'orange',  # alpha=0.8,
#               label='Moving avg. of R(t) from K(t)')

#     # plt.plot(xi, np.ones(len(xi)) * R0_K, 'g--',
#     #          label='R0 from exponential growth')
#     plt.xlabel('t (days)')
#     plt.ylabel('R(t)')
#     plt.legend(loc='best')
#     plt.xlim([np.min(t), np.max(t)])
#     # plt.ylim([0, np.nanmax([R0, *Rts])+0.5])
#     plt.ylim([0, R0+2])
#     plt.title(title + " - evolution of the reproduction number")
#     plt.grid(axis='y')
#     plt.tight_layout()
    
#     return fig02, fig03, fig04


def SIR_odet(perc_inf, beta, tau_r, days, title):
    '''
    SIR Epidemic Model

    Parameters
    ----------
    perc_inf : float
        Initially infected percentage of the population [0, 100].
    beta : float
        Probability of transmission in one day [0, 1].
    tau_r : int
        average recovery time [days].
    days : int
        Total number of simulated days.

    Returns
    -------
    s, i, r : floats
        Relative populations
    t : float
        Time array

    '''
    # calculate constants
    frac_inf = perc_inf/100
    mu = 1/tau_r

    y0 = np.array([(1-frac_inf), frac_inf, 0])
    y = y0

    def dydt(t, y):
        return np.array([-beta*y[0]*y[1],                   # ds/dt
                         beta*y[0]*y[1] - mu*y[1],          # di/dt
                         mu*y[1]])                          # dr/dt

    y = sp.integrate.solve_ivp(fun=dydt,
                               t_span=(0, days),
                               y0=y0,
                               t_eval=np.arange(0, days+1))

    # fig02 = plt.figure()
    # plt.plot(y.t, y.y.T)
    # plt.legend(["Susceptible", "Infected", "Removed"])
    # plt.text(0.8*days, 0.9, r'$R_{0}$ ='+str(np.round(R0, 2)))
    # plt.xlabel('t (days)')
    # plt.ylabel('Relative population')
    # plt.title(title)
    # plt.xlim([0, days])
    # plt.ylim([0, 1])
    # plt.grid(axis='y')
    # plt.tight_layout()

    s, i, r = [y.y[line, :] for line in range(3)]
    t = y.t
    return s, i, r, t


# def SIR_plot(s, i, r, t, R0, title):
#     y = np.array([s, i, r])
#     fig02 = plt.figure()
#     plt.plot(t, y.T)
#     plt.legend(["Susceptible", "Infected", "Removed"])
#     plt.text(0.8*(len(t)-1), 0.9, r'$R_{0}$ ='+str(np.round(R0, 2)))
#     plt.xlabel('t (days)')
#     plt.ylabel('Relative population')
#     plt.title(title)
#     plt.xlim([0, (len(t)-1)])
#     plt.ylim([0, 1])
#     plt.grid(axis='y')
#     plt.tight_layout()
    
#     # fig03 = plt.figure()
#     # plt.plot(t, pos, label="Positives")

#     # if tsDet != 0:
#     #     plt.plot(x, exponential(x, *parsDet), 'r--', alpha=0.4,
#     #              label="Deterministic exp. growth")

#     # plt.plot(xi, yi, label="Initial growth", linewidth=2)
#     # plt.plot(x, exponential(x, *pars), 'k--',
#     #          label="Exponential fit", alpha=0.8)

#     # plt.xlim([0, 3*max(xi)])
#     # plt.ylim([0, 1.4*np.nanmax(pos)])
#     # plt.xlabel('t (days)')
#     # plt.ylabel('Individuals')
#     # plt.text(D*0.5, np.nanmax(pos,)*0.75,
#     #          r'$K$ =' + str(np.round(K0, 2)) +
#     #          r'; $T_{d}$ =' + str(np.round(Td0, 2)))  # +
#     # # r'; $\tau_{s}$ =' + str(np.round(ts0, 2)))
#     # plt.legend(loc='best')
#     # plt.title(title + " - initial phase of the epidemic")
#     # plt.grid(axis='y')
#     # plt.tight_layout()
    
    
#     # fig04 = plt.figure()
#     # plt.plot(np.arange(np.min(t)-50, np.max(t)+50),  # red line at Rt == 1
#     #          [1 for i in np.arange(1, len(t)+100)],
#     #          'r--', linewidth=2, alpha=0.4)

#     # plt.plot(t, Rt, alpha=0.8,
#     #          label='R(t) as R0 times s(t)')

#     # plt.plot(t, Rti, 'grey', alpha=0.2,
#     #          label='R(t) from instant. K(t)')

#     # plt.plot(t, Rts, 'orange',  # alpha=0.8,
#     #          label='Moving avg. of R(t) from K(t)')

#     # # plt.plot(xi, np.ones(len(xi)) * R0_K, 'g--',
#     # #          label='R0 from exponential growth')
#     # plt.xlabel('t (days)')
#     # plt.ylabel('R(t)')
#     # plt.legend(loc='best')
#     # plt.xlim([np.min(t), np.max(t)])
#     # # plt.ylim([0, np.nanmax([R0, *Rts])+0.5])
#     # plt.ylim([0, R0+2])
#     # plt.title(title + " - evolution of the reproduction number")
#     # plt.grid(axis='y')
#     # plt.tight_layout()
    
#     return fig02, fig03, fig04


def growth_fit(pos, t, ts0, pars0, D, R0):
    # Locates and analyses the first phase of an epidemic spread

    # Growth flex
    # incr = np.gradient(np.gradient(np.gradient(np.gradient(pos))))  # fourth
    # incr = np.gradient(np.gradient(np.gradient(pos)))  # third derivative
    # incr = np.gradient(np.gradient(pos))             # second derivative
    # incr = np.gradient(pos)                            # first derivative

    # idx = D
    # while incr[idx] > 0:
    #     idx += 1

    # # isolates the exponential-like growth
    # xi = t[int(D/2):(int(idx*0.45))]
    # yi = pos[int(D/2):(int(idx*0.45))]
    start = np.min(np.where(np.isfinite(pos)))
    # end = int(idx*0.55)
    f = 0.16
    end = np.min(np.where(pos > f * np.nanmax(pos)))

    while end - start < 5:
        f += 0.01
        end = np.min(np.where(pos > f * np.nanmax(pos)))
    print("Initial growth: [start, end, f(end)/max] " + str([start, end, f]))

    xi = t[start:end]
    yi = pos[start:end]

    parsFit, cov = curve_fit(f=exponential,
                             xdata=xi,
                             ydata=yi,
                             bounds=(-100, 100))
    print("Exponential fit parameters: " + str(parsFit))
    KFit = parsFit[0]
    TdFit = np.log(2)/KFit

    # Serial interval
    tsFit = np.log(R0)/KFit

    x = np.linspace(0, 2*max(xi), 100)  # 1.3*max(xi)

    # fig03 = plt.figure()
    # plt.plot(t, pos, label="Positives")

    # if tsDet != 0:
    #     plt.plot(x, exponential(x, *parsDet), 'r--', alpha=0.4,
    #              label="Deterministic exp. growth")

    # plt.plot(xi, yi, label="Initial growth", linewidth=2)
    # plt.plot(x, exponential(x, *pars), 'k--',
    #          label="Exponential fit", alpha=0.8)

    # plt.xlim([0, 3*max(xi)])
    # plt.ylim([0, 1.4*np.nanmax(pos)])
    # plt.xlabel('t (days)')
    # plt.ylabel('Individuals')
    # plt.text(D*0.5, np.nanmax(pos,)*0.75,
    #          r'$K$ =' + str(np.round(K0, 2)) +
    #          r'; $T_{d}$ =' + str(np.round(Td0, 2)))  # +
    # # r'; $\tau_{s}$ =' + str(np.round(ts0, 2)))
    # plt.legend(loc='best')
    # plt.title(title + " - initial phase of the epidemic")
    # plt.grid(axis='y')
    # plt.tight_layout()

    return x, xi, parsFit, KFit, TdFit, tsFit


def contagion_metrics(s, e, i, r, t,
                      K0, ts0, pars0, D,
                      R0, tau_i, tau_r, N):
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


    '''
    # Serial interval
    # ts = tau_r + tau_i  # ACTUALLY WRONG

    # Total positives (exposed + infected)
    pos = N * (e+i)

    # Rt from s(t)
    Rt = R0 * s

    # Smoothing of positives based on the average times "tau_r + tau_i"
    # D = int(2*tsDet)
    # pos_s = pd.Series(list(pos)).rolling(window=D,
    #                                     min_periods=D,
    #                                     center=True).mean().values

    # Initial exponential growth
    x, xi, parsFit, KFit, TdFit, tsFit = growth_fit(pos, t,
                                                    ts0, pars0,
                                                    D, R0)

    if ts0 == 0:
        ts0 = tsFit
    if K0 == 0:
        K0 = KFit

    # Actual growth rate
    Ki = np.gradient(np.log(pos)) / np.gradient(t)

    # Reproduction number from instantaneours growing rate
    Rti = np.exp(Ki * (ts0))

    # Doubling time from the instantaneous growing rate
    Tdi = np.log(2)/Ki

    # Smoothed growth rate
    # Ks = np.gradient(np.log(pos_s)) / np.gradient(t)

    # Reproduction number from smoothed growing rate
    # Rts = np.exp(Ks * (ts))
    Rts = pd.Series(list(Rti)).rolling(window=D,
                                       min_periods=D,
                                       center=True).mean().values

    # Doubling time from the smoothed growing rate
    # Tds = np.log(2)/Ks
    Tds = pd.Series(list(Tdi)).rolling(window=D,
                                       min_periods=D,
                                       center=True).mean().values

    # Es = pd.Series(list(N*e)).rolling(window=D,
    #                                   min_periods=1,
    #                                   center=True).mean().values
    # xi, pars = growth_fit(Es, tau_r)

    # # R0 from the initial exponential growth
    # R0_K = np.exp(K0 * (ts))

    print("R0 [predicted, estimated]: " +
          str(np.round([R0, Rts[np.min(np.where(np.isfinite(Rts)))]], 2)))
    print("Serial [predicted, estimated]: " +
          str(np.round([ts0, tsFit], 2)))

    # fig04 = plt.figure()
    # plt.plot(np.arange(np.min(t)-50, np.max(t)+50),  # red line at Rt == 1
    #          [1 for i in np.arange(1, len(t)+100)],
    #          'r--', linewidth=2, alpha=0.4)

    # plt.plot(t, Rt, alpha=0.8,
    #          label='R(t) as R0 times s(t)')

    # plt.plot(t, Rti, 'grey', alpha=0.2,
    #          label='R(t) from instant. K(t)')

    # plt.plot(t, Rts, 'orange',  # alpha=0.8,
    #          label='Moving avg. of R(t) from K(t)')

    # # plt.plot(xi, np.ones(len(xi)) * R0_K, 'g--',
    # #          label='R0 from exponential growth')
    # plt.xlabel('t (days)')
    # plt.ylabel('R(t)')
    # plt.legend(loc='best')
    # plt.xlim([np.min(t), np.max(t)])
    # # plt.ylim([0, np.nanmax([R0, *Rts])+0.5])
    # plt.ylim([0, R0+2])
    # plt.title(title + " - evolution of the reproduction number")
    # plt.grid(axis='y')
    # plt.tight_layout()

    return KFit, Ki, tsFit, parsFit, Rt, Rti, Rts, TdFit, Tdi, Tds


def SEIR_network(G, N, perc_inf, beta, tau_i, tau_r, days, t):
    # G = jk.graph_tools(G)

    # Alternativa a grah_tools(G) per il solo degree
    k = G.degree()
    G.degree_list = [d for n, d in k]
    G.k_avg = np.mean(G.degree_list)
    # print(G.k_avg)

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
    # trends = model.build_trends(iterations)

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
    # viz = DiffusionTrend(model, trends)
    # viz.plot()
    return s, e, i, r


class randnet:
    '''
    Random Networks for pandemic studies
    '''

    def __init__(self, name, nickname, graph, graph_small):
        self.name = name
        self.nick = nickname
        self.G = graph
        self.Gmini = graph_small
        self.N = self.G.number_of_nodes()

        self.G = graph_tools(self.G)
        self.Gmini = graph_tools(self.Gmini)

        self.Gmini = graph_plots(self.Gmini, self.name, [0])
        self.fig00 = self.Gmini.fig0

        self.G = graph_plots(self.G, self.name, [1])
        self.fig01 = self.G.fig1

        with open('pickle/' + "network_" + self.nick + '.pkl', 'wb') as f:
            pickle.dump(self, f)


class pRandNeTmic(randnet):

    def __init__(self, perc_inf, beta, tau_i, tau_r, days, t):
        self.R0 = beta*tau_r
        self.t = t
        self.days = days
        self.perc_inf = perc_inf
        self.beta = beta
        self.tau_i = tau_i
        self.tau_r = tau_r

        # run deterministic reference
        s, e, i, r, t = SEIR_odet(perc_inf, beta, tau_i, tau_r, days,
                                 "SEIR deterministic model")
        p = e + i
        self.mu = 1/tau_r
        self.gamma = 1/tau_i
        A = np.array([[-self.gamma, self.beta*s[0]], [self.gamma, -self.mu]])
        eigval, eigvec = np.linalg.eig(A)
        self.K0 = eigval[0]
        self.ts0 = np.log(self.R0)/self.K0
        self.pars0 = [self.K0, p[0]*self.N] 
            
    def run(self, n):
        # run n simulations
        self.s, self.e, self.i, self.r = \
            SEIR_network(self.G, self.N, self.perc_inf, self.beta, 
                         self.tau_i, self.tau_r, self.days, self.t)
            
        self.KFit, self.Ki, self.tsFit, self.parsFit, self.Rt, self.Rti, \
            self.Rts, self.TdFit, self.Tdi, self.Tds = \
            contagion_metrics(s=self.s, e=self.e, i=self.i, r=self.r, t=self.t,
                              K0=self.K0, ts0=self.ts0, pars0=self.pars0,
                              D=int(self.ts0), R0=self.R0, tau_i=self.tau_i,
                              tau_r=self.tau_r, N=self.N)

    def plot(self):  # , beta, tau_i, tau_r, days, t):
        # pop array
        y = np.array([self.s, self.e, self.i, self.r])

        # main plot
        self.fig02 = plt.figure()
        plt.plot(self.t, y.T)
        plt.legend(["Susceptible", "Exposed", "Infected", "Removed"])
        plt.text(0.3*days, 0.9, r'$R_{0}$ ='+str(np.round(self.R0, 2)))
        plt.xlabel('t (days)')
        plt.ylabel('Relative population')
        plt.title(self.name + " - SEIR time evolution")
        plt.xlim([0, days])
        plt.ylim([0, 1])
        plt.grid(axis='y')
        plt.tight_layout()
        
        self.fig03 = plt.figure()
        plt.plot(t, pos, label="Positives")
    
        if tsDet != 0:
            plt.plot(x, exponential(x, *parsDet), 'r--', alpha=0.4,
                     label="Deterministic exp. growth")
    
        plt.plot(xi, yi, label="Initial growth", linewidth=2)
        plt.plot(x, exponential(x, *pars), 'k--',
                 label="Exponential fit", alpha=0.8)
    
        plt.xlim([0, 3*max(xi)])
        plt.ylim([0, 1.4*np.nanmax(pos)])
        plt.xlabel('t (days)')
        plt.ylabel('Individuals')
        plt.text(D*0.5, np.nanmax(pos,)*0.75,
                 r'$K$ =' + str(np.round(K0, 2)) +
                 r'; $T_{d}$ =' + str(np.round(Td0, 2)))  # +
        # r'; $\tau_{s}$ =' + str(np.round(ts0, 2)))
        plt.legend(loc='best')
        plt.title(self.name + " - initial phase of the epidemic")
        plt.grid(axis='y')
        plt.tight_layout()

        self.fig04 = plt.figure()
        plt.plot(np.arange(np.min(t)-50, np.max(t)+50),  # red line at Rt == 1
                 [1 for i in np.arange(1, len(t)+100)],
                 'r--', linewidth=2, alpha=0.4)
    
        plt.plot(t, Rt, alpha=0.8,
                 label='R(t) as R0 times s(t)')
    
        plt.plot(t, Rti, 'grey', alpha=0.2,
                 label='R(t) from instant. K(t)')
    
        plt.plot(t, Rts, 'orange',  # alpha=0.8,
                 label='Moving avg. of R(t) from K(t)')
    
        # plt.plot(xi, np.ones(len(xi)) * R0_K, 'g--',
        #          label='R0 from exponential growth')
        plt.xlabel('t (days)')
        plt.ylabel('R(t)')
        plt.legend(loc='best')
        plt.xlim([np.min(t), np.max(t)])
        # plt.ylim([0, np.nanmax([R0, *Rts])+0.5])
        plt.ylim([0, R0+2])
        plt.title(self.name + " - evolution of the reproduction number")
        plt.grid(axis='y')
        plt.tight_layout()

        # self.K0, self.Ki, self.ts, self.pars, self.Rt, self.Rti, \
        #     self.Rts, self.Td0, self.Tdi, self.Tds, self.fig03, self.fig04 = \
        #     contagion_metrics(self.s, self.e, self.i, self.r, t, K, ts, pars,
        #                       D, beta*tau_r, tau_i, tau_r, self.N, self.name)

    def save(self):
        self.fig00.savefig('immagini/' + self.nick + '_00.png')
        self.fig01.savefig('immagini/' + self.nick + '_01.png')
        self.fig02.savefig('immagini/' + self.nick + '_02.png')
        self.fig03.savefig('immagini/' + self.nick + '_03.png')
        self.fig04.savefig('immagini/' + self.nick + '_04.png')

        with open('pickle/' + "simulations_" + self.nick + '.pkl', 'wb') as f:
            pickle.dump(self, f)
