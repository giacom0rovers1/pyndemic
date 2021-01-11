#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:44:38 2020

@author: giacomo
"""
import time
import copy
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import datetime as dt
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

    print("    [1/4] Calculating connectivity...")
    k = G.degree()
    G.degree_list = [d for n, d in k]

    G.k_avg = np.mean(G.degree_list)
    G.k_std = np.std(G.degree_list)
    G.k_var = np.var(G.degree_list)

    G.degree_sequence = sorted(G.degree_list, reverse=True)
    G.k_max = np.max(G.degree_list)  # G.degree_sequence[0]
    G.k_min = np.min(G.degree_list)  # G.degree_sequence[-1]
    G.k_histD = np.array(nx.degree_histogram(G))/G.nn

    # Prima correzione per non avere matrice singolare
    G.kx = np.array(range(len(G.k_histD)), ndmin=2) + 1e-10
    G.kfm = np.dot(G.kx, G.k_histD).item()
    G.ksm = np.dot(G.kx**2, G.k_histD).item()
    G.Lma = G.ksm/G.kfm - 1

    G.Crita = G.k_avg/G.Lma

    G.ktilde = np.transpose((G.kx - 1)/G.kx)

    # Seconda correzione per non avere matrice singolare
    G.Mix = nx.degree_mixing_matrix(G)[G.k_min:, G.k_min:] + 1e-10

    G.P = 1/sum(G.Mix) * G.Mix
    G.Ckk = G.kx[:, G.k_min:] * G.P * G.ktilde[G.k_min:, :]
    G.w, G.v = np.linalg.eig(G.Ckk)
    G.Lm = np.real(np.nanmax(G.w))
    G.Crit = G.k_avg/G.Lm
    print("    Lm:   " + str(np.round([G.Lma, G.Lm], 2)))

    print("    Connectivity degree histogram completed.")

    print("    [2/4] Calculating betweenness centrality...")
    BC = nx.betweenness_centrality(G)  # , normalized = False)
    G.BC_list = [bc for bc in BC.values()]
    print("    Betweenness centrality list completed.")

    # Cl = nx.closeness_centrality(G)
    # Cl_list = [cl for cl in Cl.values()]

    # G.eig_list = nx.adjacency_spectrum(G)
    # # eig_sequence = sorted(eig_list, reverse=True)

    # G.A = nx.adj_matrix(G)  # create a sparse adjacency matrix
    # G.A = G.A.asfptype()  # convert the sparse values from int to float
    # # plt.spy(A)
    # G.eig_val, G.eig_vec = sp.sparse.linalg.eigs(G.A)

    print("    [3/4] Calculating clustering...")
    C = nx.clustering(G)
    G.C_list = [c for c in C.values()]
    G.C_avg = nx.average_clustering(G)
    print("    Clustering list completed.")

    print("    [4/4] Calculating shortest path lengths...")
    G.L_avg = nx.average_shortest_path_length(G)
    print("    Average shortest path lengt estimated.")

    toc = time.perf_counter()
    print(f'--> graph_tools() completed in {toc - tic:0.0f} seconds.\n')

    return G


def graph_plots(G,  net_name, plots_to_print=[0, 1], cmap=plt.cm.Blues):
    ''' provide 'plots_to_print' as a list '''
    colormap = truncate_colormap(cmap, 0.2, 0.8)
    plots_to_print = list(plots_to_print)

    if 0 in plots_to_print:
        # Network visualization
        G.fig0 = plt.figure(figsize=(4, 3.2))  # (7, 5.5))

        G.fig0.add_subplot(111)
        nx.draw_networkx_edges(G, pos=nx.kamada_kawai_layout(G),
                               alpha=0.3)  # 0.4
        nx.draw_networkx_nodes(G, pos=nx.kamada_kawai_layout(G),
                               node_size=30,  # 80
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
        G.fig1 = plt.figure(figsize=(6.5, 3))  # (11, 5))
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

        sk = 0
        try:  # len(yi) > 0:
            while sk < 10:
                G.sf_pars, G.sf_cov = curve_fit(f=scale_free,
                                                xdata=xi[sk:],
                                                ydata=yi[sk:],
                                                p0=[1, 1],
                                                bounds=(1e-10, np.inf))
                break
            else:
                G.sf_pars = [0.8, 1.5]
                print("Wrong power law fit")

        except ValueError:
            sk += 1

        # Degree histogram
        G.fig1.add_subplot(121)
        plt.scatter(xi, yi, alpha=0.75, s=15)
        plt.xlabel('Node connectivity degree k')
        plt.ylabel('Degree distribution p(k)')
        #plt.title('Average connectivity degree = ' + str(np.round(G.k_avg, 1)))
        plt.plot(x, G.gauss, 'r--', alpha=0.5, label='Normal distr.')
        # plt.plot(xi, G.poiss, 'y--', label='Poisson')
        plt.plot(x, scale_free(x, *G.sf_pars), 'b--',
                 alpha=0.5, label='Power law')
        plt.ylim([0, max(yi)+0.1])
        plt.legend(loc='best')

        # BC vs K
        G.fig1.add_subplot(122)
        plt.scatter(G.degree_list, G.BC_list, alpha=0.75, s=15)
        #plt.title('Average clustering coeff. = ' + str(np.round(G.C_avg, 3)))
        plt.xlabel("Node connectivity degree k")
        plt.ylabel("Node betweenness centrality BC")
        plt.xlim(0.5, G.k_max+0.5)
        plt.ylim([0, max(G.BC_list)+0.0001])
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
        plt.xlabel(r"Eigenvalue $\Lambda$")
        plt.ylabel("Density")
        plt.title("Adjacency spectrum")

        # Eigenvector functions
        G.fig4.add_subplot(212)
        for i in range(G.eig_val.size):
            lab = r'$\Lambda =$' + str(np.round(G.eig_val.real, 2)[i])
            plt.plot(G.eig_vec[:, i], alpha=0.5,
                     label=lab)
        plt.legend(loc='best')
        plt.tight_layout()

    # TODO Laplacian matrix and spectrum

    return G


# EPIDEMIC PROCESSES


def SEIR_odet(perc_inf, lmbda, tau_i, tau_r, days):
    '''
    SEIR Epidemic Model

    Parameters
    ----------
    perc_inf : float
        Initially infected percentage of the population [0, 100].
    lmbda : float
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

    # y0 = np.array([(1-frac_inf), frac_inf*(1-lmbda), frac_inf*lmbda, 0])
    y0 = np.array([(1-frac_inf), 0, frac_inf, 0])

    y = y0

    def dydt(t, y):
        return np.array([-lmbda*y[0]*y[2],                   # ds/dt
                         lmbda*y[0]*y[2] - gamma*y[1],       # de/dt
                         gamma*y[1] - mu*y[2],              # di/dt
                         mu*y[2]])                          # dr/dt

    y = sp.integrate.solve_ivp(fun=dydt,
                               t_span=(0, days),
                               y0=y0,
                               t_eval=np.arange(0, days+1))

    s, e, i, r = [y.y[line, :] for line in range(4)]
    t = y.t
    return s, e, i, r, t


def SEIR_plot(s, e, i, r, t, R0, title, pos, ts0, pars0, x, xi, yi,
              parsFit, D, KFit, TdFit, Rt, Rti):
    y = np.array([s, e, i, r])

    fig02 = plt.figure(figsize=(5.5, 5), dpi=300)
    plt.plot(t, y.T)
    # plt.legend(["s", "e", "i", "r"])
    plt.legend(["Susceptible", "Exposed", "Infected", "Removed"])
    plt.text(D+np.min(t), 0.5, r'$R_{0}$ ='+str(np.round(R0, 2)))
    plt.xlabel('t (days)')
    plt.ylabel('Relative population')
    plt.title(title)
    # plt.xlim([0, (len(t)-1)])
    plt.xlim([np.min(t), 0.66*np.max(t)])
    plt.ylim([0, 1])
    plt.grid(axis='y')
    plt.tight_layout()

    fig03 = plt.figure(figsize=(5.5, 5), dpi=300)
    plt.plot(t, pos, label="Positives")

    if ts0 != 0:
        plt.plot(x, exponential(x, *pars0), 'r--', alpha=0.4,
                 label="Deterministic exp. growth")

    plt.plot(xi, yi, label="Initial growth", linewidth=2)
    plt.plot(x, exponential(x, *parsFit), 'k--',
             label="Exponential fit", alpha=0.8)

    # plt.xlim([0, 3*max(xi)])
    plt.xlim([np.min(t), 3*np.max(xi)])
    plt.ylim([0, 1.4*np.nanmax(pos)])
    plt.xlabel('t (days)')
    plt.ylabel('Individuals')
    plt.text(D*0.5+np.min(t), np.nanmax(pos,)*0.75,
             r'$K$ =' + str(np.round(KFit, 2)) +
             r'; $T_{d}$ =' + str(np.round(TdFit, 2)))  # +
    # r'; $\tau_{s}$ =' + str(np.round(ts0, 2)))
    plt.legend(loc='best')
    plt.title(title + " - initial phase of the epidemic")
    plt.grid(axis='y')
    plt.tight_layout()

    fig04 = plt.figure(figsize=(5.5, 5), dpi=300)
    plt.plot(np.arange(np.min(t)-50, np.max(t)+50),  # red line at Rt == 1
             [1 for i in np.arange(1, len(t)+100)],
             'r--', linewidth=2, alpha=0.4)

    plt.plot(t, Rt, alpha=0.8,
             label='R(t) as R0 times s(t)')

    plt.plot(t, Rti, 'orange',
             label='R(t) from instant. K(t)')

    plt.xlabel('t (days)')
    plt.ylabel('R(t)')
    plt.legend(loc='best')
    plt.xlim([np.min(t), 0.66*np.max(t)])
    plt.ylim([0, R0+2])
    plt.title(title + " - evolution of the reproduction number")
    plt.grid(axis='y')
    plt.tight_layout()

    return fig02, fig03, fig04


def SIR_odet(perc_inf, lmbda, tau_r, days):
    '''
    SIR Epidemic Model

    Parameters
    ----------
    perc_inf : float
        Initially infected percentage of the population [0, 100].
    lmbda : float
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
        return np.array([-lmbda*y[0]*y[1],                   # ds/dt
                         lmbda*y[0]*y[1] - mu*y[1],          # di/dt
                         mu*y[1]])                          # dr/dt

    y = sp.integrate.solve_ivp(fun=dydt,
                               t_span=(0, days),
                               y0=y0,
                               t_eval=np.arange(0, days+1))

    s, i, r = [y.y[line, :] for line in range(3)]
    t = y.t
    return s, i, r, t


def SIR_plot(s, i, r, t, R0, title, pos, ts0, pars0, x, xi, yi,
             parsFit, D, KFit, TdFit, Rt, Rti):
    y = np.array([s, i, r])
    fig02 = plt.figure(figsize=(5.3, 3.8), dpi=300)  #(6.4, 4.8), dpi=300)
    plt.plot(t, y.T)
    plt.legend(["Susceptible", "Infected", "Removed"])
    plt.text(D+np.min(t), 0.5, r'$R_{0}$ ='+str(np.round(R0, 2)))
    plt.xlabel('t (days)')
    plt.ylabel('Relative population')
    plt.title(title)
    # plt.xlim([0, (len(t)-1)])
    plt.xlim([np.min(t), 0.66*np.max(t)])
    plt.ylim([0, 1])
    plt.grid(axis='y')
    plt.tight_layout()

    fig03 = plt.figure(figsize=(5.5, 5), dpi=300)
    plt.plot(t, pos, label="Positives")

    if ts0 != 0:
        plt.plot(x, exponential(x, *pars0), 'r--', alpha=0.4,
                 label="Deterministic exp. growth")

    plt.plot(xi, yi, label="Initial growth", linewidth=2)
    plt.plot(x, exponential(x, *parsFit), 'k--',
             label="Exponential fit", alpha=0.8)

    # plt.xlim([0, 3*max(xi)])
    plt.xlim([np.min(t), 3*np.max(xi)])
    plt.ylim([0, 1.4*np.nanmax(pos)])
    plt.xlabel('t (days)')
    plt.ylabel('Individuals')
    plt.text(D*0.5+np.min(t), np.nanmax(pos,)*0.75,
             r'$K$ =' + str(np.round(KFit, 2)) +
             r'; $T_{d}$ =' + str(np.round(TdFit, 2)))  # +
    # r'; $\tau_{s}$ =' + str(np.round(ts0, 2)))
    plt.legend(loc='best')
    plt.title(title + " - initial phase of the epidemic")
    plt.grid(axis='y')
    plt.tight_layout()

    fig04 = plt.figure(figsize=(5.3, 4), dpi=300)  #(6.4, 4.8), dpi=300)
    plt.plot(np.arange(np.min(t)-50, np.max(t)+50),  # red line at Rt == 1
             [1 for i in np.arange(1, len(t)+100)],
             'r--', linewidth=2, alpha=0.4)

    plt.plot(t, Rt, alpha=0.8,
             label='R(t) as R0 times s(t)')

    plt.plot(t, Rti, 'orange',
             label='R(t) from instant. K(t)')

    plt.xlabel('t (days)')
    plt.ylabel('R(t)')
    plt.legend(loc='best')
    plt.xlim([np.min(t), 0.66*np.max(t)])
    plt.ylim([0, R0+2])
    plt.title(title + " - evolution of the reproduction number")
    plt.grid(axis='y')
    plt.tight_layout()

    return fig02, fig03, fig04


def growth_fit(pos, t, ts0, pars0, D, R0):
    # Locates and analyses the first phase of an epidemic spread
    start = np.nanmin(np.where(np.isfinite(pos)))

    f = 0.16
    end = start + 1

    while (end - start) < 14:   # before: 5

        if f > 0.333:
            break

        a = np.where(pos > f * np.nanmax(pos))

        if len(a) == 0:
            end = start + 15    # before: 6
        else:
            end = np.nanmin(a)
            f += 0.001

    print("\nInitial growth:\n[s  e] f    ")
    print([start, end], np.array(f).round(3))

    xi = t[start:end]
    yi = pos[start:end]

    parsFit, cov = curve_fit(f=exponential,
                             xdata=xi,
                             ydata=yi,
                             bounds=(-1e4, 1e4))
    print("\nExp fit params:\n[ K     n(0)  ]")
    print(np.array(parsFit).round(3))

    KFit = parsFit[0]
    TdFit = np.log(2)/KFit

    # Serial interval
    tsFit = np.log(R0)/KFit

    x = np.linspace(0, 2*max(xi), 100)

    return x, xi, yi, parsFit, KFit, TdFit, tsFit


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

    # Total positives (exposed + infected)
    pos = N * (e+i)

    # Rt from s(t)
    Rt = R0 * s

    # Initial exponential growth
    x, xi, yi, parsFit, KFit, TdFit, tsFit = growth_fit(pos, t,
                                                        ts0, pars0,
                                                        D, R0)
    # if ts0 == 0:
    #     ts0 = tsFit
    # if K0 == 0:
    #     K0 = KFit

    # Actual growth rate
    Ki = np.gradient(np.log(pos)) / np.gradient(t)

    # Reproduction number from instantaneours growing rate
    Rti = np.exp(Ki * (ts0))

    # Doubling time from the instantaneous growing rate
    Tdi = np.log(2)/Ki

    # Rts = pd.Series(list(Rti)).rolling(window=D,
    #                                    min_periods=D,
    #                                    center=True).mean().values

    # Tds = pd.Series(list(Tdi)).rolling(window=D,
    #                                    min_periods=D,
    #                                    center=True).mean().values

    print("\nR0\n[pred esti]: ")
    print(np.array([R0, Rti[3*D]]).round(2))

    print("\nGrowth rate\n[pred esti fit ]")
    print(np.array([K0, Ki[3*D], KFit]).round(2))

    return x, xi, yi, KFit, Ki, tsFit, parsFit, Rt, Rti, TdFit, Tdi


def SEIR_network(G, N, perc_inf, beta, tau_i, tau_r, days, t):

    # Alternativa a graph_tools(G) per il solo degree
    k = G.degree()
    G.degree_list = [d for n, d in k]
    G.k_avg = np.mean(G.degree_list)

    # EPIDEMIC MODEL
    frac_inf = perc_inf/100
    gamma = 1/tau_i
    mu = 1/tau_r

    # Config:
    model = ep.SEIRModel(G)

    # print(model.parameters)
    # print(model.available_statuses)

    config = mc.Configuration()
    config.add_model_parameter('alpha', gamma)
    config.add_model_parameter('beta',  beta)
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

    def __init__(self, parent, perc_inf, beta, tau_i, tau_r, days):
        self.name = parent.name
        self.nick = parent.nick
        self.G = parent.G
        self.Gmini = parent.Gmini
        self.N = parent.N
        self.G = parent.G
        self.Gmini = parent.Gmini
        self.fig00 = parent.Gmini.fig0
        self.fig01 = parent.G.fig1
        self.lmbda = beta*self.G.k_avg
        self.R0 = self.lmbda*tau_r
        self.days = days
        self.perc_inf = perc_inf
        self.beta = beta
        self.tau_i = tau_i
        self.tau_r = tau_r

        # run deterministic reference
        s, e, i, r, t = SEIR_odet(self.perc_inf, self.lmbda,
                                  self.tau_i, self.tau_r, self.days)
        self.t = t
        p = e + i
        self.mu = 1/tau_r
        self.gamma = 1/tau_i
        self.A0 = np.array([[-self.gamma, self.lmbda*s[0]],
                            [self.gamma, -self.mu]])
        self.eigvalA0, self.eigvecA0 = np.linalg.eig(self.A0)
        self.K0 = self.eigvalA0[0]
        self.ts0 = np.log(self.R0)/self.K0
        self.pars0 = [self.K0, p[0]*self.N]
        self.D = int(self.ts0)

        # DBMF
        self.Ka_si = self.beta * self.G.Lma * s[0]
        self.K1_si = self.beta * self.G.Lm * s[0]

        self.Ka_sir = self.beta * self.G.Lma * s[0] - self.mu
        self.K1_sir = self.beta * self.G.Lm * s[0] - self.mu

        self.A1a = np.array([[-self.gamma, self.beta * self.G.Lma * s[0]],
                             [self.gamma, -self.mu]])
        self.eigvalA1a, self.eigvecA1a = np.linalg.eig(self.A1a)
        self.K1a = self.eigvalA1a[0]

        self.A1 = np.array([[-self.gamma, self.beta * self.G.Lm * s[0]],
                            [self.gamma, -self.mu]])
        self.eigvalA1, self.eigvecA1 = np.linalg.eig(self.A1)
        self.K1 = self.eigvalA1[0]

        self.pars1 = [self.K1a, p[0]*self.N]

    def run(self, runs):
        self.runs = runs
        if self.runs == 1:
            self.s, self.e, self.i, self.r = \
                SEIR_network(self.G, self.N, self.perc_inf, self.beta,
                             self.tau_i, self.tau_r, self.days, self.t)
            # compartments array
            self.y = np.array([self.s, self.e, self.i, self.r])
            self.pos = (self.e + self.i) * self.N
            self.x, self.xi, self.yi, \
                self.KFit, self.Ki, self.tsFit, self.parsFit, \
                self.Rt, self.Rti, self.TdFit, self.Tdi = \
                contagion_metrics(s=self.s, e=self.e, i=self.i,
                                  r=self.r, t=self.t,
                                  K0=self.K0, ts0=self.ts0,
                                  pars0=self.pars0,
                                  D=self.D, R0=self.R0, tau_i=self.tau_i,
                                  tau_r=self.tau_r, N=self.N)
        else:
            self.sm = pd.Series(data=None, dtype='float64')
            self.em = pd.Series(data=None, dtype='float64')
            self.im = pd.Series(data=None, dtype='float64')
            self.rm = pd.Series(data=None, dtype='float64')
            self.pm = pd.Series(data=None, dtype='float64')

            self.parsFitm0 = pd.Series(data=None, dtype='float64')
            self.parsFitm1 = pd.Series(data=None, dtype='float64')
            self.Rtim = pd.Series(data=None, dtype='float64')

            # run n simulations
            run = 0
            member = self.copy()

            while run < runs:
                print("\n" + str(run+1) + " of " + str(runs))

                member.s, member.e, member.i, member.r = \
                    SEIR_network(self.G, self.N, self.perc_inf, self.beta,
                                 self.tau_i, self.tau_r, self.days, self.t)

                member.pos = np.array((member.e + member.i) * self.N)

                try:
                    member.x, member.xi, member.yi, \
                        member.KFit, member.Ki, member.tsFit, member.parsFit, \
                        member.Rt, member.Rti, \
                        member.TdFit, member.Tdi = \
                        contagion_metrics(s=member.s, e=member.e, i=member.i,
                                          r=member.r, t=self.t,
                                          K0=self.K0, ts0=self.ts0,
                                          pars0=self.pars0,
                                          D=self.D, R0=self.R0,
                                          tau_i=self.tau_i,
                                          tau_r=self.tau_r, N=self.N)
                    run += 1

                except ValueError:
                    now = dt.datetime.now()
                    print("\nA VALUE ERROR OCCURRED in contagion_metrics()")
                    print(now)
                    logname = 'pickle/valerror_' + \
                        now.strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'
                    with open(logname, 'wb') as f:
                        pickle.dump([member, run, now], f)
                    print("Error log saved. Repeating run " + str(run))
                    continue

                self.sm = self.sm.append(pd.Series(member.s))
                self.em = self.em.append(pd.Series(member.e))
                self.im = self.im.append(pd.Series(member.i))
                self.rm = self.rm.append(pd.Series(member.r))
                self.pm = self.pm.append(pd.Series(member.pos))

                self.parsFitm0 = \
                    self.parsFitm0.append(pd.Series(member.parsFit[0]))
                self.parsFitm1 = \
                    self.parsFitm1.append(pd.Series(member.parsFit[1]))

                self.Rtim = self.Rtim.append(pd.Series(member.Rti))

            self.s = np.array([self.sm[i].median() for i in self.t])
            self.e = np.array([self.em[i].median() for i in self.t])
            self.i = np.array([self.im[i].median() for i in self.t])
            self.r = np.array([self.rm[i].median() for i in self.t])
            self.pos = np.array([self.pm[i].median() for i in self.t])

            self.s05 = np.array([self.sm[i].quantile(0.05) for i in self.t])
            self.e05 = np.array([self.em[i].quantile(0.05) for i in self.t])
            self.i05 = np.array([self.im[i].quantile(0.05) for i in self.t])
            self.r05 = np.array([self.rm[i].quantile(0.05) for i in self.t])
            self.p05 = np.array([self.pm[i].quantile(0.05) for i in self.t])

            self.s95 = np.array([self.sm[i].quantile(0.95) for i in self.t])
            self.e95 = np.array([self.em[i].quantile(0.95) for i in self.t])
            self.i95 = np.array([self.im[i].quantile(0.95) for i in self.t])
            self.r95 = np.array([self.rm[i].quantile(0.95) for i in self.t])
            self.p95 = np.array([self.pm[i].quantile(0.95) for i in self.t])

            # Contagion metrics of the median scenario
            self.x, self.xi, self.yi, \
                self.KFit, self.Ki, self.tsFit, self.parsFit, \
                self.Rt, self.Rti, \
                self.TdFit, self.Tdi = \
                contagion_metrics(s=self.s, e=self.e, i=self.i,
                                  r=self.r, t=self.t,
                                  K0=self.K0, ts0=self.ts0,
                                  pars0=self.pars0,
                                  D=self.D, R0=self.R0, tau_i=self.tau_i,
                                  tau_r=self.tau_r, N=self.N)

            self.Rt05 = self.R0 * self.s05
            self.Rt95 = self.R0 * self.s95

            self.parsFit50 = [self.parsFitm0.median(), self.parsFitm1.median()]
            self.parsFit05 = [self.parsFitm0.quantile(0.05),
                              self.parsFitm1.quantile(0.05)]
            self.parsFit95 = [self.parsFitm0.quantile(0.95),
                              self.parsFitm1.quantile(0.95)]

            self.KFit50 = self.parsFit50[0]
            self.TdFit50 = np.log(2)/self.KFit50

            self.Rti50 = np.array([self.Rtim[i].median() for i in self.t])
            self.Rti05 = np.array([self.Rtim[i].quantile(0.05)
                                   for i in self.t])
            self.Rti95 = np.array([self.Rtim[i].quantile(0.95)
                                   for i in self.t])

    def plot(self):
        
        self.Gmini = graph_plots(self.Gmini, self.name, [0])
        self.fig00 = self.Gmini.fig0

        self.G = graph_plots(self.G, self.name, [1])
        self.fig01 = self.G.fig1

        if self.runs == 1:
            # main plot
            self.fig02 = plt.figure(figsize=(5.3, 4), dpi=300)  #(6.4, 4.8), dpi=300)
            plt.plot(self.t, self.y.T)
            plt.legend(["Susceptible", "Exposed", "Infected", "Removed"])
            plt.text(self.D+np.min(self.t), 0.5, r'$R_{0}$ =' +
                     str(np.round(self.R0, 2)))
            plt.xlabel('t (days)')
            plt.ylabel('Relative population')
            plt.title(self.name + " - SEIR time evolution")
            plt.xlim([0, 0.66*self.days])
            plt.ylim([0, 1])
            plt.grid(axis='y')
            plt.tight_layout()

            # initial growth
            self.fig03 = plt.figure(figsize=(5.3, 4), dpi=300)  #(6.4, 4.8), dpi=300)
            plt.plot(self.t, self.pos, label="Positives")

            if self.ts0 != 0:
                plt.plot(self.x, exponential(self.x, *self.pars0),
                         'r--', alpha=0.4,
                         label="Deterministic exp. growth")

                plt.plot(self.x, exponential(self.x, *self.pars1),
                         'b--', alpha=0.4,
                         label="DBMF unc. exp. growth")

            plt.plot(self.xi, self.yi, label="Initial growth", linewidth=2)
            plt.plot(self.x, exponential(self.x, *self.parsFit), 'k--',
                     label="Exponential fit", alpha=0.8)

            plt.xlim([0, 3*max(self.xi)])
            plt.ylim([0, 1.4*np.nanmax(self.pos)])
            plt.xlabel('t (days)')
            plt.ylabel('Individuals')
            plt.text(self.D*0.5+np.min(self.t), np.nanmax(self.pos,)*0.75,
                     r'$K$ =' + str(np.round(self.KFit, 2)) +
                     r'; $T_{d}$ =' + str(np.round(self.TdFit, 2)))  # +
            plt.legend(loc='best')
            plt.title(self.name + " - initial phase of the epidemic")
            plt.grid(axis='y')
            plt.tight_layout()

            # Rt evolution
            self.fig04 = plt.figure(figsize=(5.3, 4), dpi=300)  #(6.4, 4.8), dpi=300)
            plt.plot(np.arange(np.min(self.t)-50, np.max(self.t)+50),
                     [1 for i in np.arange(1, len(self.t)+100)],
                     'r--', linewidth=2, alpha=0.4)
            plt.plot(np.arange(np.min(self.t)-50, np.max(self.t)+50),
                     [self.G.Crita for i in np.arange(1, len(self.t)+100)],
                     'b--', linewidth=2, alpha=0.4)

            plt.plot(self.t, self.Rt, alpha=0.8,
                     label='R(t) as R0 times s(t)')

            plt.plot(self.t, self.Rti, 'orange',
                     label='R(t) from instant. K(t)')

            plt.xlabel('t (days)')
            plt.ylabel('R(t)')
            plt.legend(loc='best')
            plt.xlim([np.min(self.t), 0.66*np.max(self.t)])
            plt.ylim([0, self.R0+2])
            plt.title(self.name + " - evolution of the reproduction number")
            plt.grid(axis='y')
            plt.tight_layout()

        else:
            # main plot
            self.fig02 = plt.figure(figsize=(5.3, 4), dpi=300)  #(6.4, 4.8), dpi=300)

            plt.fill_between(self.t, self.s05, self.s95, alpha=0.3)
            plt.plot(self.t, self.s, label="Susceptible")

            plt.fill_between(self.t, self.e05, self.e95, alpha=0.3)
            plt.plot(self.t, self.e, label="Exposed")

            plt.fill_between(self.t, self.i05, self.i95, alpha=0.3)
            plt.plot(self.t, self.i, label="Infected")

            plt.fill_between(self.t, self.r05, self.r95, alpha=0.3)
            plt.plot(self.t, self.r, label="Removed")

            plt.legend(loc='best')
            plt.text(self.D+np.min(self.t), 0.5, r'$R_{0}$ =' +
                     str(np.round(self.R0, 2)))
            plt.xlabel('t (days)')
            plt.ylabel('Relative population')
            plt.title(self.name + " - SEIR time evolution")
            # plt.xlim([0, self.days])
            plt.xlim([np.min(self.t), 0.66*np.max(self.t)])
            plt.ylim([0, 1])
            plt.grid(axis='y')
            plt.tight_layout()

            # initial growth
            self.fig03 = plt.figure(figsize=(5.3, 4), dpi=300)  #(6.4, 4.8), dpi=300)
            plt.fill_between(self.t, self.p05, self.p95, alpha=0.3)
            plt.plot(self.t, self.pos, label="Positives")

            if self.ts0 != 0:
                plt.plot(self.x, exponential(self.x, *self.pars0),
                         'r--', alpha=0.8,
                         label="Deterministic exp. growth")

                plt.plot(self.x, exponential(self.x, *self.pars1),
                         'b--', alpha=0.4,
                         label="DBMF unc. exp. growth")

            plt.plot(self.xi, self.yi, label="Initial growth", linewidth=2)

            plt.fill_between(self.x,
                             exponential(self.x, *self.parsFit05),
                             exponential(self.x, *self.parsFit95),
                             facecolor='grey', alpha=0.2)
            plt.plot(self.x, exponential(self.x, *self.parsFit50), 'k--',
                     label="Exponential fit", alpha=0.8)

            # plt.xlim([0, 3*max(self.xi)])
            plt.xlim([np.min(self.t), 3*np.max(self.xi)])
            plt.ylim([0, 1.4*np.nanmax(self.pos)])
            plt.xlabel('t (days)')
            plt.ylabel('Individuals')
            plt.text(self.D*0.5+np.min(self.t), np.nanmax(self.pos,)*0.75,
                     r'$K$ =' + str(np.round(self.KFit50, 2)) +
                     r'; $T_{d}$ =' + str(np.round(self.TdFit50, 2)))
            plt.legend(loc='best')
            plt.title(self.name + " - initial phase of the epidemic")
            plt.grid(axis='y')
            plt.tight_layout()

            # Rt evolution
            self.fig04 = plt.figure(figsize=(5.3, 4), dpi=300)  #(6.4, 4.8), dpi=300)
            plt.plot(np.arange(np.min(self.t)-50, np.max(self.t)+50),
                     [1 for i in np.arange(1, len(self.t)+100)],
                     'r--', linewidth=2, alpha=0.4)
            plt.plot(np.arange(np.min(self.t)-50, np.max(self.t)+50),
                     [self.G.Crita for i in np.arange(1, len(self.t)+100)],
                     'b--', linewidth=2, alpha=0.4)

            plt.fill_between(self.t, self.Rt05, self.Rt95, alpha=0.3)
            plt.plot(self.t, self.Rt, alpha=0.8,
                     label='R(t) as R0 times s(t)')

            plt.fill_between(self.t, self.Rti05, self.Rti95,
                             facecolor='orange', alpha=0.3)
            plt.plot(self.t, self.Rti50, 'orange',
                     label='R(t) from instant. K(t)')

            plt.xlabel('t (days)')
            plt.ylabel('R(t)')
            plt.legend(loc='best')
            plt.xlim([np.min(self.t), 0.66*np.max(self.t)])
            plt.ylim([0, self.R0+2])
            plt.title(self.name + " - evolution of the reproduction number")
            plt.grid(axis='y')
            plt.tight_layout()

    def copy(self):
        return copy.copy(self)

    def save(self):
        self.fig00.savefig('immagini/' + self.nick + '_00.png')
        self.fig01.savefig('immagini/' + self.nick + '_01.png')
        self.fig02.savefig('immagini/' + self.nick + '_02.png')
        self.fig03.savefig('immagini/' + self.nick + '_03.png')
        self.fig04.savefig('immagini/' + self.nick + '_04.png')

        with open('pickle/' + "simulations_" + self.nick + '.pkl', 'wb') as f:
            pickle.dump(self, f)
