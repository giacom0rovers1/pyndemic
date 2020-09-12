#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 10:22:16 2020

@author: giacomo

@title: Modello virale dinamico
"""

# Example: write Fibonacci series up to n
def fib(n):   
    """Print a Fibonacci series up to n."""
    [a, b] = [0, 1]
    while a < n:
        print(a, end=' ')
        [a, b] = [b, a + b]
    print()


def clock(network, t):
    """Advances the system of one step"""
    evol(network)
    t += 1

class network:
    """Population network"""
    
class node:
    """Population element"""
    
# caratteristiche del nodo (persona)
node
    .alive = T/F
    .inf = T/F                # stato infetto-non infetto
    .ill = T/F                # sintomi - no sintomi
    .init = t(infect -> T)    # momento del contagio
    .res = [0-1]              # resistenza del soggetto (modificabile, assegnata random)
    
    .links                    # caratteristiche del network
    

evol <- function(node, d, r){
# funzione di evoluzione secondo i parametri d = ritardo tra infezione e sintomi (incubazione), r = ritardo tra infezione e contagiosità
    
    if(node.alive == F) next
    if(node.inf == F) next
    
    if (node.inf==T & node.ill==F & node.res<1 & t==node.init+d){ node.ill <- T }        # apparsa dei sintomi
    
    if (node.inf==T & t==node.init+r){ viral(node)}         # contagio
    
    
    # progresso malattia (abbassa res)
    
    # cura malattia (alza res)
    
    
    if(node.inf & res = 1){                                 # guarigione (contare res, degenza)
        node.inf <- F   
        node.ill <- F
    }
    
    if(node.inf & node.res == 0){ node.alive <- F }         # morte 
    
}
    
viral <- function(node, a){
# funzione di contagio secondo il coefficiente a = percentuale di link contagiati in media al giorno (anche meno di uno)
    
    aa = gauss(mean=a)
    ll <- signif(length(node.links)*aa, 0)
    
    sorted <- rand.sort(node.links)[1:ll] 
    
    infect(node.links[sorted])
}


infect <-- function(node){
# funzione di infezione
    node.inf <- T
}


