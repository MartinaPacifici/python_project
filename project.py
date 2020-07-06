# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:59:43 2020

@author: feder
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import timeit
import random
import math
import json 
data = json.load(open("C:\\Users\\feder\\OneDrive\\Documenti\\Fede\\University\\STAT_M1\\AlgoritmiPython\\project\\dpc-covid19-ita-province.json"))

corona = pd.DataFrame(data)
corona = corona.drop(corona[corona.lat==0.0].index)
corona.reset_index(inplace=True)
corona.drop('index',axis=1)


pd.MultiIndex.from_frame(corona)

df = corona[0:107]

# grafo costruito senza ottimizzazione
G = nx.Graph()
G.add_nodes_from(corona.sigla_provincia)
G.number_of_nodes()
for i in range(107):
    for j in range(107):
        if (i != j) & (abs(df.lat[i]-df.lat[j]) <= 0.8) & (abs(df.long[i]-df.long[j]) <= 0.8):
            G.add_edge(df.sigla_provincia[i],df.sigla_provincia[j])

G.number_of_edges()

#dataset ordinati per lat e long
sorted_dfx = df.sort_values(by = 'long', kind = 'mergesort')
sorted_dfx.reset_index(inplace=True)

sorted_dfy = df.sort_values(by = 'lat', kind = 'mergesort')
sorted_dfy.reset_index(inplace=True)

#functions
def binarySearch(alist, item, prov, d):
    first = 0
    last = len(alist)-1
    found = False
    while first<=last and not found:
        midpoint = (first + last)//2
        if alist[midpoint]==float(item):
            found = True
            return list(prov[(alist>=float(alist[midpoint]-d)) & (alist<=float(alist[midpoint]+d))])
        else:
            if float(item) < alist[midpoint]:
                last = midpoint-1
            else:
                first = midpoint+1
    return []


def inter_city(A,B,elem):
    C = {}
    for el in B:
        if el!=elem and el not in C and el in A:
            C[el]=0
    return C

# metodo ottimizzato col binarySearch
P = nx.Graph()
P.add_nodes_from(sorted_dfx.sigla_provincia)

for el in sorted_dfx.sigla_provincia:
    lista_legami_x = binarySearch(sorted_dfx.long, sorted_dfx.long[sorted_dfx.sigla_provincia==el], sorted_dfx.sigla_provincia, 0.8)
    lista_legami_y = binarySearch(sorted_dfy.lat, sorted_dfy.lat[sorted_dfy.sigla_provincia==el], sorted_dfy.sigla_provincia, 0.8)
    lst_x = {j:0 for j in lista_legami_x}
    lst_y = {j:0 for j in lista_legami_y}
    città_vicine = inter_city(lst_x, lst_y, el)
    edge_city = []
    for j in città_vicine:
        edge_city.append((el, j))
    P.add_edges_from(edge_city)

P.number_of_edges()
nx.draw(P)

#Generate 2000 pairs of double (x,y) with x in [30,50) and y in [10,20).
 #   Repeat the algorithm at step 1, building a graph R using NetworkX where 
  #  each pair is a node and two nodes are connected with the same rule reported
   # above, still with d=0.08. If the algorithm at step 1 takes too long, repeat
    #step 1. Note that here d=0.08 (and not 0.8 as in the previous item), as in 
    #this way the resulting graph is sparser.
    


latit = [] 
longit = []  
nodes = [] 
for i in range(2000):
    latit.append(random.uniform(30,50))
    longit.append(random.uniform(10,20))
    nodes.append(str(i))
    
dataRandom = {'province': nodes,
              'longitudine' : longit,
              'latitudine' : latit}    

dataR = pd.DataFrame(dataRandom, columns = ['province', 'longitudine', 'latitudine'])


R = nx.Graph()
R.add_nodes_from(dataR.province)
R.number_of_nodes()

sort_dataRx= dataR.sort_values(by = 'longitudine', kind = 'mergesort')
sort_dataRx.reset_index(inplace = True)
sort_dataRy = dataR.sort_values(by = 'latitudine', kind ='mergesort')
sort_dataRy.reset_index(inplace = True)


for el in sort_dataRx.province:
    legami_x = binarySearch(sort_dataRx.longitudine, sort_dataRx.longitudine[sort_dataRx.province==el], sort_dataRx.province, 0.08)
    legami_y = binarySearch(sort_dataRy.latitudine, sort_dataRy.latitudine[sort_dataRy.province==el], sort_dataRy.province, 0.08)
    lst_x = {j:0 for j in lista_legami_x}
    lst_y = {j:0 for j in lista_legami_y}
    città_vicine = inter_city(lst_x, lst_y, el)
    edge_city = []
    for j in città_vicine:
        edge_city.append((el, j))
    R.add_edges_from(edge_city)


R.number_of_edges()
    


##### aggiungo distanza per avere un grafo pesato

for el in sorted_dfx.sigla_provincia:
    lista_legami_x = binarySearch(sorted_dfx.long, sorted_dfx.long[sorted_dfx.sigla_provincia==el], sorted_dfx.sigla_provincia, 0.8)
    lista_legami_y = binarySearch(sorted_dfy.lat, sorted_dfy.lat[sorted_dfy.sigla_provincia==el], sorted_dfy.sigla_provincia, 0.8)
    lst_x = {j:0 for j in lista_legami_x}
    lst_y = {j:0 for j in lista_legami_y}
    città_vicine = inter_city(lst_x, lst_y, el)
    edge_city = []
    for j in città_vicine:
        distanza_x = (float(sorted_dfx.long[sorted_dfx.sigla_provincia==el])-float(sorted_dfx.long[sorted_dfx.sigla_provincia==j]))**2
        distanza_y = (float(sorted_dfy.lat[sorted_dfy.sigla_provincia==el])-float(sorted_dfy.lat[sorted_dfy.sigla_provincia==j]))**2
        distanza = math.sqrt(distanza_x+distanza_y)
        edge_city.append((el, j, distanza))
    P.add_weighted_edges_from(edge_city)
               

P.number_of_edges()
nx.draw(P)


for el in sort_dataRx.province:
    legami_x = binarySearch(sort_dataRx.longitudine, sort_dataRx.longitudine[sort_dataRx.province==el], sort_dataRx.province, 0.08)
    legami_y = binarySearch(sort_dataRy.latitudine, sort_dataRy.latitudine[sort_dataRy.province==el], sort_dataRy.province, 0.08)
    lst_x = {j:0 for j in lista_legami_x}
    lst_y = {j:0 for j in lista_legami_y}
    città_vicine = inter_city(lst_x, lst_y, el)
    edge_city = []
    for j in città_vicine:
        distanza_x = ((sort_dataRx.longitudine[sort_dataRx.province==el])-(sort_dataRx.longitudine[sort_dataRx.province==j]))**2
        distanza_y = ((sort_dataRy.latitudine[sort_dataRy.province==el])-(sort_dataRy.latitudine[sort_dataRy.province==j]))**2
        distanza = math.sqrt(distanza_x+distanza_y)
        edge_city.append((el, j, distanza))
    R.add_weighted_edges_from(edge_city)
 
R.number_of_edges()



######## algoritmo Bellman-Ford 

def bell_ford(graph, source):
    
    # inizializzazione
    dist = {}
    pred = {}
    
    for v in graph.node():
        dist[v] = math.inf
        pred[v] = None
    
    dist[source] = 0
    
    #relax
    for i in range(len(graph)-1): #Run this until is converges
        for u in graph:
            for v in graph[u]: #For each neighbour of u
                if dist[v] > dist[u] + graph[u][v]['weight']:
                    # Record this lower distance
                    dist[v]  = dist[u] + graph[u][v]['weight']
                    pred[v] = u
                if dist[u] > dist[v] + graph[u][v]['weight']:
                    # Record this lower distance
                    dist[u]  = dist[v] + graph[u][v]['weight']
                    pred[u] = v
    # Step 3: check for negative-weight cycles
    for u in graph:
        for v in graph[u]:
            assert dist[v] <= dist[u] + graph[u][v]['weight']
    return dist, pred

bell_ford(P, 'OR')[0]
bell_ford(P, 'TP')[0]
bell_ford(P, 'KR')[0]
bell_ford(P, 'FI')[0]
list(nx.connected_components(P))

bell_ford(R, ('2'))[0]

##################### Closeness Centrality

def closeness(graph, u):
    dizy_dist = bell_ford(graph, u)[0]
    close = 0.0
    for key in dizy_dist:
        if dizy_dist[key] != 0 and dizy_dist[key] != math.inf:
            close = close + 1/dizy_dist[key]
    return close
closeness(P,'TP')
closeness(P,'FI')           

def closeness_norm_dist(graph, u):
    dizy_dist = bell_ford(graph, u)[0]
    total = 0
    n_short_path = 0
    cc= 0.0
    for key in dizy_dist:
        if dizy_dist[key] != 0 and dizy_dist[key] != math.inf:
            total = total + dizy_dist[key]
            n_short_path+=1
    if total > 0.0 and len(graph)>1:
        s = (n_short_path)/(len(graph)-1)
        cc = ((n_short_path)/total)*s
    return cc
closeness_norm_dist(P,'TP')
closeness_norm_dist(P,'FI')

def closeness_norm_short(graph, u):
    dizy_dist = nx.shortest_path_length(graph, u)
    total = 0
    n_short_path = 0
    cc = 0.0
    for key in dizy_dist:
        if dizy_dist[key] != 0 and dizy_dist[key] != math.inf:
            total = total + dizy_dist[key]
            n_short_path+=1
    if total > 0.0 and len(graph)>1:
        s = (n_short_path)/(len(graph)-1)
        cc = ((n_short_path)/total)*s
    return cc    

closeness_norm_short(P,'TP')
closeness_norm_short(P,'FI')
closeness_norm_short(P,'KR')

closeness_norm_short(R, '2')
nx.closeness_centrality(R)['2']

clos_wei = nx.closeness_centrality(P,distance = 'weight')['OR']
closy = nx.closeness_centrality(P)['TP']

# nodo più centrale
def massimaClosy(graph):
    maximum = 0.0
    city = ''
    for v in graph.nodes():
        closy = closeness_norm_dist(graph,v)
        if closy >= maximum:
            maximum = closy
            city = v
    return city, maximum

massimaClosy(P)

def get_key(val, my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 
    return "key doesn't exist"

dist_max = max(list(nx.closeness_centrality(P,distance = 'weight').values()))
get_key(dist_max,nx.closeness_centrality(P,distance = 'weight'))