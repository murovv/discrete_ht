#!/usr/bin/env sage -python
import python_tsp.utils
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy
from networkx.algorithms.approximation.independent_set import maximum_independent_set
from networkx.algorithms.approximation import min_weighted_vertex_cover
import postman_problems.solver as magic
from postman_problems.stats import calculate_postman_solution_stats
from randomized_tsp.tsp import tsp
from haversine import haversine as eDist
from networkx.algorithms import tree
import numpy as np
import dwave_networkx as dnx
europe_countries_list = list()
mates = list()
e = []
v = []
def formatAndRead():
    inp = open("countries.txt", "r")
    outp = open("formated.txt", "w")
    lines = inp.readlines()
    for line in lines:
        if ' ' != line[0]:
            str = line.split(':')[0].split('[')[0].split('(')[0].strip(' ').rstrip('\n')
            if not (str in europe_countries_list):
                europe_countries_list.append(str)
    i=-1
    global mates
    mates = [[] for i in range (len(europe_countries_list))]
    push = []
    for line in lines:
        if line[0] == ' ':
            next_mate = line.split(':')[0].split('[')[0].split('(')[0].strip(' ')
            if next_mate in europe_countries_list:
                outp.write(" " + next_mate + "\n")
                push.append(next_mate)
            #else:
                #print("deleted:" + next_mate)
        else:
            if(i!=-1):
                mates[i]=push.copy()
            i+=1
            push = []
            outp.write(line)
    mates[i] = push.copy()
    return
def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)
def generate_csv():
    outp = open("table.csv","w")
    global v
    global mates
    outp.write("node1,node2,distance\n")
    for i,j in e:
        outp.write(i+','+j+',1\n')

formatAndRead()
gStar = nx.Graph()
gStar.add_nodes_from(europe_countries_list)
for i in range(len(europe_countries_list)):
    for mate in mates[i]:
        gStar.add_edge(europe_countries_list[i],mate)
pos = nx.planar_layout(gStar,scale=10)
nx.draw(gStar,pos = pos,with_labels=True)
plt.show()

v = max(nx.connected_components(gStar),key=len)
g = nx.subgraph(gStar,v)
pos = nx.planar_layout(g,scale=10)
nx.draw(g,pos = pos,with_labels=True)
plt.show()

#B
print("|V| =",len(v))
e = g.edges
print("|E| =",len(e))
delta = min([val for (node, val) in g.degree()])
print("delta =",delta)
DELTA = max([val for (node, val) in g.degree()])
print("DELTA =",DELTA)
rad = nx.radius(g)
print("rad =",rad)
diam = nx.diameter(g)
print("diam =",diam)
girth = 3#norway-sweden-finland
print("girth =",girth)
center = nx.center(g)
print("center =",center)
vertex_connectivity = nx.node_connectivity(g)
print("k =",vertex_connectivity)
edge_connectivity = nx.edge_connectivity(g)
print("labda =",edge_connectivity)
#C
node_color = []
vertex_coloring = nx.greedy_color(g,strategy='DSATUR')
while max([ vertex_coloring[key] for key in vertex_coloring.keys()] )>3:
    vertex_coloring = nx.greedy_color(g, strategy='DSATUR')
for node in g.nodes:
    print(node)
    if(vertex_coloring[node]==0):
        node_color.append("red")
    elif (vertex_coloring[node] == 1):
        node_color.append("orange")
    elif (vertex_coloring[node] == 2):
        node_color.append("yellow")
    elif (vertex_coloring[node] == 3):
        node_color.append("green")
    else:
        print("ALARM",vertex_coloring[node])
print(len(node_color))
pos = nx.planar_layout(g)
nx.draw(g, pos = pos,with_labels=True, node_color=node_color)
plt.show()
plt.clf()
print("vertex colorinng:",vertex_coloring)
#E
max_clique = nx.algorithms.approximation.max_clique(g)
print("max clique:", max_clique)
#F
maximum_stable_set = maximum_independent_set(g)
print("maximum stable set:",maximum_stable_set)
#G
maximal_matching = nx.max_weight_matching(g,maxcardinality=1)
print("maximum matching:",maximal_matching)
#H
min_vertex_cover = min_weighted_vertex_cover(g)
print("minimum vertex cover:",min_vertex_cover)
#I
min_edge_cover = nx.min_edge_cover(g)
print("minimum edge cover:",min_edge_cover)
#J
v_list = list(g.nodes)
Matr = numpy.asarray(nx.to_numpy_matrix(g))
for i in range(len(Matr)):
    for j in range(len(Matr[i])):
        if(Matr[i][j]==0):
            Matr[i][j]=len(nx.shortest_path(g,str(v_list[i]),str(v_list[j])))
        else:
            Matr[i][j]=int(1)
numpy.set_printoptions(threshold= sys.maxsize)
lv_circuit = [0 for i in range(100)]
tsp_obj = tsp(Matr)
for iter in range(1):

    permutation, _cost = tsp_obj.ant_colony(num_of_ants=10,pheromone_evapouration=0.2)
    v_circuit = []

    for i in range(len(permutation)-1):
        app = nx.shortest_path(g,str(v_list[permutation[i]]),str(v_list[permutation[i+1]]))
        v_circuit.extend(app[1:])
    if(len(v_circuit)<len(lv_circuit)):
        lv_circuit = v_circuit
print("vertex circuit:",lv_circuit)
#K
#здесь нужнен networkx версии 2.0, который не поддерживает planar_layout
#e_circuit,graph = magic.cpp(edgelist_filename="/home/paperblade/PycharmProjects/pythonProject4/table.csv")
#print("edge circuit:",e_circuit)

#L
two_vertex_connected_nodes = list(nx.biconnected_components(gStar))
articulation_points = list(nx.articulation_points(gStar))
print("art_points",articulation_points)
block_cut_tree = nx.Graph()
block_cut_tree.add_nodes_from(articulation_points)
int_block_mapping = {}
it = 0
for i in two_vertex_connected_nodes:
    int_block_mapping[it] = i
    block_cut_tree.add_node(it)
    it+=1

for art_point in articulation_points:
    for key in int_block_mapping.keys():
        if art_point in int_block_mapping[key]:
            block_cut_tree.add_edge(art_point,key)
plt.clf()
pos = nx.planar_layout(block_cut_tree,scale=10)
nx.draw(block_cut_tree,pos = pos,with_labels=True)
plt.show()
print("Legend for blocks:")
for key in int_block_mapping.keys():
    print(key,":",int_block_mapping[key])
#M
two_edge_connected_nodes = nx.k_edge_components(gStar,2)
print("2-edge-connected components:",list(two_edge_connected_nodes))
#N
#отдельный файл spqr.py
#O
coords = dict();
coords_file = open("coords.txt","r").readlines()
it = 0
while(it<len(coords_file)):
    name = coords_file[it].strip('\n').strip(':')
    it+=1
    lat = float(coords_file[it].strip(' ').strip('\n'))
    it+=1
    lon = float(coords_file[it].strip(' ').strip('\n'))
    it+=1
    coords[name] = (lat, lon)
for start,end in g.edges:
    g[start][end]['weight'] = round(eDist(coords[start],coords[end])*10)/10
mst = tree.minimum_spanning_tree(g, algorithm="kruskal")
edgelist = list(mst)
plt.clf()
pos = nx.planar_layout(mst,scale=10)
labels = nx.get_edge_attributes(mst,'weight')
nx.draw(mst,pos = pos,with_labels=True)
#nx.draw_networkx_edge_labels(mst,pos,edge_labels=labels)

plt.show()
#P
weights = {}
for country in mst.nodes():
    mst_copy = mst.copy()
    mst_copy.remove_node(country)
    branches = nx.connected_components(mst_copy)
    max_weight = 0
    for branch in branches:
        this_weight=0
        for u in branch:
            for v in branch:
                e = (u,v)
                if( e in mst.edges()):
                        this_weight+=mst[u][v]["weight"]
        max_weight = max(max_weight,this_weight)
    weights[country] = max_weight/2
minK = 100000000
min = []
for key in weights.keys():
    if(weights[key]<minK):
        min.clear()
        minK = weights[key]
        min.append(key)
    elif(weights[key]==minK):
        min.append(key)

print("centroid:",min)
mapping = {}
it = 0
for i in mst.nodes:
    mapping[i] = it
    it+=1
mst = nx.relabel_nodes(mst,mapping)
print("Prufer code:",nx.to_prufer_sequence(mst))
print("Legend:",mapping)
