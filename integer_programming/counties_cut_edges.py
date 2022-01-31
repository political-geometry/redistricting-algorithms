from gurobipy import *
import random
a = random.randint(0,10000000000)
import networkx as nx
from gerrychain.random import random
random.seed(a)
from gerrychain import Graph
import matplotlib.pyplot as plt
from math import sqrt,e
import os
import geopandas as gpd
import time
import csv


statelist = ['iowa','arkansas']
num_dist = {'arkansas':4, 'iowa':4}
ep_list = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]

for state in statelist:
    for ep in ep_list:
        graph_path = "./../data/"+state+".json"  

        G = Graph.from_json(graph_path)
        node_list = list(G.nodes())
        edge_list = list(G.edges())
        num_districts = num_dist[state]
        num_counties = len(node_list)
        ideal_pop = sum([G.nodes[v]["TOTPOP"] for v in G.nodes()])/num_districts
        
        # Create a new model
        m = Model("lp")

        # Create variables
        #x_ik per county x district
        x_mat = [[0]*len(node_list) for i in range(num_districts)]
        for i in range(num_districts):
            for j in range(len(node_list)):
                x_mat[i][j] = m.addVar(ub=1.0, name="x_"+str(i)+"_"+str(j))

        #w_e per edge
        w_mat = [0]*len(edge_list)
        for i in range(len(edge_list)):
            w_mat[i] = m.addVar(ub=1.0, name="w_"+str(i))

        # Set objective: 
        #linear:
        #cut edges
        obj = sum(w_mat)
        m.setObjective(obj, GRB.MINIMIZE)

        # # Add population constraints: 
        for i in range(num_districts):
            m.addConstr(sum([x_mat[i][j]*G.nodes[node_list[j]]["TOTPOP"] for j in range(len(node_list))]) >= (1-ep)*ideal_pop, "c0_"+str(i))
            m.addConstr(sum([x_mat[i][j]*G.nodes[node_list[j]]["TOTPOP"] for j in range(len(node_list))]) <= (1+ep)*ideal_pop, "c1_"+str(i))

        # Add coverage constraints:
        for j in range(len(node_list)):
            m.addConstr(sum([x_mat[i][j] for i in range(num_districts)]) ==1, "c2_"+str(j))

        for e in range(len(edge_list)):
            u_ind = node_list.index(edge_list[e][0])
            v_ind = node_list.index(edge_list[e][1])
            for i in range(num_districts):
                m.addConstr(w_mat[e] - (x_mat[i][u_ind] - x_mat[i][v_ind]) >= 0, "c3_"+str(e)+'_'+str(i))
                m.addConstr(w_mat[e] - (x_mat[i][v_ind] - x_mat[i][u_ind]) >= 0, "c4_"+str(e)+'_'+str(i))

        for i in range(num_districts):
            for j in range(len(node_list)):
                x_mat[i][j].vType = GRB.INTEGER

        m.optimize()

        dist_assign = [0]*len(node_list)
        node_assign = {x:num_districts+1 for x in G.nodes()}
        for i in range(num_districts):
            for j in range(len(node_list)):
                dist_assign[j] += i*x_mat[i][j].x
                node_assign[node_list[j]] += i*x_mat[i][j].x

        pops = []
        for i in range(num_districts):
            pops.append(sum([x_mat[i][j].x*G.nodes[node_list[j]]["TOTPOP"] for j in range(len(node_list))]))

        out_row = [time.time(), state, ep, num_districts, obj.getValue(), max(pops)-min(pops), dist_assign]
        with open("cut_edges_state.txt", 'a+') as cut_edges_file:
            writer = csv.writer(cut_edges_file)
            writer.writerow(out_row)
