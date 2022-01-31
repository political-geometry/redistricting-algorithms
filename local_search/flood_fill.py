"""
graph flooding algorithms:

"""


import random
a = random.randint(0,10000000000)
import networkx as nx
from gerrychain.random import random
random.seed(a)
from gerrychain import Graph, Partition
from gerrychain.updaters import Tally, cut_edges
import matplotlib.pyplot as plt
import matplotlib
import operator
import numpy as np
from math import sqrt, floor
import math
import os
import csv
import json
import geopandas as gpd

rand_colors = ['blue', 'orange', 'green', 'red', 'yellow','indigo', 'darkorange', 'yellowgreen', 'saddlebrown', 'pink', 'dimgray', 'cornflowerblue', 'cyan', 'gainsboro']

def draw_graph(G, num_districts, pos_map, plan_assignment):
    col_map = {v:plan_assignment[v] if v in plan_assignment.keys() else num_districts for v in G.nodes()}

    plt.figure()
    nx.draw(G, pos = {x: pos_map[x] for x in G.nodes()}, node_size = 50, width = 2, node_color=[col_map[v] for v in G.nodes()], cmap = matplotlib.colors.ListedColormap(rand_colors))
    plt.show()

def neutral_flood_fill(G, ideal_pop, pop_map, num_districts, ep, pos_map, viz):

    district_counter = 0    
    processed_nodes = set()
    plan_assignment = {}

    seed = random.choice(list(G.nodes()))
    plan_assignment[seed] = district_counter

    local_frontier = set(G.neighbors(seed))
    global_frontier = set(G.neighbors(seed))
    
    processed_nodes.add(seed)
    cur_dist_pop = pop_map[seed]
    if viz:
        draw_graph(G, num_districts, pos_map, plan_assignment)
    
    unassigned_pop = ideal_pop*num_districts


    while district_counter < num_districts:
        # counter = 0
        ideal_pop = (unassigned_pop)/(num_districts-district_counter)
        dist_complete = False
        while not dist_complete:
            assert(cur_dist_pop <= (1+ep)*ideal_pop)
            if len(local_frontier) > 0:
                spread = random.choice(list(local_frontier))
                if cur_dist_pop + pop_map[spread] > (1+ep)*ideal_pop:
                    local_frontier.remove(spread)
                else:
                    processed_nodes.add(spread)
                    plan_assignment[spread] = district_counter
                    cur_dist_pop += pop_map[spread]

                    local_frontier = local_frontier.union(set(G.neighbors(spread))).difference(processed_nodes)
                    global_frontier = global_frontier.union(set(G.neighbors(spread))).difference(processed_nodes)
                    if viz:
                        draw_graph(G, num_districts, pos_map, plan_assignment)
            if len(local_frontier) == 0 and cur_dist_pop < (1-ep)*ideal_pop:
                return False, plan_assignment
            if len(local_frontier) == 0 and cur_dist_pop >= (1-ep)*ideal_pop:
                dist_complete = True
            if cur_dist_pop >= ideal_pop:
                dist_complete = True

        unassigned_pop -= cur_dist_pop
        district_counter += 1

        if district_counter >= num_districts and unassigned_pop > 0:
            return False, plan_assignment
        
        elif district_counter >= num_districts:
            return True, plan_assignment

        if len(global_frontier) == 0:
            return False, plan_assignment
        
        seed = random.choice(list(global_frontier))
        plan_assignment[seed] = district_counter
        processed_nodes.add(seed)
        global_frontier = global_frontier.union(set(G.neighbors(seed))).difference(processed_nodes)
        local_frontier = set(G.neighbors(seed)).difference(processed_nodes)
        cur_dist_pop = pop_map[seed]
        if viz:
            draw_graph(G, num_districts, pos_map, plan_assignment)

