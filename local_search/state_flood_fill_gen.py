import flood_fill
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
import time

def generate_state(G, num_districts, ep, max_tries):
    pop_map = {v:G.nodes[v]['TOTPOP'] for v in G.nodes()}
    pos_map = {v:(float(G.nodes[v]['INTPTLON10']),float(G.nodes[v]['INTPTLAT10'])) for v in G.nodes()}
    ideal_pop = sum([G.nodes[v]['TOTPOP'] for v in G.nodes()])/num_districts

    bound_nodes = []
    for v in G.nodes():
        if G.nodes[v]['boundary_node']:
            bound_nodes.append(v)

    for i in range(max_tries):
        success, plan =  flood_fill.neutral_flood_fill(G, ideal_pop, pop_map, num_districts, ep, pos_map, False)
        if success:
            break
    return success, plan
