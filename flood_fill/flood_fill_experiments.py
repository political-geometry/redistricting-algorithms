import flood_fill
import random
a = random.randint(0,10000000000)
import networkx as nx
from gerrychain.random import random
from gerrychain import Graph
random.seed(a)
import csv
import time
import os
import math


def flood_fill_experiment(G, ep, num_dist, run_type, num_runs, pop_map, pos_map, ideal_pop, bounding_boxes, bounding_box, seeded, seed_type, seed_nodes, data_dir):
    cut_length = []

    for i in range(3):
        successes = 0
        plans_found = []
        for j in range(num_runs):
            if j%math.floor(num_runs/10) == 0:
                print(j, successes)
            if bounding_box:
                success, plan =  flood_fill.bounding_box_flood_fill(G, ideal_pop, pop_map, num_dist, ep, bounding_boxes, pos_map,False)
            elif seeded:
                if seed_type == 'bound' or seed_type == 'rand':
                    seeds = random.sample(seed_nodes, num_dist)
                elif seed_type == 'zone':
                    assert(len(seed_nodes)==num_dist)
                    seeds = [random.choice(zone) for zone in seed_nodes]
                else:
                    print('INVALID SEED TYPE')
                    assert(False)
                success, plan = flood_fill.seeded_neutral_flood_fill(G, seeds, ideal_pop, pop_map, num_dist, ep, pos_map, False, False)
            else:
                success, plan =  flood_fill.neutral_flood_fill(G, ideal_pop, pop_map, num_dist, ep, pos_map,False)
            if success:
                successes += 1
                plans_found.append(plan)

        assert len(plans_found) == successes
        
        out_row = [time.time(), num_runs, ep, run_type, num_dist, bounding_box, successes, successes/num_runs]
        with open(data_dir + "success_rates.txt", 'a+') as success_file:
            writer = csv.writer(success_file)
            writer.writerow(out_row)
        
        cut_freq = {x:0 for x in G.edges()}

        for plan in plans_found:
            cut_edges = [e for e in G.edges() if plan[e[0]] != plan[e[1]]]
            cut_length.append(len(cut_edges))
            for e in cut_edges:
                cut_freq[e] += 1
        
        out_row = [time.time(), num_runs, ep, successes, run_type, num_dist, bounding_box, cut_freq]
        with open(data_dir + "cut_freq.txt", 'a+') as cut_freq_file:
            writer = csv.writer(cut_freq_file)
            writer.writerow(out_row)
    
    out_row = [time.time(), len(cut_length), ep, num_dist, bounding_box, cut_length]
    with open(data_dir + "cut_length.txt", 'a+') as cut_len_file:
        writer = csv.writer(cut_len_file)
        writer.writerow(out_row)


output_path = "./output/"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
num_dist = 4
num_runs = 100000
            
iowa = Graph.from_json('./../data/iowa.json')
iowa_pop_map = {v:iowa.nodes[v]['TOTPOP'] for v in iowa.nodes()}
iowa_pos_map = {v:(float(iowa.nodes[v]['INTPTLON10']),float(iowa.nodes[v]['INTPTLAT10'])) for v in iowa.nodes()}
iowa_ideal_pop = sum([iowa.nodes[v]['TOTPOP'] for v in iowa.nodes()])/4
iowa_bounding_boxes = {}
node_dict = {}

for v in iowa.nodes():
    node_dict[iowa.nodes[v]['COUNTY']] = v
with open('./../data/iowa_bb.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        iowa_bounding_boxes[node_dict[row[0]]] = [row[1],row[2],row[3],row[4]]


grid6 = nx.grid_graph([6,6])
grid6_pop_map = {v:1 for v in grid6.nodes()}
grid6_pos_map= {v:v for v in grid6.nodes()}
grid6_ideal_pop = 9
grid6_bounding_boxes = {v:[v[0],v[0],v[1],v[1]] for v in grid6.nodes()}

grid10 = nx.grid_graph([10,10])
grid10_pop_map = {v:1 for v in grid10.nodes()}
grid10_pos_map= {v:v for v in grid10.nodes()}
grid10_ideal_pop = 25
grid10_bounding_boxes = {v:[v[0],v[0],v[1],v[1]] for v in grid10.nodes()}


iowa_zones = [[] for i in range(4)]
for v in iowa.nodes():
    if float(iowa.nodes[v]['INTPTLAT10']) < 42.05:
        if float(iowa.nodes[v]['INTPTLON10']) < -93.5:
            iowa_zones[0].append(v)
        else:
            iowa_zones[1].append(v)
    else:
        if float(iowa.nodes[v]['INTPTLON10']) < -93.5:
            iowa_zones[2].append(v)
        else:
            iowa_zones[3].append(v)

iowa_bound_nodes = []
for v in iowa.nodes():
    if iowa.nodes[v]['boundary_node']:
        iowa_bound_nodes.append(v)



grid6_zones = [[] for i in range(4)]
for v in grid6.nodes():
    if v[0] < 3:
        if v[1] < 3:
            grid6_zones[0].append(v)
        else:
            grid6_zones[1].append(v)
    else:
        if v[1] < 3:
            grid6_zones[2].append(v)
        else:
            grid6_zones[3].append(v)

grid6_bound_nodes = []
for v in grid6.nodes():
    if len(list(grid6.neighbors(v))) < 4:
        grid6_bound_nodes.append(v)


grid10_zones = [[] for i in range(4)]
for v in grid10.nodes():
    if v[0] < 5:
        if v[1] < 5:
            grid10_zones[0].append(v)
        else:
            grid10_zones[1].append(v)
    else:
        if v[1] < 5:
            grid10_zones[2].append(v)
        else:
            grid10_zones[3].append(v)

grid10_bound_nodes = []
for v in grid10.nodes():
    if len(list(grid10.neighbors(v))) < 4:
        grid10_bound_nodes.append(v)

flood_fill_experiment(grid6, 0, num_dist, 'grid6_std', num_runs, grid6_pop_map, grid6_pos_map, grid6_ideal_pop, None, False, False, None, None, output_path)
flood_fill_experiment(grid6, 0, num_dist, 'grid6_bb', num_runs, grid6_pop_map, grid6_pos_map, grid6_ideal_pop, grid6_bounding_boxes, True, False, None, None, output_path)
flood_fill_experiment(grid6, 0, num_dist, 'grid6_randseed', num_runs, grid6_pop_map, grid6_pos_map, grid6_ideal_pop, None, False, True, 'rand', grid6.nodes(), output_path)
flood_fill_experiment(grid6, 0, num_dist, 'grid6_boundseed', num_runs, grid6_pop_map, grid6_pos_map, grid6_ideal_pop, None, False, True, 'bound', grid6_bound_nodes, output_path)
flood_fill_experiment(grid6, 0, num_dist, 'grid6_zoneseed', num_runs, grid6_pop_map, grid6_pos_map, grid6_ideal_pop, None, False, True, 'zone', grid6_zones, output_path)

flood_fill_experiment(grid10, 0.05, num_dist, 'grid10_std', num_runs, grid10_pop_map, grid10_pos_map, grid10_ideal_pop, None, False, False, None, None, output_path)
flood_fill_experiment(grid10, 0.05, num_dist, 'grid10_bb', num_runs, grid10_pop_map, grid10_pos_map, grid10_ideal_pop, grid10_bounding_boxes, True, False, None, None, output_path)
flood_fill_experiment(iowa, 0.05, num_dist, 'iowa_std', num_runs, iowa_pop_map, iowa_pos_map, iowa_ideal_pop, None, False, False, None, None, output_path)
flood_fill_experiment(iowa, 0.05, num_dist, 'iowa_bb', num_runs, iowa_pop_map, iowa_pos_map, iowa_ideal_pop, iowa_bounding_boxes, True, False, None, None, output_path)
