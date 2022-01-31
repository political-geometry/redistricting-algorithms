import random
a = random.randint(0,10000000000)
import networkx as nx
from gerrychain.random import random
random.seed(a)
from gerrychain import Graph
import os
import csv
import time
import shapefile
from shapely.geometry import shape

def power_diagram_plans(G, num_districts, ideal_pop, ep, centroid_dict, input_file):
    os.system("./do_redistrict " + str(num_districts)+" "+ input_file+ " > pd_temp_out.txt 2>error")
    file = open('pd_temp_out.txt','r')
    linenum = 1
    node_assign = {}
    pop_assign = {x:0 for x in range(num_districts)}
    for line in file:
        if linenum == 1:
            line_one = line
        if linenum > 5:
            line_items = line.split(" ")
            if len(line_items) == 4:
                node_assign[centroid_dict[(round(float(line_items[0])), round(float(line_items[1])))]] = int(line_items[2])
                pop_assign[int(line_items[2])] += float(line_items[3])
            elif len(line_items) > 4:
                max_pop = 0
                max_ind = 0
                pop_sum = 0
                for i in range(round(len(line_items[2:])/2)):
                    pop_sum += float(line_items[2+2*i+1])
                    if float(line_items[2+2*i+1]) > max_pop:
                        max_pop = float(line_items[2+2*i+1])
                        max_ind = 2*i+2
                node_assign[centroid_dict[(round(float(line_items[0])), round(float(line_items[1])))]] = int(line_items[max_ind])
                pop_assign[int(line_items[max_ind])] += pop_sum
        linenum += 1
    pop_error = max([abs(pop_assign[x]-ideal_pop) for x in range(num_districts)])/ideal_pop
    if pop_error > ep:
        return False, node_assign
    for i in range(num_districts):
        node_list = [v for v in G.nodes() if node_assign[v] == i]
        G2 = G.subgraph(node_list)
        if not nx.is_connected(G2):
            return False, node_assign
    return True, node_assign



centroid_dict = {}
pop_dict = {}
output_path = "./output/"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

for graph_name in {'iowa','grid10'}:

    if graph_name == 'iowa':
        filename = './census_2010_county_DP1/census_2010_county_DP1.shp'
        graph = Graph.from_file(filename)
        pop_map = {v:graph.nodes[v]['DP0010001'] for v in graph.nodes()}
        pos_map = {v:(float(graph.nodes[v]['INTPTLON10']),float(graph.nodes[v]['INTPTLAT10'])) for v in graph.nodes()}
        ideal_pop = sum([graph.nodes[v]['DP0010001'] for v in graph.nodes()])/4
        geoid_map  = {graph.nodes[v]["GEOID10"]:v for v in graph.nodes()}

        drop_list = []
        for v in graph.nodes():
            if graph.nodes[v]['DP0010001']<1:
                drop_list.append(v)
        for v in drop_list:
            graph.remove_node(v)

        sf = shapefile.Reader(filename)
        for shape_rec in sf.iterShapeRecords():
            pop = shape_rec.record[7]
            geoid = shape_rec.record[0]
            if pop > 0:
                cent = shape(shape_rec.shape).centroid
                centroid_dict[(round(cent.x),round(cent.y))] = geoid_map[geoid]
        input_file = 'read_iowa_counties.txt'
        out_name = 'iowa_counties'

    if graph_name == 'grid10':
        graph = nx.grid_graph([10,10])
        pop_map = {v:1 for v in graph.nodes()}
        pos_map= {v:v for v in graph.nodes()}
        centroid_dict = {(v[0],v[1]): v for v in graph.nodes()}
        ideal_pop = 250
        input_file = 'read_10x10_grid.txt'
        out_name = 'grid10'


    num_dist = 4
    num_runs = 100000
    ep = 0.05

    success, plan = power_diagram_plans(graph, num_dist, ideal_pop, ep, centroid_dict, input_file)

    cut_length = []
    successes = 0
    attempts = 0
    plans_found = []
    for j in range(num_runs):
        if j%1000 == 0:
            print(j, attempts)           
        success = False
        while not success:
            success, plan = power_diagram_plans(graph, num_dist, ideal_pop, ep, centroid_dict, input_file)
            attempts += 1
        successes += 1
        plans_found.append(plan)

    assert len(plans_found) == successes

    out_row = [time.time(), num_runs, ep, successes, successes/attempts]
    with open(output_path + out_name+"_power_diagram_success_rates.txt", 'a+') as success_file:
        writer = csv.writer(success_file)
        writer.writerow(out_row)

    cut_freq = {x:0 for x in graph.edges()}

    for plan in plans_found:
        cut_edges = [e for e in graph.edges() if plan[e[0]] != plan[e[1]]]
        cut_length.append(len(cut_edges))
        for e in cut_edges:
            cut_freq[e] += 1

    out_row = [time.time(), num_runs, ep, cut_freq]
    with open(output_path +  out_name+"_power_diagram_cut_freq.txt", 'a+') as cut_freq_file:
        writer = csv.writer(cut_freq_file)
        writer.writerow(out_row)

    out_row = [time.time(), len(cut_length), ep, cut_length]
    with open(output_path +  out_name+"_power_diagram_cut_length.txt", 'a+') as cut_len_file:
        writer = csv.writer(cut_len_file)
        writer.writerow(out_row)
