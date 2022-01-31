# Import for I/O

import random
a = random.randint(0,10000000000)
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
import time
import csv
from gerrychain.random import random
random.seed(a)
from gerrychain import MarkovChain, Graph
from gerrychain.constraints import (
    Validator,
    single_flip_contiguous,
    within_percent_of_ideal_population,
)
from gerrychain.updaters import Tally, cut_edges
from gerrychain.partition import Partition
import networkx as nx
from state_flood_fill_gen import generate_state
import os

####################################################################################


def hill_climb(partition):
    bound = 1
    if partition.parent is not None:
        if len(partition.parent['cut_edges']) < len(partition['cut_edges']):
            bound = 0
    
    return random.random() < bound
        
        
def anneal(partition):
    tmax = 1.5
    tmin = .005
    
    Temp = tmax-((tmax-tmin)/num_steps)*t

    bound = 1
    if partition.parent is not None:
        if len(partition.parent['cut_edges']) < len(partition['cut_edges']): 
            bound = np.e**((len(partition.parent['cut_edges'])-len(partition['cut_edges']))/Temp)
    
    return random.random() < bound        
    
##########################################################    

def slow_reversible_propose(partition):
    """Proposes a random boundary flip from the partition in a reversible fasion
    by selecting a boundary node at random and uniformly picking one of its
    neighboring parts.
    Temporary version until we make an updater for this set.
    :param partition: The current partition to propose a flip from.
    :return: a proposed next `~gerrychain.Partition`
    """

    b_nodes = {x[0] for x in partition["cut_edges"]}.union(
        {x[1] for x in partition["cut_edges"]}
    )

    flip = random.choice(list(b_nodes))
    neighbor_assignments = list(
        set(
            [
                partition.assignment[neighbor]
                for neighbor in partition.graph.neighbors(flip)
            ]+[partition.assignment[flip]]
        )
    )
    neighbor_assignments.remove(partition.assignment[flip])

    flips = {flip: random.choice(neighbor_assignments)}

    return partition.flip(flips)


def reversible_propose(partition):
    boundaries1 = {x[0] for x in partition["cut_edges"]}.union(
        {x[1] for x in partition["cut_edges"]}
    )

    flip = random.choice(list(boundaries1))
    return partition.flip({flip: -partition.assignment[flip]})

###############################################################
def common_refinement(partition1, partition2):
    graph_pared = partition1.graph.copy()
    graph_pared.remove_edges_from(partition1["cut_edges"].union(partition2["cut_edges"]))
    refine_dict = {}
    counter = 0
    for i in nx.connected_components(graph_pared):
        for v in list(i):
            refine_dict[v] = counter
        counter += 1
    return Partition(partition1.graph, refine_dict, partition1.updaters)
    

def merge_parts(partition, k):
    #until there are k parts, merge adjacent parts with smallest sum
    assert (len(partition.parts) >= k)
    while len(partition.parts) > k:
        min_pair = (partition.assignment[list(partition["cut_edges"])[0][0]],partition.assignment[list(partition["cut_edges"])[0][1]])
        min_sum = math.inf
        for e in partition["cut_edges"]:
            edge_sum = partition["population"][partition.assignment[e[0]]] + partition["population"][partition.assignment[e[1]]]
            if  edge_sum < min_sum:                
                min_sum = edge_sum
                min_pair = (partition.assignment[e[0]], partition.assignment[e[1]])
        merge_dict = {v:min_pair[0] if partition.assignment[v] == min_pair[1] else partition.assignment[v] for v in partition.graph.nodes()}
        partition = Partition(partition.graph, merge_dict, partition.updaters)
    keyshift = dict(zip(list(partition.parts.keys()), range(len(partition.parts.keys()))))
    keydict = {v:keyshift[partition.assignment[v]] for v in partition.graph.nodes()}
    return Partition(partition.graph, keydict, partition.updaters)

def shift_pop(partition, k, ep, rep_max):
    counter = 0
    tot_pop = sum([dict(partition["population"])[i] for i in dict(partition["population"]).keys()])
    ideal_pop = tot_pop/len(partition)
    while max([abs(partition["population"][i]-ideal_pop) for i in dict(partition["population"]).keys()]) > ep*ideal_pop and counter < rep_max:
        best_pair = list(partition["cut_edges"])[0]
        best_score = 0
        for e in partition["cut_edges"]:
            score =  abs(partition["population"][partition.assignment[e[0]]] - partition["population"][partition.assignment[e[1]]])
            if max([abs(partition["population"][partition.assignment[e[i]]]-ideal_pop) for i in [0,1]]) > ep*ideal_pop and score > best_score:
                if partition["population"][partition.assignment[e[0]]]> partition["population"][partition.assignment[e[1]]]:
                    max_part = 0
                    min_part = 1
                else:
                    max_part = 1
                    min_part = 0
                if partition["population"][partition.assignment[e[max_part]]]- ideal_pop > ep*ideal_pop:
                    max_high = True
                else:
                    max_high = False
                if partition["population"][partition.assignment[e[min_part]]]- ideal_pop > ep*ideal_pop:
                    min_high = True
                else:
                    min_high = False
                if ideal_pop - partition["population"][partition.assignment[e[max_part]]] > ep*ideal_pop:
                    max_low = True
                else:
                    max_low = False
                if ideal_pop - partition["population"][partition.assignment[e[min_part]]] > ep*ideal_pop:
                    min_low = True
                else:
                    min_low = False
                val = partition.graph.nodes[e[max_part]]["TOTPOP"]
                # print(e, score, best_score, max_high, min_high, max_low, min_low, val)
                if (min_low and not max_low) or (max_high and not min_high):
                    if partition["population"][partition.assignment[e[max_part]]]-val > (1-ep)*ideal_pop and partition["population"][partition.assignment[e[min_part]]]+val < (1+ep)*ideal_pop:
                        subg = partition.graph.subgraph(partition.parts[partition.assignment[e[max_part]]]).copy()
                        subg.remove_node(e[max_part])
                        if nx.is_connected(subg):
                            best_pair = (e[max_part], e[min_part])
                            best_score = score
        if best_score == 0:
            break
        else:
            shift_dict = {v:partition.assignment[best_pair[1]] if v == best_pair[0] else partition.assignment[v] for v in partition.graph.nodes()}
        partition = Partition(partition.graph, shift_dict, partition.updaters)
        counter += 1
    
    return partition 


def record_partition(partition, t, run_type):
    with open('./output/'+ graph_name+".txt", 'a+') as partition_file:
        writer = csv.writer(partition_file)
        out_row = [time.time(), graph_name, run_type, t, len(graph.nodes())]+[partition.assignment[x] for x in graph.nodes()]
        writer.writerow(out_row)   

####################################################################

k = 4
ns = 200
p = 0.4
exp_num=1
pop_bal=1
num_steps = 100000
max_adjust = 10000
prob_crossover = 0.01
print_step = 10000
population_size = 10

graph_name = 'iowa'
graph_path = './../data/'+graph_name+'.json'

plot_path = './opt_plots/'
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
output_path = './output/'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

graph = Graph.from_json(graph_path)
node_list = list(graph.nodes())
edge_list = list(graph.edges())
num_districts = k
num_counties = len(node_list)
ideal_pop = sum([graph.nodes[v]["TOTPOP"] for v in graph.nodes()])/num_districts
ep = 0.05

#IOWA
init_dists = {88: 0, 75: 0, 98: 0, 82: 0, 45: 0, 29: 0, 33: 0, 37: 0, 96: 0, 80: 0, 78: 0, 40: 0, 81: 0, 63: 0, 83: 0, 1: 0, 74: 0, 0: 0, 62: 0, 4: 0, 86: 0, 27: 0, 6: 0, 52: 0, 89: 0, 11: 0, 91: 0, 15: 0, 23: 0, 31: 0, 85: 0, 59: 0, 5: 1, 10: 1, 38: 1, 8: 1, 92: 1, 16: 1, 24: 1, 53: 1, 76: 1, 94: 1, 28: 1, 35: 1, 34: 1, 51: 2, 93: 2, 54: 2, 95: 2, 25: 2, 73: 2, 22: 2, 47: 2, 71: 2, 41: 2, 60: 3, 17: 3, 12: 3, 30: 3, 65: 3, 79: 3, 36: 3, 50: 3, 56: 3, 58: 3, 49: 3, 18: 3, 9: 3, 7: 3, 87: 3, 90: 3, 44: 3, 77: 3, 13: 3, 14: 3, 66: 3, 42: 3, 20: 3, 69: 3, 55: 3, 70: 3, 46: 3, 19: 3, 61: 3, 2: 3, 67: 3, 97: 3, 43: 3, 72: 3, 26: 3, 39: 3, 84: 3, 32: 3, 64: 3, 21: 3, 57: 3, 3: 3, 48: 3, 68: 3}
cddict = {node_list[i]:int(init_dists[i]) for i in range(len(node_list))}

# Necessary updaters go here
updaters = {
    "population": Tally("TOTPOP", alias="population"),
    "cut_edges": cut_edges,
}


starting_partition = Partition(graph, assignment=cddict, updaters=updaters)

pos ={x:(float(graph.nodes[x]["INTPTLON10"]),float(graph.nodes[x]["INTPTLAT10"])) for x in graph.nodes()}

popbound = within_percent_of_ideal_population(starting_partition, ep)
record_partition(starting_partition, 0, 'starting')

##############################HILLCLIMBING##############################
gchain = MarkovChain(
    slow_reversible_propose,  # propose_random_flip,#propose_chunk_flip, # ,
    Validator([single_flip_contiguous, popbound]),
    accept=hill_climb,  # aca,#cut_accept,#always_accept,#
    initial_state=starting_partition,
    total_steps=num_steps,
)

ce_hist = []
t = 0
cuts = []
for part in gchain:
    cuts.append(len(part["cut_edges"]))
    t += 1
    if t % print_step == 0:
        print(t)
        record_partition(part, t, 'mid_hill')

print("finished hill")
record_partition(part, t, 'end_hill')
part_hill = part

all_cuts=[cuts[:]]

##############################Anneal##############################
gchain = MarkovChain(
    slow_reversible_propose,  # propose_random_flip,#propose_chunk_flip, # ,
    Validator([single_flip_contiguous, popbound]),
    accept=anneal,  # aca,#cut_accept,#always_accept,#
    initial_state=starting_partition,
    total_steps=num_steps,
)

ce_hist = []
t = 0
cuts = []
for part in gchain:
    cuts.append(len(part["cut_edges"]))
    t += 1
    if t % print_step == 0:
        print(t)
        record_partition(part, t, 'mid_anneal')


print("finished annealing")
record_partition(part, t, 'end_anneal')
part_anneal = part

all_cuts.append(cuts[:])

##############################Evolutionary##############################
partition_list = [starting_partition]
for i in range(population_size-1):
    #flood fill:
    success = False
    while not success:
        success, new_plan = generate_state(graph, k, ep, 10000)
    partition = Partition(graph, assignment=new_plan, updaters=updaters)
    for i in partition.parts.keys():
        subg = partition.graph.subgraph(partition.parts[i]).copy()
        assert(nx.is_connected(subg))
    partition_list.append(partition)

for p in partition_list:
    record_partition(p, 0, 'begin_evol')

cur_partition = 0
gchain = MarkovChain(
    slow_reversible_propose,  # propose_random_flip,#propose_chunk_flip, # ,
    Validator([single_flip_contiguous, popbound]),
    accept=hill_climb,  # aca,#cut_accept,#always_accept,#
    initial_state=starting_partition,
    total_steps=num_steps,
)

ce_hist = []
t = 0
cuts_min = []
cuts_max = []
for part in gchain:
    cuts_min.append(min([len(p["cut_edges"]) for p in partition_list]))
    cuts_max.append(max([len(p["cut_edges"]) for p in partition_list]))
    c_dicts = [{list(dict(partition_list[j]["population"]).keys())[i]:i for i in range(len(dict(partition_list[j]["population"]).keys()))} for j in range(len(partition_list))]
    t += 1
    if t % print_step == 0:
        print(t)
        for partition in partition_list:
            record_partition(partition, t, 'mid_evol')

    partition_list[cur_partition] = gchain.state
    crossover = random.random()
    while crossover < prob_crossover:
        p1, p2 = random.sample(range(len(partition_list)), 2)
        part1 = partition_list[p1]
        part2 = partition_list[p2]
        part_refine = common_refinement(part1, part2)
        part_merge = merge_parts(part_refine,k)
        part_shift = shift_pop(part_merge, k, ep, max_adjust)
        if len(part1["cut_edges"]) > len(part2["cut_edges"]):
            replace_part = p1
        else:
            replace_part = p2
        if max([abs(part_shift["population"][i]-ideal_pop) for i in range(len(part_shift))]) <= ep*ideal_pop:
            partition_list[replace_part] = part_shift
        crossover = random.random()

    cur_partition = random.choice(range(len(partition_list)))
    gchain.state  = partition_list[cur_partition]
    gchain.state.parent = None


print("finished evolutionary")

part_evol= part
for partition in partition_list:
    record_partition(partition, t, 'end_evol')

c_dicts = [{list(dict(partition_list[j]["population"]).keys())[i]:i for i in range(len(dict(partition_list[j]["population"]).keys()))} for j in range(len(partition_list))]


all_cuts.append(cuts_max)
all_cuts.append(cuts_min)

##############################Compare##############################
plt.figure()
plt.title("Cut Lengths")
plt.plot(all_cuts[0],'r',label='Hill')
plt.plot(all_cuts[1],'b',label='Anneal')
plt.plot(all_cuts[2],'lightgreen',label='EvolutionaryMax')
plt.plot(all_cuts[3],'darkgreen',label='EvolutionaryMin')
plt.legend()

plt.savefig(
    "./opt_plots/0_cuts_comparison" + str(exp_num) + "_" + str(pop_bal) + "pop.png"
)
plt.close()

with open('./output/'+ graph_name+"cuts.txt", 'a+') as partition_file:
    writer = csv.writer(partition_file)

    writer.writerow([time.time(), graph_name, num_steps, 'hill', len(graph.nodes())]+all_cuts[0])
    writer.writerow([time.time(), graph_name, num_steps, 'anneal', len(graph.nodes())]+all_cuts[1])
    writer.writerow([time.time(), graph_name, num_steps, 'evol_max', len(graph.nodes())]+all_cuts[2])
    writer.writerow([time.time(), graph_name, num_steps, 'evol_min', len(graph.nodes())]+all_cuts[3])