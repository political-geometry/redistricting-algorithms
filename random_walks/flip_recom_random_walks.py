# Import for I/O

import random
a = random.randint(0,10000000000)
from functools import partial
import time
import csv

from gerrychain.random import random
random.seed(a)
from gerrychain.proposals import recom
from gerrychain import MarkovChain, Graph
from gerrychain.constraints import (
    Validator,
    single_flip_contiguous,
    within_percent_of_ideal_population,
)
from gerrychain.accept import always_accept
from gerrychain.tree import recursive_tree_part
from gerrychain.updaters import Tally, cut_edges
from gerrychain.partition import Partition
import networkx as nx
import os


####################################################################################
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


output_path = './output/'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
start_time = time.time()

k = 4
ep = 0.05
tree_walk = True
num_steps = 100000
print_step = 1000

for graph_name in ['iowa','grid']:
    if graph_name == 'iowa':
        graph_path = './../../data/'+graph_name+'.json'
        graph = Graph.from_json(graph_path)
    elif graph_name == 'grid':
        graph = nx.grid_graph([10,10])
        for v in graph.nodes():
            graph.nodes[v]["TOTPOP"] = 1
    else:
        print('INVALID GRAPH CHOICE')
        assert(False)

    node_list = list(graph.nodes())
    edge_list = list(graph.edges())
    num_districts = k
    ideal_pop = sum([graph.nodes[v]["TOTPOP"] for v in graph.nodes()])/num_districts

    cddict = recursive_tree_part(graph, range(k), ideal_pop, "TOTPOP", .02, 3)

    # Necessary updaters go here
    updaters = {
        "population": Tally("TOTPOP", alias="population"),
        "cut_edges": cut_edges,
    }

    seed_partition = Partition(graph, assignment=cddict, updaters=updaters)

    if tree_walk:
        ideal_population = sum(seed_partition["population"].values()) / len(seed_partition)
        tw_proposal = partial(
            recom,
            pop_col="TOTPOP",
            pop_target=ideal_population,
            epsilon=0.02,
            node_repeats=1,
        )
        popbound = within_percent_of_ideal_population(seed_partition, 0.05)
        init_gchain = MarkovChain(
            tw_proposal,  # propose_chunk_flip,
            Validator([popbound]),
            accept=always_accept,
            initial_state=seed_partition,
            total_steps=100,
        )
        t = 0
        for plan in init_gchain:
            t += 1
        init_part = plan
    else:
        init_part = seed_partition

    print("finished tree walk")
    starting_partition = Partition(graph, assignment=dict(init_part.assignment), updaters=updaters)
    popbound = within_percent_of_ideal_population(starting_partition, ep)


    ############################## FLIP ##############################
    gchain = MarkovChain(
        slow_reversible_propose,
        Validator([single_flip_contiguous, popbound]),
        accept=always_accept, 
        initial_state=starting_partition,
        total_steps=num_steps,
    )

    cut_edges_dict = {e:0 for e in graph.edges()}
    cut_len = []
    t = 0
    for part in gchain:
        if t% print_step == 0:
            print('flip',graph_name,t)
        cut_len.append(len(part["cut_edges"]))
        for e in part["cut_edges"]:
            cut_edges_dict[e] += 1
        pop_error = max([abs(x-ideal_pop) for x in part["population"].values()])/ideal_pop
        if pop_error > ep:
            print('flip', graph_name, pop_error)
        t += 1

    with open('./output/'+ graph_name+"_chain_cut_len_"+str(round(start_time))+".txt", 'a+') as partition_file:
        writer = csv.writer(partition_file)
        writer.writerow([time.time(), graph_name, num_districts, num_steps, 'flip', cut_len])

    with open('./output/'+ graph_name+"_chain_cut_edges_"+str(round(start_time))+".txt", 'a+') as partition_file:
        writer = csv.writer(partition_file)
        writer.writerow([time.time(), graph_name, num_districts, num_steps, 'flip', cut_edges_dict])


    ############################## RECOM ##############################
    recom_proposal = partial(recom, pop_col="TOTPOP", pop_target=ideal_pop, epsilon=ep, node_repeats=2)
    gchain = MarkovChain(
        recom_proposal,  # propose_random_flip,#propose_chunk_flip, # ,
        Validator([popbound]),
        accept=always_accept,  # aca,#cut_accept,#always_accept,#
        initial_state=starting_partition,
        total_steps=num_steps,
    )

    cut_edges_dict = {e:0 for e in graph.edges()}
    cut_len = []
    t = 0
    for part in gchain:
        if t% print_step == 0:
            print('recom',graph_name,t)
        cut_len.append(len(part["cut_edges"]))
        for e in part["cut_edges"]:
            cut_edges_dict[e] += 1
        pop_error = max([abs(x-ideal_pop) for x in part["population"].values()])/ideal_pop
        if pop_error > ep:
            print('recom', graph_name, pop_error)
        t += 1


    with open('./output/'+ graph_name+"_chain_cut_len_"+str(round(start_time))+".txt", 'a+') as partition_file:
        writer = csv.writer(partition_file)
        writer.writerow([time.time(), graph_name, num_districts, num_steps, 'recom', cut_len])

    with open('./output/'+ graph_name+"_chain_cut_edges_"+str(round(start_time))+".txt", 'a+') as partition_file:
        writer = csv.writer(partition_file)
        writer.writerow([time.time(), graph_name, num_districts, num_steps, 'recom', cut_edges_dict])
