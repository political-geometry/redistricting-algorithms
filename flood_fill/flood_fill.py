"""
graph flooding algorithms:

author: amy becker

"""

import random
a = random.randint(0,10000000000)
import networkx as nx
from gerrychain.random import random
random.seed(a)
import matplotlib.pyplot as plt
import matplotlib
from math import sqrt


rand_colors = ['blue', 'orange', 'green', 'red', 'yellow','indigo', 'darkorange', 'yellowgreen', 'saddlebrown', 'pink', 'dimgray', 'cornflowerblue', 'cyan', 'gainsboro']

def draw_graph(G, num_districts, pos_map, plan_assignment):
    col_map = {v:plan_assignment[v] if v in plan_assignment.keys() else num_districts for v in G.nodes()}

    plt.figure()
    nx.draw(G, pos = {x: pos_map[x] for x in G.nodes()}, node_size = 50, width = 2, node_color=[col_map[v] for v in G.nodes()], cmap = matplotlib.colors.ListedColormap(rand_colors))
    plt.show()

#Cirincione Flood Fill 
#Also used in BARD
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

#Cirincione variant
def bounding_box_flood_fill(G, ideal_pop, pop_map, num_districts, ep, bounding_box_dict, pos_map, viz):

    district_counter = 0    
    processed_nodes = set()
    plan_assignment = {}
    bounding_boxes = {}

    seed = random.choice(list(G.nodes()))
    plan_assignment[seed] = district_counter
    bounding_boxes[district_counter] = bounding_box_dict[seed]

    local_frontier = set(G.neighbors(seed))
    global_frontier = set(G.neighbors(seed))
    
    processed_nodes.add(seed)
    cur_dist_pop = pop_map[seed]
    if viz:
            draw_graph(G, num_districts, pos_map, plan_assignment)

    unassigned_pop = ideal_pop*num_districts

    while district_counter < num_districts:
        ideal_pop = (unassigned_pop)/(num_districts-district_counter)
        dist_complete = False
        while not dist_complete:
            assert(cur_dist_pop <= (1+ep)*ideal_pop)
            if len(local_frontier) > 0:

                bounded_frontier = []
                box_x_min = bounding_boxes[district_counter][0]
                box_x_max = bounding_boxes[district_counter][1]
                box_y_min = bounding_boxes[district_counter][2]
                box_y_max = bounding_boxes[district_counter][3]

                for v in local_frontier:
                    if box_x_min <= bounding_box_dict[v][0] and box_x_max >= bounding_box_dict[v][1] and box_y_min <= bounding_box_dict[v][2] and box_y_max >= bounding_box_dict[v][3]:
                        bounded_frontier.append(v)

                if len(bounded_frontier) > 0:
                    spread = random.choice(bounded_frontier)
                else:
                    spread = random.choice(list(local_frontier))

                if cur_dist_pop + pop_map[spread] > (1+ep)*ideal_pop:
                    local_frontier.remove(spread)
                else:
                    bounding_boxes[district_counter] = [min(box_x_min,bounding_box_dict[spread][0]),
                                                        max(box_x_max,bounding_box_dict[spread][1]),
                                                        min(box_y_min,bounding_box_dict[spread][2]),
                                                        max(box_y_max,bounding_box_dict[spread][3])]
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
        bounding_boxes[district_counter] = bounding_box_dict[seed]
        if viz:
            draw_graph(G, num_districts, pos_map, plan_assignment)
         

        
#Vickrey model
def farthest_flood_fill(G, ideal_pop, pop_map, num_districts, ep, distance_dict, pos_map, viz):

    district_counter = 0    
    processed_nodes = set()
    plan_assignment = {}
    
    saved_seed = random.choice(list(G.nodes()))
    saved_seed_distances = distance_dict[saved_seed]
    plan_assignment[saved_seed] = num_districts-1
    processed_nodes.add(saved_seed)

    seed = saved_seed_distances[-1]
    saved_seed_distances.remove(seed)
    plan_assignment[seed] = district_counter

    local_frontier = set(G.neighbors(seed))
    processed_nodes.add(seed)
    cur_dist_pop = pop_map[seed]

    if viz:
        draw_graph(G, num_districts, pos_map, plan_assignment)
           
    while district_counter < num_districts:
        while cur_dist_pop < (1-ep)*ideal_pop:

            if len(local_frontier) == 0:
                return False, plan_assignment
            
            for i in distance_dict[seed]:
                if i in local_frontier:
                    spread = i
                    break
    
            saved_seed_distances.remove(spread)
            processed_nodes.add(spread)
            plan_assignment[spread] = district_counter
            cur_dist_pop += pop_map[spread]
            
            local_frontier = local_frontier.union(set(G.neighbors(spread))).difference(processed_nodes)
            if viz:
                draw_graph(G, num_districts, pos_map, plan_assignment)
           
        district_counter += 1
        
        if district_counter >= num_districts:
            return True, plan_assignment
        

        if district_counter == num_districts - 1:
            seed = saved_seed
        else:
            seed = saved_seed_distances[-1]
            saved_seed_distances.remove(seed)
            plan_assignment[seed] = district_counter
            processed_nodes.add(seed)
        local_frontier = set(G.neighbors(seed)).difference(processed_nodes)
        cur_dist_pop = pop_map[seed]
        if viz:
            draw_graph(G, num_districts, pos_map, plan_assignment)
           
        

def seeded_neutral_flood_fill(G, seeds, ideal_pop, pop_map, num_districts, ep, pos_map, loop, viz):
  
    processed_nodes = set()
    plan_assignment = {}
    cur_dist_pop = {}

    # randomly choose more seeds until num_seeds == num_districts
    while len(seeds) < num_districts:
        seed = random.choice(list(G.nodes()))
        if seed not in seeds:
            seeds.append(seed)

    for i in range(len(seeds)):
        plan_assignment[seeds[i]] = i
    
    for seed in seeds:
        processed_nodes.add(seed)
        cur_dist_pop[plan_assignment[seed]] = pop_map[seed]

    local_frontiers = {i:set([v for v in G.neighbors(seeds[i]) if cur_dist_pop[i] + pop_map[v] <= (1+ep)*ideal_pop]).difference(processed_nodes) for i in range(len(seeds))}
    if viz:
            draw_graph(G, num_districts, pos_map, plan_assignment)

    dist_looper = 0

    while len(processed_nodes) < len(G.nodes()):

        if sum([len(frontier) for frontier in local_frontiers.values()]) == 0:
            return False, plan_assignment

        #different ways to choose spread dist:
        #loop through remaining:
        if loop:
            while dist_looper not in [i for i in local_frontiers.keys() if len(local_frontiers[i]) > 0]:
                dist_looper = (dist_looper + 1)%num_districts
            spread_dist = dist_looper
            dist_looper = (dist_looper + 1)%num_districts

        #randomly choose dist
        else:
            spread_dist = random.choice([i for i in local_frontiers.keys() if len(local_frontiers[i]) > 0])

        spread = random.choice(list(local_frontiers[spread_dist]))
        processed_nodes.add(spread)
        plan_assignment[spread] = spread_dist
        cur_dist_pop[spread_dist] += pop_map[spread]

        if viz:
            draw_graph(G, num_districts, pos_map, plan_assignment)
        
        local_frontiers[spread_dist] = set([v for v in set(G.neighbors(spread)).union(local_frontiers[spread_dist]) if cur_dist_pop[spread_dist] + pop_map[v] <= (1+ep)*ideal_pop]).difference(processed_nodes)
    
        local_frontiers = {i:local_frontiers[i].difference(processed_nodes) for i in range(len(seeds))}

        for dist in range(num_districts):
            if len(local_frontiers[dist]) == 0 and cur_dist_pop[dist] < (1-ep)*ideal_pop:
                return False, plan_assignment
    
    return True, plan_assignment



def neutral_iid(G, ideal_pop, pop_map, ep, pos_map, viz):

    district_nodes = []
    processed_nodes = set()
    seed = random.choice(list(G.nodes()))
    district_nodes.append(seed)

    local_frontier = set(G.neighbors(seed))
    processed_nodes.add(seed)
    cur_dist_pop = pop_map[seed]
    
    while cur_dist_pop < ideal_pop:

        if len(local_frontier) == 0 and cur_dist_pop < (1-ep)*ideal_pop:
            return False, district_nodes
        if len(local_frontier) == 0 and cur_dist_pop >= (1-ep)*ideal_pop:
            return True, district_nodes
        
        spread = random.choice(list(local_frontier))
        if cur_dist_pop + pop_map[spread] > (1+ep)*ideal_pop:
            local_frontier.remove(spread)
        else:
            processed_nodes.add(spread)
            district_nodes.append(spread)
            cur_dist_pop += pop_map[spread]

            local_frontier = local_frontier.union(set(G.neighbors(spread))).difference(processed_nodes)
    if viz:
        col_map = {v:0 if v in district_nodes else 1 for v in G.nodes()}

        plt.figure()
        nx.draw(G, pos = {x: pos_map[x] for x in G.nodes()}, node_size = 500, width = 2, node_color=[col_map[v] for v in G.nodes()], cmap = matplotlib.colors.ListedColormap(rand_colors))
        plt.show()

    return True, district_nodes

def bounding_box_iid(G, ideal_pop, pop_map, ep, bounding_box_dict, pos_map, viz):

    district_nodes = []
    processed_nodes = set()
    seed = random.choice(list(G.nodes()))
    district_nodes.append(seed)

    local_frontier = set(G.neighbors(seed))
    processed_nodes.add(seed)
    cur_dist_pop = pop_map[seed]
    bounding_box = bounding_box_dict[seed]
    while cur_dist_pop < ideal_pop:


        if len(local_frontier) == 0 and cur_dist_pop < (1-ep)*ideal_pop:
            return False, district_nodes
        if len(local_frontier) == 0 and cur_dist_pop >= (1-ep)*ideal_pop:
            return True, district_nodes

        bounded_frontier = []
        box_x_min = bounding_box[0]
        box_x_max = bounding_box[1]
        box_y_min = bounding_box[2]
        box_y_max = bounding_box[3]

        for v in local_frontier:
            if box_x_min <= bounding_box_dict[v][0] and box_x_max >= bounding_box_dict[v][1] and box_y_min <= bounding_box_dict[v][2] and box_y_max >= bounding_box_dict[v][3]:
                bounded_frontier.append(v)

        if len(bounded_frontier) > 0:
            spread = random.choice(bounded_frontier)
        else:
            spread = random.choice(list(local_frontier))

        if cur_dist_pop + pop_map[spread] > (1+ep)*ideal_pop:
            local_frontier.remove(spread)
        else:
            bounding_box = [min(box_x_min,bounding_box_dict[spread][0]),
                                                max(box_x_max,bounding_box_dict[spread][1]),
                                                min(box_y_min,bounding_box_dict[spread][2]),
                                                max(box_y_max,bounding_box_dict[spread][3])]
            processed_nodes.add(spread)
            district_nodes.append(spread)
            cur_dist_pop += pop_map[spread]

            local_frontier = local_frontier.union(set(G.neighbors(spread))).difference(processed_nodes)
                
    if viz:
        col_map = {v:0 if v in district_nodes else 1 for v in G.nodes()}

        plt.figure()
        nx.draw(G, pos = {x: pos_map[x] for x in G.nodes()}, node_size = 500, width = 2, node_color=[col_map[v] for v in G.nodes()], cmap = matplotlib.colors.ListedColormap(rand_colors))
        plt.show()
    return True, district_nodes



def grid_dist(u,v,queen_adj,noise_term):
    if queen_adj:
        dist = sqrt((v[0]-u[0])**2+(v[1]-u[1])**2)
    else:
        dist = abs(v[0]-u[0])+abs(v[1]-u[1])
    if noise_term:
        dist += random.random()/100
    return dist


def add_to_dict(G, assignment_dict, new_assignment):
    for v in G.nodes():
        assignment_dict[G.node[v]['COUNTY']].append(new_assignment[v])
    return assignment_dict

