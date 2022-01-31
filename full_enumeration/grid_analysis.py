import networkx as nx
from gerrychain import Graph
import matplotlib.pyplot as plt
import csv
import os

output_path = "./output/"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

filepath = './enumerations/enum_[6,6]_[9]_4_rc.txt'
G = nx.grid_graph([6,6])
node_list = [e for e in G.nodes()]
state_space = []

with open(filepath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        cut = {}
        for i in range(len(node_list)):
            cut[node_list[i]] = row[i]
        state_space.append(cut)


cut_freq = {x:0 for x in G.edges()}
d2_freq = {x:0 for x in G.nodes()}
cut_length = []


for cut in state_space:
    cut_edges = [e for e in G.edges() if cut[e[0]] != cut[e[1]]]
    for e in cut_edges:
        cut_freq[e] += 1
    cut_length.append(len(cut_edges))


out_row = ['enum_[6,6]_[9]_4_rc', cut_freq]
with open(output_path + "enum_cut_freq.txt", 'a+') as cut_freq_file:
    writer = csv.writer(cut_freq_file)
    writer.writerow(out_row)
    
out_row = ['enum_[6,6]_[9]_4_rc',cut_length]
with open(output_path + "enum_cut_length.txt", 'a+') as cut_len_file:
    writer = csv.writer(cut_len_file)
    writer.writerow(out_row)

