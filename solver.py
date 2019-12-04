import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import numpy as np
import random
import string
import math
import matplotlib.pyplot as plt
import networkx as nx

from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """

    # Reformat the adjacency matrix
    # unscaled_G = adj_mat_to_graph(adjacency_matrix)
    # upscale_matrix(adjacency_matrix)
    # print(adjacency_matrix)
    G = adj_mat_to_graph(adjacency_matrix)
    #drawGraph(G)
    index_to_location = dict(zip(G.nodes(), list_of_locations))
    location_to_index = dict(zip(list_of_locations, G.nodes()))
    G = nx.relabel_nodes(G, index_to_location)

    tour, dropoff_map = metric_TSP_solver(G, starting_car_location, list_of_homes)
    tour = [location_to_index[loc] for loc in tour]
    dropoff_map = [location_to_index[loc] for loc in dropoff_map]
    return tour, dropoff_map

def metric_TSP_solver(G, starting_car_location, list_of_homes):
    T = nx.minimum_spanning_tree(G)
    # Generates a DFS call sequence.
    marked = {}
    for node in G.nodes:
        marked[node] = False
    dfs_traversal = []
    def gen_dfs(node):
        dfs_traversal.append(node)
        marked[node] = True
        is_leaf = True
        for neighbor in T.neighbors(node):
            if not marked[neighbor]:
                is_leaf = False
                gen_dfs(neighbor)
        if not is_leaf:
            dfs_traversal.append(node)
    gen_dfs(starting_car_location)

    # Saves indices of visited locations.
    visited = {}
    # List of locations represeting the car path.
    tour = dfs_traversal[:]
    # Maps locations to the homes of the TAs who were dropped off at the location.
    dropoff_map = {}
    for i in range(len(dfs_traversal)):
        node = dfs_traversal[i]
        if node not in visited: # Visiting a location for the first time.
            visited[node] = i
        else: # We have arrived at a location loop.
            start = visited[node]
            end = i + 1
            location_loop = dfs_traversal[start:end]
            drive_cost = compute_drive_cost(G, location_loop)
            dropoff_cost, dropped_homes = compute_dropoff_cost(G, location_loop, list_of_homes)
            # If it's better to drop all the TAs living withing the loop at the start than to drive the entire loop.
            if (dropoff_cost < drive_cost):
                for location in location_loop[1:]:
                    tour.remove(location)
                # Storing for output file.
                dropoff_map[node] = dropped_homes
                # Removing homes of the TAs that were dropped off and creating a SUPER home.
                for home in dropped_homes:
                    list_of_homes.remove(home)
                list_of_homes.append(node)
    return tour, dropoff_map


# Calculates the cost of driving the entire loop.
def compute_drive_cost(G, location_loop):
    cost = 0
    for i in range(len(location_loop) - 1):
        source = location_loop[i]
        dest = location_loop[i + 1]
        cost += (2 / 3) * (G[source][dest]['weight'])
    return cost

# Calculates the cost of dropping off all TAs who live within the loop at the start.
def compute_dropoff_cost(G, location_loop, list_of_homes):
    cost = 0
    source = location_loop[0]
    dropped_homes = []
    for node in location_loop:
        if node in list_of_homes:
            walk_distance, _ = nx.single_source_dijkstra(source=source, target=node, cutoff=None, weight='weight')
            cost += walk_distance
            dropped_homes.append(node)
    return cost, dropped_homes


"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)

def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)

def drawGraph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

#############################

def adj_mat_to_graph(adj_mat):
    adj_mat = np.matrix(adj_mat)
    G = nx.from_numpy_matrix(adj_mat)
    return G

def upscale_matrix(adj_mat):
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat[0])):
            adj_mat[i][j] *= 10 * 10

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
