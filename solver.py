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
import copy

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
    G, message = adjacency_matrix_to_graph(adjacency_matrix)
    #drawGraph(G)
    index_to_location = dict(zip(G.nodes(), list_of_locations))
    location_to_index = dict(zip(list_of_locations, G.nodes()))
    newG = nx.relabel_nodes(G, index_to_location)

    #tour, dropoff_map = metric_TSP_solver(newG, starting_car_location, list_of_homes[:], list_of_homes[:])
    tour, dropoff_map = simplified_metric_TSP_solver(newG, starting_car_location, list_of_homes[:])

    indexed_tour = [location_to_index[loc] for loc in tour]

    indexed_dropoff_map = {}
    for loc in dropoff_map:
        indexed_dropoff_map[location_to_index[loc]] = [location_to_index[home] for home in dropoff_map[loc]]

    print(cost_of_solution(G, indexed_tour, indexed_dropoff_map))
    return indexed_tour, indexed_dropoff_map

def simplified_metric_TSP_solver(G, starting_car_location, list_of_homes):
    T = nx.minimum_spanning_tree(G)
    dropoff_locations, dropoff_loc_to_homes = find_dropoff_locations(T, starting_car_location, list_of_homes, G)
    # Checks if one dropoff location supersedes another.
    replaced_dropoff_locations = []
    for i in range(len(dropoff_locations)):
        for j in range(i + 1, len(dropoff_locations)):
            locA = dropoff_locations[i]
            locB = dropoff_locations[j]
            homes_dropped_off_A = dropoff_loc_to_homes[locA]
            homes_dropped_off_B = dropoff_loc_to_homes[locB]
            if set(homes_dropped_off_A).issuperset(set(homes_dropped_off_B)):
                if locB not in replaced_dropoff_locations:
                    replaced_dropoff_locations.append(locB)

    # print(dropoff_locations)
    # print(replaced_dropoff_locations)
    # Remove superseded dropoff locations.
    for location in replaced_dropoff_locations:
        dropoff_locations.remove(location)
        del dropoff_loc_to_homes[location]

    # Store all the homes dropped off indirectly.
    dropped_homes = []
    for location in dropoff_locations:
        for home in dropoff_loc_to_homes[location]:
            dropped_homes.append(home)

    # Locations car must visit include dropoff locations and homes that need direct door dropoff.
    locations_to_visit = dropoff_locations + list_of_homes
    for location in locations_to_visit:
        if location in dropped_homes:
            locations_to_visit.remove(location)

    # Finding homes that need direct door dropoff.
    direct_homes = []
    for home in list_of_homes:
        if home not in dropped_homes:
            direct_homes.append(home)

    # Updating dropoff map to include direct door dropoff homes.
    for home in direct_homes:
        if home not in dropoff_loc_to_homes.keys():
            dropoff_loc_to_homes[home] = [home]

    tour = find_tour(G, starting_car_location, dropoff_loc_to_homes.keys())

    # print(dropoff_loc_to_homes)
    return tour, dropoff_loc_to_homes

# Traverses the entire MST, computing dropoff locations.
def find_dropoff_locations(T, starting_car_location, list_of_homes, G):
    stack = []
    stack.append(starting_car_location)
    visited = {}
    for node in T.nodes:
        visited[node] = False
    dropoff_locations = []
    dropoff_loc_to_homes = {}
    while len(stack) > 0:
        node = stack.pop()
        visited[node] = True
        # Deep copy of visited dict needed to ensure that we only traverse the subtree.
        dropoff_cost, subtree_homes = root_dropoff_cost(T, node, list_of_homes, G, copy.deepcopy(visited))
        drive_cost = root_drive_cost(T, node, list_of_homes, G, copy.deepcopy(visited))
        if 0 < dropoff_cost and dropoff_cost < drive_cost:
            dropoff_locations.append(node) # Classify node as a dropoff location.
            dropoff_loc_to_homes[node] = subtree_homes # Build the dropoff map.

        for neighbor in T.neighbors(node):
            if not visited[neighbor]:
                stack.append(neighbor)
    return dropoff_locations, dropoff_loc_to_homes


# Returns a list of all TA homes contained in the subtree rooted at starting_location.
def find_subtree_homes(T, starting_location, list_of_homes, visited):
    subtree_homes = []
    stack = []
    stack.append(starting_location)
    while len(stack) > 0:
        node = stack.pop()
        visited[node] = True
        if node in list_of_homes:
            subtree_homes.append(node)
        for neighbor in T.neighbors(node):
            if not visited[neighbor]:
                stack.append(neighbor)
    return subtree_homes

# Calculates cost of dropping off all the TA's whose homes are in the subtree rooted at starting_location.
def root_dropoff_cost(T, starting_car_location, list_of_homes, G, visited):
    subtree_homes = find_subtree_homes(T, starting_car_location, list_of_homes, visited)
    dropoff_cost = 0
    for home in subtree_homes:
        walk_distance, _ = nx.single_source_dijkstra(G, source=starting_car_location, target=home, cutoff=None, weight='weight')
        dropoff_cost += walk_distance
    return dropoff_cost, subtree_homes

# Calculates the cost of a tour that visits all TA homes contained in the subtree rooted at starting_location.
# NOTE: not guaranteed to be min cost tour, but pretty decent heuristic.
def root_drive_cost(T, starting_car_location, list_of_homes, G, visited):
    subtree_homes = find_subtree_homes(T, starting_car_location, list_of_homes, visited)
    drive_tour = find_tour(G, starting_car_location, subtree_homes)
    driving_cost = compute_drive_cost(G, drive_tour)
    return driving_cost

def metric_TSP_solver(G, starting_car_location, list_of_homes, unchanged_list_of_homes):
    T = nx.minimum_spanning_tree(G)

    # Generates a DFS call sequence.
    marked = {}
    for node in G.nodes:
        marked[node] = False
    dfs_traversal = []

    def gen_dfs(node):
        dfs_traversal.append(node)
        marked[node] = True
        for neighbor in T.neighbors(node):
            if not marked[neighbor]:
                gen_dfs(neighbor)
            if dfs_traversal[len(dfs_traversal) - 1] != node:
                dfs_traversal.append(node)

    gen_dfs(starting_car_location)

    # print(dfs_traversal)

    # Saves indices of visited locations.
    visited = {}
    # List of locations the car must visit.
    locations = []
    # Maps locations to the homes of the TAs who were dropped off at the location.
    dropoff_map = {}
    for i in range(len(dfs_traversal)):
        node = dfs_traversal[i]
        if node not in visited: # Visiting a location for the first time.
            visited[node] = i
            if node in list_of_homes: # If the node is a home, we must visit it in the final tour.
                locations.append(node)
        else: # We have arrived at a location loop.
            start = visited[node]
            end = i + 1
            location_loop = dfs_traversal[start:end]
            drive_cost = compute_drive_cost(G, location_loop)
            dropoff_cost, dropped_homes = compute_dropoff_cost(G, location_loop, list_of_homes, unchanged_list_of_homes)

            # If it's better to drop all the TAs living withing the loop at the start than to drive the entire loop.
            if (dropoff_cost < drive_cost and dropoff_cost > 0):

                # Removing homes of the TAs that were dropped off.
                overall_dropped_locations = []

                for home in dropped_homes:
                    if home in list_of_homes: # Not sure if this is necessary.
                        if home in unchanged_list_of_homes:
                            overall_dropped_locations.append(home)
                        list_of_homes.remove(home)

                    if home in locations:
                        locations.remove(home)
                    # Also removing super dropoff locations within the location loop.
                    if home in dropoff_map:
                        overall_dropped_locations.extend(dropoff_map[home])
                        del dropoff_map[home]

                dropoff_map[node] = overall_dropped_locations # Storing for output file.
                locations.append(node) # The final tour must include this dropoff location.
                list_of_homes.append(node) # Classifying the dropoff location as a pseudo home.

            else: # Else, it's better to drive the loop. Update the index of visited location.
                visited[node] = i
                for loc in location_loop:
                    if loc in list_of_homes:
                        locations.append(loc) # The final tour should include all homes in the location loop.

        # print(dropoff_map)
    # print(len(locations))
    # print(locations)
    # print(dropoff_map)
    tour = find_tour(G, starting_car_location, locations)
    return tour, dropoff_map

def find_tour(G, starting_car_location, locations):
    tour = [] # Stores the path the car takes.
    current_loc = starting_car_location
    visited = {}
    for loc in locations:
        visited[loc] = False

    while (has_not_visited_all_locations(visited)):
        # Compute the shortest paths tree rooted at the current location.
        path_lengths = nx.single_source_dijkstra_path_length(G, source=current_loc, cutoff=None, weight='weight')
        closest_loc = None
        closest_loc_distance = float('inf')
        # Finding the closest location.
        for loc in locations:
            if not visited[loc]:
                loc_distance = path_lengths[loc]
                if (loc_distance < closest_loc_distance):
                    closest_loc = loc
                    closest_loc_distance = loc_distance

        # Compute the desired shortest path and append to tour.
        chosen_path = nx.dijkstra_path(G, source=current_loc, target=closest_loc, weight='weight')
        tour += chosen_path[:(len(chosen_path) - 1)]
        # Visit the closest location and update the current location.
        visited[closest_loc] = True
        current_loc = closest_loc

    # Compute the shortest path back to the start.
    return_path = nx.dijkstra_path(G, source=current_loc, target=starting_car_location, weight='weight')
    tour += return_path
    return tour

def has_not_visited_all_locations(visited):
    return False in visited.values()

# Calculates the cost of driving the entire loop.
def compute_drive_cost(G, location_loop):
    cost = 0
    for i in range(len(location_loop) - 1):
        source = location_loop[i]
        dest = location_loop[i + 1]
        # print('s: ' + source)
        # print('d: ' + dest)
        cost += (2.0 / 3.0) * (G[source][dest]['weight'])
    return cost

# Calculates the cost of dropping off all TAs who live within the loop at the start.
def compute_dropoff_cost(G, location_loop, list_of_homes, main_list_of_homes):
    cost = 0
    source = location_loop[0]
    dropped_homes = []
    for node in location_loop:
        if node in main_list_of_homes:
            walk_distance, _ = nx.single_source_dijkstra(G, source=source, target=node, cutoff=None, weight='weight')
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
    output_directory = args.output_directory + '/outputs/'
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
