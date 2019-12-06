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

    G, _ = adjacency_matrix_to_graph(adjacency_matrix)

    index_to_location = dict(zip(G.nodes(), list_of_locations))
    location_to_index = dict(zip(list_of_locations, G.nodes()))
    newG = nx.relabel_nodes(G, index_to_location)
    # drawGraph(G)

    #tour, dropoff_map = metric_TSP_solver(newG, starting_car_location, list_of_homes[:], list_of_homes[:])
    tour, dropoff_map = simplified_metric_TSP_solver(newG, starting_car_location, list_of_homes[:])

    indexed_tour = [location_to_index[loc] for loc in tour]

    indexed_dropoff_map = {}
    for loc in dropoff_map:
        indexed_dropoff_map[location_to_index[loc]] = [location_to_index[home] for home in dropoff_map[loc]]

    print(cost_of_solution(G, indexed_tour, indexed_dropoff_map))
    return indexed_tour, indexed_dropoff_map

####### SHITTINESS

def acceptance_probability(candidate_weight, curr_weight, temp):
    return math.exp(-abs(candidate_weight - curr_weight) / temp)

def accept(candidate_weight, curr_weight, temp):
    if candidate_weight < curr_weight:
        return 0
    else:
        if random.random() < acceptance_probability(candidate_weight, curr_weight, temp):
            return 1
    return 2

def shitty_solver(G, starting_car_location, main_list_of_super_nodes, valid_drop_off_map):
    temp = 1000
    stopping_temp = 1e-3
    alpha = 0.999999
    stopping_iter = 1e7
    swap_prob = 1
    iteration = 1

    all_shortest_paths = nx.shortest_path(G)
    shortest_lengths = dict(nx.floyd_warshall(G))

    # print(shortest_lengths)

    list_of_super_nodes = main_list_of_super_nodes[:]

    if starting_car_location in list_of_super_nodes:
        list_of_super_nodes.pop(starting_car_location)

    best_tour = list_of_super_nodes[:]
    best_cost = float('inf')

    curr_tour = list_of_super_nodes[:]
    curr_cost = float('inf')

    while temp >= stopping_temp and iteration < stopping_iter:
        prob = random.random()

        new_version = curr_tour[:]
        if prob < swap_prob:
            first_random = random.randint(0, len(new_version)-1)
            second_random = random.randint(0, len(new_version)-1)
            new_version[first_random], new_version[second_random] = new_version[second_random], new_version[first_random]
        
        new_version_cost = shitty_cost_calculator(new_version, starting_car_location, shortest_lengths)
        # if(new_version_cost < 32.0):
            # print("new version cost: ", new_version_cost)
        acceptance = accept(new_version_cost, curr_cost, temp)

        if acceptance <= 1:
            curr_cost = new_version_cost
            curr_tour = list(new_version)
        if acceptance == 0:
            if new_version_cost < best_cost:
                best_cost = new_version_cost
                best_tour = list(new_version)
        # print("best cost: ", best_cost)
        temp *= alpha
        iteration += 1
    
    final_tour = shitty_tour_calculator(best_tour, starting_car_location, all_shortest_paths)
    return final_tour

def shitty_cost_calculator(tour_nodes, starting_car_location, shortest_lengths):
    start_to_first = shortest_lengths[starting_car_location][tour_nodes[0]]
    last_to_start = shortest_lengths[tour_nodes[len(tour_nodes) - 1]][starting_car_location]
    intermediaries = sum([shortest_lengths[tour_nodes[i]][tour_nodes[i+1]] for i in range(len(tour_nodes) - 1)])
    return start_to_first + intermediaries + last_to_start
    #  tour_nodes[i]tour_nodes[i+1] for i in range(len(tour_nodes) + 1)])

def shitty_tour_calculator(tour_nodes, starting_car_location, shortest_paths):
    main_tour = [starting_car_location]
    main_tour.extend(shortest_paths[starting_car_location][tour_nodes[0]][1:])
    for i in range(len(tour_nodes) - 1):
        current_addition = shortest_paths[tour_nodes[i]][tour_nodes[i+1]]
        current_addition = current_addition[1:]
        main_tour.extend(current_addition)
    main_tour.extend(shortest_paths[tour_nodes[len(tour_nodes)-1]][starting_car_location][1:])
    return main_tour

####### SHITTINESS

####### OLD VERSION

# Optimization ideas:
# 1. find_tour can examimine more tours
# 2. use some commercial algo for find_tour

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
    # locations_to_visit = dropoff_locations + list_of_homes
    # for location in locations_to_visit:
    #     if location in dropped_homes:
    #         locations_to_visit.remove(location)

    # Finding homes that need direct door dropoff.
    direct_homes = []
    for home in list_of_homes:
        if home not in dropped_homes:
            direct_homes.append(home)

    # Updating dropoff map to include direct door dropoff homes.
    for home in direct_homes:
        if home not in dropoff_loc_to_homes.keys():
            dropoff_loc_to_homes[home] = [home]

    # tour = find_tour(G, starting_car_location, dropoff_loc_to_homes.keys())
    # tour = find_better_tour(G, starting_car_location, dropoff_loc_to_homes.keys())
    tour = shitty_solver(G, starting_car_location, list(dropoff_loc_to_homes.keys()), dropoff_loc_to_homes)

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
    # drive_tour = find_tour(G, starting_car_location, subtree_homes)
    drive_tour = find_better_tour(G, starting_car_location, subtree_homes)
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

    # Saves indices of visited locations.
    visited = {}
    # List of locations the car must visit.
    locations = []
    # Maps locations to the homes of the TAs who were dropped off at the location.
    dropoff_map = {}

    super_nodes = []
    super_node_costs = {}
    new_tour = []

    # print("DFS: ", dfs_traversal)

    for i in range(len(dfs_traversal)):
        node = dfs_traversal[i]

        if node not in visited: # Visiting a location for the first time.
            visited[node] = i
            new_tour.append(node)
            if node in unchanged_list_of_homes: # If the node is a home, we must visit it in the final tour.
                locations.append(node)
        else:
            start = visited[node]
            end = i + 1
            location_loop = dfs_traversal[start:end] # [start_name ... start_name]

            drive_cost = old_compute_drive_cost(G, location_loop, super_nodes, super_node_costs)
            dropoff_cost, dropped_homes = old_compute_dropoff_cost(G, location_loop, unchanged_list_of_homes, dropoff_map)
            # print("Current location loop: ", location_loop)
            # print("Dropoff cost: {0}, Drive cost: {1}".format(dropoff_cost, drive_cost))
            # If it's better to drop all the TAs living withing the loop at the start than to drive the entire loop.
            if (dropoff_cost < drive_cost and dropoff_cost > 0):

                for home in dropped_homes:
                    # if home in list_of_homes: # Not sure if this is necessary.
                    #     list_of_homes.remove(home)
                    if home in locations:
                        locations.remove(home)
                    # Also removing super dropoff locations within the location loop.
                    if home in dropoff_map:
                        del dropoff_map[home]
                
                for location in location_loop:
                    if location != dfs_traversal[start]:
                        if location in dropoff_map:
                            del dropoff_map[location]
                
                for i in range(start, end):
                    new_tour.pop()

                dfs_traversal[start:end] = [dfs_traversal[start] for i in range(len(dfs_traversal[start:end]))]

                dropoff_map[node] = dropped_homes # Storing for output file.
                # changing_map[node] = dropped_homes
                locations.append(node) # The final tour must include this dropoff location.
                # list_of_homes.append(node) # Classifying the dropoff location as a pseudo home.
                super_nodes.append(node)
                super_node_costs[node] = dropoff_cost

            else: # Else, it's better to drive the loop. Update the index of visited location.
                if(dropoff_cost <= 0):
                    for loc in location_loop[1:len(location_loop)+1]:
                        new_tour.pop()
                    # add the path
                else:
                    pass
                    # don't include path
                visited[node] = i
                for loc in location_loop:
                    new_tour.append(loc)
                    if loc in list_of_homes:
                        locations.append(loc) # The final tour should include all homes in the location loop.
            # print("Costs for supers: ", super_node_costs)
            # print()
            # # print("New Tour",new_tour)
            # print("Following drop off map:", dropoff_map)


    tour = find_tour(G, starting_car_location, dropoff_map.keys())
    # print("New Tour: ", new_tour)
    return tour, dropoff_map

# Calculates the cost of driving the entire loop.
def old_compute_drive_cost(G, location_loop, super_nodes, super_node_costs):
    cost = 0
    for i in range(len(location_loop) - 1):
        source = location_loop[i]
        dest = location_loop[i + 1]
        if(source == dest):
            continue
        cost += (2.0 / 3.0) * (G[source][dest]['weight'])
        # print(sum[super_node_costs[i] for i in super_nodes if i in location_loop[1:len(location_loop)-1]])
    set_loop = set(location_loop)
    # print("cost before adding super node: ", cost)
    # print("super node costs: ", [super_node_costs[i] for i in set_loop if i in super_nodes])
    total = cost + sum([super_node_costs[i] for i in set_loop if i in super_nodes])
    return total

# This returns the total cost of leaving the people at this location
# The dropoff map includes each the current location if it is a home
def old_compute_dropoff_cost(G, location_loop, unchanged_list_of_homes, dropoff_map):
    cost = 0
    source = location_loop[0]
    location_loop = set(location_loop[1:len(location_loop)- 1])

    dropped_homes = []
    for node in location_loop:
        if node in unchanged_list_of_homes and not node in dropoff_map:
            walk_distance, _ = nx.single_source_dijkstra(G, source=source, target=node, cutoff=None, weight='weight')
            cost += walk_distance
            dropped_homes.append(node)
        if node in dropoff_map:
            for sub_node in dropoff_map[node]:
                walk_distance, _ = nx.single_source_dijkstra(G, source=source, target=sub_node, cutoff=None, weight='weight')
                cost += walk_distance
                dropped_homes.append(sub_node)
    if source in unchanged_list_of_homes:
        dropped_homes.append(source)
    return cost, dropped_homes

####### OLD VERSION

####### NEW VERSION

# Calculates cost of dropping off all the TA's whose homes are in the subtree rooted at starting_location.
def find_better_tour(G, starting_car_location, locations):
    if len(locations) < 3:
        return find_tour(G, starting_car_location, locations)

    tour = []
    visited = {}
    for loc in locations:
        visited[loc] = False

    visited[starting_car_location] = True
    tour.append(starting_car_location)

    # Store tours and visited map for all second location possibilities.
    second_loc_to_tour = {}
    second_loc_to_visited = {}
    for loc in locations:
        if not visited[loc] and G.has_edge(starting_car_location, loc):
            second_loc_to_tour[loc] = copy.deepcopy(tour)
            second_loc_to_visited[loc] = copy.deepcopy(visited)

    if not second_loc_to_tour:
        return find_tour(G, starting_car_location, locations)

    for second_loc in second_loc_to_tour.keys():
        current_loc = second_loc
        while (has_not_visited_all_locations(second_loc_to_visited[second_loc])):
            path_lengths = nx.single_source_dijkstra_path_length(G, source=current_loc, cutoff=None, weight='weight')
            closest_loc = None
            closest_loc_distance = float('inf')

            for loc in locations:
                if not second_loc_to_visited[second_loc][loc]:
                    loc_distance = path_lengths[loc]
                    if (loc_distance < closest_loc_distance):
                        closest_loc = loc
                        closest_loc_distance = loc_distance

            chosen_path = nx.dijkstra_path(G, source=current_loc, target=closest_loc, weight='weight')
            second_loc_to_tour[second_loc] += chosen_path[:(len(chosen_path) - 1)]

            second_loc_to_visited[second_loc][closest_loc] = True
            current_loc = closest_loc

        return_path = nx.dijkstra_path(G, source=current_loc, target=starting_car_location, weight='weight')
        second_loc_to_tour[second_loc] += return_path

    # for tour in second_loc_to_tour.values():
        # print(tour)
    # print(len(list(second_loc_to_tour.values())))
    # Pick the best tour
    best_second_loc_tour = min(second_loc_to_tour.values(), key = lambda tour : compute_drive_cost(G, tour))
    return best_second_loc_tour

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

####### NEW VERSION
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
