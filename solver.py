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
    G, message = adjacency_matrix_to_graph(adjacency_matrix)
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
    tour = []
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
                # for loc in location_loop[1:]:
                #     tour.remove(loc)
                # Storing for output file.
                dropoff_map[node] = dropped_homes
                # Removing homes of the TAs that were dropped off and creating a SUPER home.
                # for home in dropped_homes:
                #     list_of_homes.remove(home)
                # list_of_homes.append(node)
            else: # Else, it's better to drive the loop. Update the index of visited location.
                visited[node] = i
                dropoff_map[node] = [] # Don't drop off any TA, but still visit this location.
    return tour, dropoff_map


# Calculates the cost of driving the entire loop.
def compute_drive_cost(G, location_loop):
    cost = 0
    for i in range(len(location_loop) - 1):
        source = location_loop[i]
        dest = location_loop[i + 1]
        cost += (2.0 / 3.0) * (G[source][dest]['weight'])
    return cost

# Calculates the cost of dropping off all TAs who live within the loop at the start.
def compute_dropoff_cost(G, location_loop, list_of_homes):
    cost = 0
    source = location_loop[0]
    dropped_homes = []
    for node in location_loop:
        if node in list_of_homes:
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
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

// Include OpenMP
#include <omp.h>

#include "mandelbrot.h"
#include "parameters.h"

void print_part(__m256d item, char * name){
    double * item_double = (double*) &item;
    printf("%s: ", name);
    for(int j = 0; j < 4; j++){
        printf("%d ", item_double[j]);
    }
    printf("\n");
}

uint32_t iterations(__m256d point_r_256, __m256d point_i_256, __m256d thresholds, struct parameters params) {
    __m256d z_r_s_256 = _mm256_set1_pd(0);
    __m256d z_i_s_256 = _mm256_set1_pd(0);
    int num_zeros = 0;
    for (int i = 1; i <= params.maxiters; i++) {

        __m256d new_z_r_s_256 = _mm256_sub_pd(_mm256_mul_pd(z_r_s_256, z_r_s_256), _mm256_mul_pd(z_i_s_256, z_i_s_256));
        __m256d new_z_i_s_256 = _mm256_mul_pd(_mm256_mul_pd(z_r_s_256, z_i_s_256), _mm256_set1_pd(2));
        z_r_s_256 = _mm256_add_pd(new_z_r_s_256, point_r_256);
        z_i_s_256 = _mm256_add_pd(new_z_i_s_256, point_i_256);

        __m256d magnitudes = _mm256_add_pd(_mm256_mul_pd(z_r_s_256,z_r_s_256), _mm256_mul_pd(z_i_s_256, z_i_s_256));

        __m256d comparers = _mm256_cmp_pd(magnitudes, thresholds, 29);
        double comparer_double[4];
        _mm256_storeu_pd(comparer_double, comparers);
        int temp_num_zeros = 0;
        for(int j = 0; j < 4; j++){
            if(comparer_double[j] == 0){
                temp_num_zeros++;
            }
        }
        num_zeros = temp_num_zeros;
        if(num_zeros == 4){
            return 4;
        }
    }
    return num_zeros;
}

void mandelbrot(struct parameters params, double scale, int32_t *num_pixels_in_set) {

    int32_t num_zero_pixels = 0;
    int i, j;

    for (i = params.resolution; i >= -params.resolution; i--) {
        // #pragma omp parallel for reduction(+:num_zero_pixels) private(j, flag, point_r, point_i, z_r, z_i, new_z_r, new_z_i)
        for (j = -params.resolution; j <= params.resolution; j += 4) {
            
            __m256d j_256 = _mm256_set_pd((double) (j+3),(double) (j+2),(double) (j+1),(double) (j));
            __m256d i_256 = _mm256_set1_pd(i);
            __m256d center_real_256 = _mm256_set1_pd(creal(params.center));
            __m256d center_imag_256 = _mm256_set1_pd(cimag(params.center));
            __m256d scale_256 = _mm256_set1_pd(scale);
            __m256d resolution_256 = _mm256_set1_pd(params.resolution);

            __m256d point_r_256 = _mm256_add_pd(center_real_256, _mm256_mul_pd(j_256, _mm256_div_pd(scale_256, resolution_256)));
            __m256d point_i_256 = _mm256_add_pd(center_imag_256, _mm256_mul_pd(i_256, _mm256_div_pd(scale_256, resolution_256)));
            __m256d thresholds = _mm256_set1_pd(params.threshold * params.threshold);

            double d[4];
            _mm256_storeu_pd(d,point_r_256);
            printf("%f %f %f %f \n", d[0], d[1], d[2], d[3]);

            int flag = iterations(point_r_256, point_i_256, thresholds, params);

            num_zero_pixels+=flag;
        }
    }
    *num_pixels_in_set = num_zero_pixels;
}
