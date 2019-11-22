import string
import networkx as nx
import matplotlib.pyplot as plt
import random

# 1. Create a set of location names using a location count
# 2. Create a set of locations that are the locations for the TAs
# 3. Create a starting point
# 4. Create a graph
#     a. we need to ensure connectedness
#         i. we need to make sure that there is a spanning tree and then we can add to it as necessary as necessary
#     b. we need to ensure triangle inequality is satisfied
#         ii. we need to find the shortest paths from each vertex to all other vertices
#             1) If there is an edge from that vertex to the other vertex, we check if it satisfies the triangle inequality with the shortest path using Floyd Warshall

def createLocationNames(locationCount):
    N = 5
    locationNames = set()
    while len(locationNames) < locationCount:
        locationNames.add(''.join(random.choices(string.ascii_uppercase + string.digits, k = N)))
    return list(locationNames)

def createTALocations(locations, taCount):
    taLocations = set()
    while len(taLocations) < taCount:
        taLocations.add(random.choice(locations))
    return list(taLocations)

def createGraph(locationNames, taLocations, numLocations, num_neighbors, volatility, max_weight):
    G = nx.connected_watts_strogatz_graph(numLocations, num_neighbors, volatility)
    mapping = dict(zip(G.nodes(), locationNames))
    mapping2 = dict(zip(locationNames, G.nodes()))
    G = nx.relabel_nodes(G, mapping)
    for edge in G.edges():
        u = edge[0]
        v = edge[1]
        G[u][v]['weight'] = random.randint(1, max_weight)
    return G, mapping2

def drawGraph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

def do_shortest_paths(G):
    r = nx.shortest_path(G)
    return r

def adjust_edge_weights(G):
    for edge in G.edges.data():
        u = edge[0]
        v = edge[1]
        info = edge[2]
        weight = info.get('weight', 0)
        shortest_path, _ = nx.single_source_dijkstra(G, source=u, target=v, cutoff=None, weight='weight')
        if (weight > shortest_path):
            weight = shortest_path - 1
        G[u][v]['weight'] = weight

def graph_to_adjacency(G, mapping, num_locations):
    edges = G.edges.data()
    adj_mat = [[0 for i in range(num_locations)] for i in range(num_locations)]

    for edge in edges:
        in_vert = edge[0]
        out_vert = edge[1]
        info = edge[2]

        num_of_in = mapping[in_vert]
        num_of_out = mapping[out_vert]

        adj_mat[num_of_out][num_of_in] = info.get('weight', 0)
        adj_mat[num_of_in][num_of_out] = info.get('weight', 0)

    return adj_mat

def adj_to_string(adj_mat):
    s = ""
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat[0])):
            s += (str(adj_mat[i][j]) + " ") if adj_mat[i][j] else "x "
        s += "\n"
    return s

def string_entire(loc, ta, start, graph_str):
    s = ""
    s += str(len(loc)) + "\n"
    s += str(len(ta)) + "\n"
    for location in loc:
        s += location + " "
    s += "\n"
    for tas in ta:
        s += tas + " "
    s += "\n"
    s += start + "\n"
    s += graph_str
    return s

if __name__ == "__main__":
    num_loc = 7
    num_ta = 5
    num_neighbors = 3
    volatility = 1
    max_weight = 9

    loc = createLocationNames(num_loc)
    ta_loc = createTALocations(loc, num_ta)

    G, mapping = createGraph(loc, ta_loc, num_loc, num_neighbors, volatility, max_weight)
    adj_mat = graph_to_adjacency(G, mapping, num_loc)
    s = adj_to_string(adj_mat)
    final_str = string_entire(loc, ta_loc, random.choice(loc), s)
    print(final_str)
    ret = do_shortest_paths(G)

    adjust_edge_weights(G)


    adj_mat = graph_to_adjacency(G, mapping, num_loc)
    s = adj_to_string(adj_mat)
    final_str = string_entire(loc, ta_loc, random.choice(loc), s)
    print(final_str)
    print(mapping)
    drawGraph(G)
