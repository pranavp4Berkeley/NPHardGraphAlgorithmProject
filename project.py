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
        res = locationNames.add(''.join(random.choices(string.ascii_uppercase + string.digits, k = N)))
    return list(locationNames)

def createTALocations(locations, taCount):
    taLocations = set()
    while len(taLocations) < taCount:
        taLocations.add(random.choice(locations))
    return list(taLocations)

def createGraph(locationNames, taLocations, numLocations, taCount, num_neighbors, volatility):
    G = nx.connected_watts_strogatz_graph(numLocations, num_neighbors, volatility)
    mapping = dict(zip(G.nodes(), locationNames))
    mapping2 = dict(zip(locationNames, G.nodes()))
    G = nx.relabel_nodes(G, mapping)
    for edge in G.edges():
        u = edge[0]
        v = edge[1]
        G[u][v]['weight'] = random.randint(1, 10)
    return G, mapping2

def drawGraph(G):
    #nx.draw(G, with_labels=True, font_weight='bold')
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G,pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

def do_shortest_paths(G):
    r = nx.shortest_path(G)
    return r

def graph_to_adjacency(G, mapping, num_locations):
    edges = G.edges.data()
    adj_mat = [[0 for i in range(num_locations)] for i in range(num_locations)]

    for edge in edges:
        in_vert = edge[0]
        out_vert = edge[1]
        info = edge[2]
        
        num_of_in = mapping[in_vert]
        num_of_out = mapping[out_vert]

        adj_mat[num_of_out][num_of_in] = info.get("weight", 0)
        adj_mat[num_of_in][num_of_out] = info.get("weight", 0)
    
    return adj_mat

def adj_to_string(adj_mat):
    s = ""
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat[0])):
            s += (str(adj_mat[i][j]) + " ") if adj_mat[i][j] else "x "
        s += "\n"
    return s

if __name__ == "__main__":
    num_loc = 20
    num_ta = 10
    num_neighbors = 5
    volatility = 1

    loc = createLocationNames(num_loc)
    ta_loc = createTALocations(loc, num_ta)
    G, mapping = createGraph(loc, ta_loc, num_loc, num_ta, num_neighbors, volatility)
    adj_mat = graph_to_adjacency(G, mapping, num_loc)
    s = adj_to_string(adj_mat)
    ret = do_shortest_paths(G)
    print(s)
    drawGraph(G)
