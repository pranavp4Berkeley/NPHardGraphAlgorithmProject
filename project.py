import string
import networkx as nx
import matplotlib.pyplot as plt
import random
N = 10

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
    N = 10
    locationNames = set()
    while len(locationNames) < locationCount:
        res = locationNames.add(''.join(random.choices(string.ascii_uppercase + string.digits, k = N)))
    return list(locationNames)

def createTALocations(locations, taCount):
    taLocations = set()
    while len(taLocations) < taCount:
        taLocations.add(random.choice(locations))
    return list(taLocations)

def createGraph(locationNames, taLocations, numLocations, taCount):
    G = nx.connected_watts_strogatz_graph(N, 5, 1)
    mapping = dict(zip(G.nodes(), locationNames))
    G = nx.relabel_nodes(G, mapping)
    for edge in G.edges():
        u = edge[0]
        v = edge[1]
        G[u][v]['weight'] = random.randint(1, 10)
    return G

def drawGraph(G):
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

if __name__ == "__main__":
    loc = createLocationNames(10)
    ta_loc = createTALocations(loc, 5)
    G = createGraph(loc, ta_loc, 10, 5)
    drawGraph(G)
