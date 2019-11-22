import string 
import networkx as nx
import random

# 1. Create a set of location names using a location count
# 2. Create a set of locations that are the locations for the TAs
# 3. Create a starting point
# 4. Create a graph
#     a. we need to ensure connectedness
#         i. we need to make sure that there is a spanning tree and then we can add to it as necessary 
#     b. we need to ensure triangle inequality is satisfied
#         ii. we need to find the shortest paths from each vertex and if the shortest path from
#             it to another vertex is not a single line, then we have to add an additional line

def createLocationNames(locationCount):
    locationNames = set()
    while len(locationNames) < locationCount:
        res = locationNames.add(''.join(random.choices(string.ascii_uppercase + string.digits, k = N)))
    return list(locationNames)

def createTALocations(locations, taCount):
    taLocations = set()
    while len(taLocations) < taCount:
        res = taLocations.add(''.join(random.choices(string.ascii_uppercase + string.digits, k = N)))
    return list(taLocations)

def createGraph(locationNames, taLocations, numLocations, taCount):
    pass