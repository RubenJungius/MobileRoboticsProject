'''
Implementation used from https://www.udacity.com/blog/2021/10/implementing-dijkstras-algorithm-in-python.html
'''

import sys

def findPath(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())
    shortest_path = {}
    previous_nodes = {}

    MAX_DIST = sys.maxsize # init all nodes to max distance for the path planning after
    for node in unvisited_nodes:
        shortest_path[node] = MAX_DIST
    shortest_path[start_node]=0 # manually set start node to 0

    while unvisited_nodes:
        current_min_node = None
        for node in unvisited_nodes: # Find current node with lowest Distance that has not been visited yet
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node
        
        neigbhors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neigbhors: # Update all adjacent nodes if old stored path is larger than new one
            tentive_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentive_value < shortest_path[neighbor]: # if new connection is shorter than the previously stored one
                shortest_path[neighbor] = tentive_value
                previous_nodes[neighbor] = current_min_node # update path to current min_node

        unvisited_nodes.remove(current_min_node) # Mark as visited

    return previous_nodes, shortest_path

def get_shortest(previous_nodes, shortest_path, start_node, target_node):
    path = []
    node = target_node
    while node != start_node:
        path.append(node)
        node = previous_nodes[node]
        
    # Add the start node manually
    path.append(start_node)
    return path
