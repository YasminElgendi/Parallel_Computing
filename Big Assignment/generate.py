import sys
import random
import networkx as nx
from tqdm import tqdm


def main():
    output_folder = "graphs/"

    if len(sys.argv) != 4:
        print("Usage: python generate_graph.py vertices edges")
        return

    num_of_vertices = int(sys.argv[1])
    num_of_edges = int(sys.argv[2])
    output_file = sys.argv[3]

    # check if the the number of edges follow the contraints
    if num_of_edges < num_of_vertices - 1 or num_of_edges > num_of_vertices*(num_of_vertices-1)/2:
        print("Number of edges is not possible for the given number of vertices")
        return

    # Start with a spanning tree to ensure connectivity
    graph = nx.Graph()
    graph.add_nodes_from(range(num_of_vertices))

    # Create a spanning tree
    nodes = list(range(num_of_vertices))
    random.shuffle(nodes)
    
    for i in tqdm(range(num_of_vertices - 1)):
        u = nodes[i]
        v = nodes[i + 1]
        graph.add_edge(u, v)

    # Add additional random edges until reaching the desired number of edges
    # tbar = tqdm(range(graph.number_of_edges()))
    while graph.number_of_edges() < num_of_edges:
        u = random.randint(0, num_of_vertices - 1)
        v = random.randint(0, num_of_vertices - 1)

        if u != v and not graph.has_edge(u, v):
            graph.add_edge(u, v)

    print(
        f"Generated a graph with {num_of_vertices} vertices and {num_of_edges} edges.")

    # create a list of edges with both (u, v) and (v, u) for each edge and sort it by the first element
    edges = []
    for edge in graph.edges():
        edges.append(edge)
        edges.append((edge[1], edge[0]))

    # sort the list of edges using the first element of each edge
    edges.sort(key=lambda x: x[0])

    # Print the edges of the graph
    with open(output_folder + output_file, 'w') as csr:
        csr.write(f"{num_of_vertices} {num_of_edges*2}\n")
        for edge in edges:
            csr.write(f"{edge[0]} {edge[1]}\n")

    csr.close()



if __name__ == "__main__":
    main()


# python generate.py 100 200 graph_100.txt (100 = number of vertices)
