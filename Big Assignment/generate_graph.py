import sys
import numpy as np

# the file structure for the graph should be
# n m   where n is the number of vertices and m is the number of edges
# 1 2   the next m lines will be the edges in the graph where the first number is the source node and the second is the destination
# 1 3
# 2 3
# ...

# the script should take the number of nodes and edges and the output filename as arguments
# n m output.txt


def main():
    # Check if correct number of arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python generate_graph.py vertices edges")
        return

    num_of_vertices = int(sys.argv[1])
    num_of_edges = int(sys.argv[2])
    output_file = sys.argv[3]

    # check if number of edges is less than the number of vertices
    # the number of edges should be greater than or equal to the number of vertices
    if num_of_edges > num_of_vertices*(num_of_vertices-1)/2:
        print("Number of edges is not possible for the given number of vertices")
        return

    # generate random class
    vertices = np.arange(num_of_vertices)

    # few adjustmens
    # it has to be a fully connected graph so each vertex is written at least once
    # sort the array before writing to the file

    sources = []
    destinations = []

    with open(output_file, 'w') as f:
        # write the number of vertices and edges
        f.write(str(num_of_vertices) + ' ' + str(num_of_edges) + '\n')
        # write at least one edge for each vertex
        for vertex in vertices:
            dest = np.random.choice(vertices)
            while dest == vertex:
                dest = np.random.choice(vertices)

            sources.append(vertex)
            destinations.append(dest)

        # update the number of edges
        num_of_edges -= num_of_vertices

        for _ in range(num_of_edges):
            src = np.random.choice(vertices)
            dest = np.random.choice(vertices)
            while src == dest:
                dest = np.random.choice(vertices)

            # check if the edge already exists
            while (src, dest) in zip(sources, destinations):
                src = np.random.choice(vertices)
                dest = np.random.choice(vertices)
                while src == dest:
                    dest = np.random.choice(vertices)

            sources.append(src)
            destinations.append(dest)

        # sort the arrays based on the sources
        sources, destinations = zip(*sorted(zip(sources, destinations)))

        for src, dest in zip(sources, destinations):
            f.write(str(src) + ' ' + str(dest) + '\n')

    f.close()

    print("Graph generated successfully")


if __name__ == '__main__':
    main()
