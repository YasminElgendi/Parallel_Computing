#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <queue>
#include "graph.h"

__host__ vector<int> BFS(Graph &graph, int startVertex)
{
    // create a queue for BFS
    queue<int> bfsQueue;
    vector<bool> visited(graph.numberOfVertices, false);
    vector<int> bfsTraversal;

    // mark the current node as visited and enqueue it
    visited[startVertex] = true;
    bfsQueue.push(startVertex);

    while (!bfsQueue.empty())
    {
        // dequeue a vertex from queue and print it
        int currentVertex = bfsQueue.front();
        bfsQueue.pop();

        bfsTraversal.push_back(currentVertex);

        // get all adjacent vertices of the dequeued vertex
        // if an adjacent has not been visited, then mark it visited and enqueue it
        for (int i = 0; i < graph.adjacencyList[currentVertex].size(); i++)
        {
            int adjacentVertex = graph.adjacencyList[currentVertex][i];
            if (!visited[adjacentVertex])
            {
                visited[adjacentVertex] = true;
                bfsQueue.push(adjacentVertex);
            }
        }
    }

    return bfsTraversal;
}

int main(int argc, char *argv[])
{
    // read the filename from the command line

    // declarations
    char *filename = argv[1];
    FILE *file = fopen(filename, "r");
    Graph graph;
    int vertices, edges;
    vector<int> bfsGraph;

    createGraph(file, graph, vertices, edges); // reads and creates graph from the file given
    // printGraph(graph);                         // prints the graph

    // BFS traversal
    bfsGraph = BFS(graph, 0);

    for (int i = 0; i < bfsGraph.size(); i++)
    {
        printf("%d ", bfsGraph[i]);
    }

    return 0;
}