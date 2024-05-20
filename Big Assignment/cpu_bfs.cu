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
#include "timer.h"

__host__ void bfs_cpu(unsigned int *srcPtrs, unsigned int *dst, unsigned int *level, int srcVertex)
{
    // this function will be used to compare the results of the GPU with the CPU
    // define a queue to store the vertices
    std::queue<int> verticesQueue;

    // vector<bool> visited(vertices, false);

    // mark the current node as visited and enqueue it
    level[srcVertex] = 0;
    verticesQueue.push(srcVertex);

    while (!verticesQueue.empty())
    {
        // dequeue a vertex from queue and print it
        int currentVertex = verticesQueue.front();
        verticesQueue.pop();

        int start = srcPtrs[currentVertex];
        int end = srcPtrs[currentVertex + 1];

        // get all adjacent vertices of the dequeued vertex
        // if an adjacent has not been visited, then mark it visited and enqueue it
        for (int i = start; i < end; i++)
        {
            int neighbour = dst[i];
            if (level[neighbour] == UINT_MAX)
            {
                verticesQueue.push(neighbour);
                level[neighbour] = level[currentVertex] + 1;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Please provide the paths of the input and output files and the source vertex\n");
        return 1;
    }

    FILE *inputFile;
    FILE *outputFile;
    int srcVertex = atoi(argv[3]);
    Timer timer;

    inputFile = fopen(argv[1], "r");
    outputFile = fopen(argv[2], "w");

    if (!inputFile || !outputFile)
    {
        printf("Please provide the correct path of both files");
        return 1;
    }

    // 1. Allocate host memory for the graph
    int vertices, edges;

    // Read the number of vertices and edges
    fscanf(inputFile, "%d %d", &vertices, &edges);

    unsigned int *srcPtrs = (unsigned int *)malloc((vertices + 1) * sizeof(unsigned int)); // allocate with the actual number of source vertices => directed graph
    unsigned int *dst = (unsigned int *)malloc(edges * sizeof(unsigned int));
    unsigned int *level = (unsigned int *)malloc(vertices * sizeof(unsigned int));
    unsigned int *levelCPU = (unsigned int *)malloc(vertices * sizeof(unsigned int));

    // Initialize the level of each vertex to -1
    // and the source vertex to 0
    for (int i = 0; i < vertices; i++)
    {
        if (i == srcVertex)
        {
            level[i] = 0;
            levelCPU[i] = 0;
        }
        else
        {
            level[i] = UINT_MAX;
            levelCPU[i] = UINT_MAX;
        }
    }

    // Construct the graph using the CSR representation
    CSRGraph(inputFile, srcPtrs, dst, edges);

    // Run the CPU BFS
    timer.start();
    bfs_cpu(srcPtrs, dst, levelCPU, srcVertex);
    timer.stop();

    // 7. Write the result to the output file
    FILE * cpuOutputFile = fopen("output/cpu_output.txt", "w");
    for (int i = 0; i < vertices; i++)
    {
        fprintf(cpuOutputFile, "%d %d\n", i, levelCPU[i]);
    }

    double cpuTime = timer.elapsed();
    printf("\033[0;33m"); // set color to yellow
    printf("CPU Time: %f ms\n", cpuTime);
    printf("\033[0m"); // reset color

    return 0;
}