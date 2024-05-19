#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include "graph.h"
#include "timer.h"

/*
 Edge Centric Appraoch:
    - Each thread is assigned to an edge 
    - We use the COO graph representation for this approach
    - The kernel will be called for each level in the graph
    - The update will happen for the edge that has a vertex in the precious level and a vertex in the current level that was not visited
    - No for loops => less control divergence
*/



__host__ void bfs_cpu(int *src, int *dst, int *level, int vertices, int edges, int srcVertex)
{
    unsigned int currentLevel = 1;
    unsigned int vertexVisited = 1;

    while (vertexVisited)
    {
        vertexVisited = 0;

        for (int i = 0; i < edges; i++)
        {
            int sourceVertex = src[i];
            int destinationVertex = dst[i];

            if (level[sourceVertex] == currentLevel - 1 && level[destinationVertex] == UINT_MAX)
            {
                level[destinationVertex] = currentLevel;
                vertexVisited = 1;
            }
        }

        currentLevel++;
    }
}


int main(char argc, char *argv[])
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

    int *src = (int *)malloc(edges * sizeof(int));
    int *dst = (int *)malloc(edges * sizeof(int));
    int *level = (int *)malloc(vertices * sizeof(int));
    // unsigned int *vertexVisited = (unsigned int *)malloc(sizeof(unsigned int));

    // Create a graph using the COO representation
    // Construct the graph using the COO representation
    COOGraph(inputFile, src, dst, edges);

    // printGraph(src, dst, edges, vertices, 0);
    int *levelCPU = ( int *)malloc(vertices * sizeof( int));

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

    // Run the CPU BFS
    timer.start();
    bfs_cpu(src, dst, levelCPU, vertices, edges,srcVertex);
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
    
    // 7. Write the result to the output file
    for (int i = 0; i < vertices; i++)
    {
        fprintf(outputFile, "%d %d\n", i, level[i]);
    }

    // Close files
    fclose(inputFile);
    fclose(outputFile);

    // Free host memory
    free(src);
    free(dst);
    free(level);

   

    return 0;
}