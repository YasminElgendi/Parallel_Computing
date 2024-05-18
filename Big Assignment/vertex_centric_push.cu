#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include "graph.h"
#include "timer.h"

// Objective of BFS: To find the shortest path from a source vertex to all other vertices in an unweighted graph
// output file: the line number corresponds to the vertex number and the value corresponds to the level of the vertex from the source vertex

/*
 Vertex Centric Paralleisation of BFS:
    1. Each vertex is assigned a thread
    2. Each thread processes the vertex and its neighbours

    A vertex-centric parallel implementation assigns threads to vertices and has each thread perform an operation on its vertex, which usually involves
    iterating over the neighbors of that vertex

    There are 2 implementations:
    1. Focuses on the outgoing edges of the vertex (top-down approach)
        Since the CSR format stores the outgoing edges of a vertex, we eill use it for this approach
    2. Focuses on the incoming edges of the vertex

    The function will be called for each level in the graph

*/

// For later use: cudaDeviceSynchronize() => Waits for all kernels in all streams on a CUDA device to complete.

__global__ void vertex_centric_push_bfs(int *srcPtrs, int *dst, int *level, int currentLevel, int vertices, int edges, int *vertexVisited)
{
    // each thread is assigned a vertex
    // since this is considered a 1-D array we will use the the x index to get the vertex for each thread
    // following the basis of a vector addition
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary conditions
    // check if the vertex is inside the graph
    if (vertex < vertices)
    {
        // printf("Vertex: %d, Level[%d]: %d, Current Level: %d\n", vertex, vertex, level[vertex], currentLevel);
        if (level[vertex] == currentLevel - 1) // then loop on all neighbours of the vertex
        {
            // printf("Vertex: %d, Vertex Visited: %d\n", vertex, *vertexVisited);
            // get the starting and ending index of the edges of the vertex
            // the srcPtrs contain the starting index of the edges of the vertex for each row (vertex)
            int start = srcPtrs[vertex];
            int end = srcPtrs[vertex + 1];

            // printf("Vertex: %d, Start: %d, End: %d\n", vertex, start, end);

            // iterate over the neighbours of the vertex
            for (int i = start; i < end; i++)
            {
                int neighbour = dst[i];
                // check if the neighbour has not been visited
                if (level[neighbour] == -1)
                {
                    // printf("Vertex: %d, Neighbour: %d, Level[%d]: %d, Current Level: %d\n", vertex, neighbour, neighbour, level[neighbour], currentLevel);
                    level[neighbour] = currentLevel;
                    *vertexVisited = 1;
                    // printf("Vertex: %d, Neighbour: %d, Level[%d]: %d, Current Level: %d\n", vertex, neighbour, neighbour, level[neighbour], currentLevel);
                    // printf("Vertex: %d, Vertex Visited: %d\n", vertex, *vertexVisited);
                }
            }
            // printf("Vertex: %d, Vertex Visited: %d\n", vertex, *vertexVisited);
        }
    }
}

__host__ void cpu_bfs(int *srcPtrs, int *dst, int *level, int vertices, int edges)
{
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

    int *srcPtrs = (int *)malloc((vertices + 1) * sizeof(int)); // allocate with the actual number of source vertices => directed graph
    int *dst = (int *)malloc(edges * sizeof(int));
    int *level = (int *)malloc(vertices * sizeof(int));
    int *srcNames = (int *)malloc((vertices + 1) * sizeof(int));
    // unsigned int *vertexVisited = (unsigned int *)malloc(sizeof(unsigned int));

    // printf("Host memory allocated successfully\n");

    // Initialize the level of each vertex to -1
    // and the source vertex to 0
    for (int i = 0; i < vertices; i++)
    {
        if (i == srcVertex)
            level[i] = 0;
        else
            level[i] = -1;
    }

    // Construct the graph using the CSR representation
    CSRGraph(inputFile, srcPtrs, dst, edges);

    // 2. Allocate device memory for the graph
    timer.start();
    int *deviceSrc;
    int *deviceDst;
    int *deviceLevel;
    int *deviceVertexVisited;

    cudaMalloc((void **)&deviceSrc, (vertices + 1) * sizeof(int));
    cudaMalloc((void **)&deviceDst, edges * sizeof(int));
    cudaMalloc((void **)&deviceLevel, vertices * sizeof(int));
    cudaMalloc((void **)&deviceVertexVisited, sizeof(int));

    // printf("\nDevice memory allocated successfully\n");
    timer.stop();
    double allocationTime = timer.elapsed();
    printf("Time taken to allocate device memory: %f ms\n", allocationTime);

    // 3. Copy memory to the device
    timer.start();
    cudaMemcpy(deviceSrc, srcPtrs, (vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDst, dst, edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLevel, level, vertices * sizeof(int), cudaMemcpyHostToDevice);
    timer.stop();

    // printf("Copied to GPU successfully\n");
    double copyingTime = timer.elapsed();

    printf("Time taken to copy memory to the device: %f ms\n", copyingTime);

    // 4. Set the number of threads and blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (vertices + threadsPerBlock - 1) / threadsPerBlock;
    unsigned vertexVisited = 1;

    int currentLevel = 1; // we start from level 1 since we already set the level of the source vertex to 0

    // printf("Launching Kernel\n");
    // 5. Launch the kernel
    timer.start();
    while (vertexVisited)
    {
        vertexVisited = 0;
        cudaMemcpy(deviceVertexVisited, &vertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice); // copy the vertexVisited to the device before the launch of each kernel

        // kernel processes each level
        // the kernel will be called for each level in the graph
        // global synchronisation across different levels
        vertex_centric_push_bfs<<<threadsPerBlock, blocksPerGrid>>>(deviceSrc, deviceDst, deviceLevel, currentLevel, vertices, edges, deviceVertexVisited);

        cudaMemcpy(&vertexVisited, deviceVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost); // copy the vertexVisited back to the host after the kernel has finished to check whether any vertex has been visited if not the max depth reached
        currentLevel++;
    }

    cudaDeviceSynchronize(); // wai for all kernels to finish so that the level array is updated
    timer.stop();
    double kernelTime = timer.elapsed();
    printf("\033[0;34m"); // set colour to blue
    printf("Kernel Time: %f ms\n", kernelTime);
    printf("\033[0m"); // reset color

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        // in red
        printf("\033[0;31m");
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        // reset color
        printf("\033[0m");
    }

    // printf("Kernel executed successfully\n");

    // 6. Copy the result back to the host
    cudaMemcpy(level, deviceLevel, vertices * sizeof(int), cudaMemcpyDeviceToHost);

    // printf("Copied back to host successfully\n");

    // 7. Write the result to the output file
    for (int i = 0; i < vertices; i++)
    {
        fprintf(outputFile, "%d %d\n", i, level[i]);
    }

    // Close files
    fclose(inputFile);
    fclose(outputFile);

    // Free host memory
    free(srcPtrs);
    free(dst);
    free(level);

    // Free device memory
    timer.start();
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
    cudaFree(deviceLevel);
    cudaFree(deviceVertexVisited);
    timer.stop();
    double deallocationTime = timer.elapsed();

    printf("Time taken to deallocate device memory: %f ms\n", deallocationTime);

    double totalGPUTime = allocationTime + copyingTime + kernelTime + deallocationTime;

    printf("\033[0;32m"); // set color to green
    printf("Total GPU time: %f ms\n", totalGPUTime);
    printf("\033[0m"); // reset color

    return 0;
}