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
 Vertex Centric Direction-Based Approach:
 - This is a hybrid implementation of the push and pull approaches
 - It starts with the top-down approach and then switches to the bottom-up approach
 - At first the bottom-up approach is inefficient since it has to check all the neighbours of the vertex and the vertex will be not any in the previous level
 - In the Initial iterations th etop-down approach is more efficient since there are very few neighbours in the previous level
 - To implement the hybrid approach the kernels will be called based on the current level
 - The cuda kernels will not be changed
 - Only the calling of the kernels in the main function will be changed
*/

__global__ void vertex_centric_pull_bfs(unsigned int *srcPtrs, unsigned int *dst, unsigned int *level, int currentLevel, int vertices, int edges, unsigned int *vertexVisited)
{
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex < vertices)
    {
        if (level[vertex] == UINT_MAX)
        {
            // check if my neighbours are in the previous level
            unsigned int start = srcPtrs[vertex];
            unsigned int end = srcPtrs[vertex + 1];
            for (unsigned int i = start; i < end; i++)
            {
                unsigned int neighbour = dst[i];
                if (level[neighbour] == currentLevel - 1)
                {
                    level[vertex] = currentLevel;
                    *vertexVisited = 1;
                    break; // if one of the vertex neighbours has been visited then no need to check the rest of the neighbours (we're just setting the level of the current vertex)
                }
            }
        }
    }
}

__global__ void vertex_centric_push_bfs(unsigned int *srcPtrs, unsigned int *dst, unsigned int *level, int currentLevel, int vertices, int edges, unsigned int *vertexVisited)
{
    // each thread is assigned a vertex
    // since this is considered a 1-D array we will use the the x index to get the vertex for each thread
    // following the basis of a vector addition
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary conditions
    // check if the vertex is inside the graph
    if (vertex < vertices)
    {
        // printf("Vertex: %d, Vertex Visited: %d\n", vertex, *vertexVisited);
        if (level[vertex] == currentLevel - 1) // then loop on all neighbours of the vertex
        {
            // get the starting and ending index of the edges of the vertex
            // the srcPtrs contain the starting index of the edges of the vertex for each row (vertex)
            unsigned int start = srcPtrs[vertex];
            unsigned int end = srcPtrs[vertex + 1];

            // iterate over the neighbours of the vertex
            for (unsigned int i = start; i < end; i++)
            {
                unsigned int neighbour = dst[i];
                // printf("Vertex: %d, Neighbour: %d, Level: %d, Current Level: %d\n", vertex, neighbour, level[neighbour], currentLevel);
                // check if the neighbour has not been visited
                if (level[neighbour] == UINT_MAX)
                {
                    level[neighbour] = currentLevel;
                    *vertexVisited = 1;
                }
            }
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

    unsigned int *srcPtrs = (unsigned int *)malloc((vertices + 1) * sizeof(unsigned int));
    unsigned int *dst = (unsigned int *)malloc(edges * sizeof(unsigned int));
    unsigned int *level = (unsigned int *)malloc(vertices * sizeof(unsigned int));
    // unsigned int *vertexVisited = (unsigned int *)malloc(sizeof(unsigned int));

    // printf("Host memory allocated successfully\n");

    // Initialize the level of each vertex to -1
    // and the source vertex to 0
    for (int i = 0; i < vertices; i++)
    {
        if (i == srcVertex)
            level[i] = 0;
        else
            level[i] = UINT_MAX;
    }

    // Create a graph using the CSR representation

    // Construct the graph using the CSR representation
    CSRGraph(inputFile, srcPtrs, dst, edges);

    // 2. Allocate device memory for the graph
    timer.start();
    unsigned int *deviceSrc;
    unsigned int *deviceDst;
    unsigned int *deviceLevel;
    unsigned int *deviceVertexVisited;

    cudaMalloc((void **)&deviceSrc, (vertices + 1) * sizeof(unsigned int));
    cudaMalloc((void **)&deviceDst, edges * sizeof(unsigned int));
    cudaMalloc((void **)&deviceLevel, vertices * sizeof(unsigned int));
    cudaMalloc((void **)&deviceVertexVisited, sizeof(unsigned int));

    // printf("\nDevice memory allocated successfully\n");
    timer.stop();
    double allocationTime = timer.elapsed();
    printf("Time taken to allocate device memory: %f ms\n", allocationTime);

    // 3. Copy memory to the device
    timer.start();
    cudaMemcpy(deviceSrc, srcPtrs, (vertices + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDst, dst, edges * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLevel, level, vertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
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
        vertexVisited = 0; // reset the vertexVisited to 0 for each level

        // copy the vertexVisited to the device before the launch of each kernel
        cudaMemcpy(deviceVertexVisited, &vertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);

        // kernel processes each level
        // the kernel will be called for each level in the graph
        // global synchronisation across different levels


        if (currentLevel == 1) // if the current vertex is 1 then use the top-down approach (can be done for the first few levels)
            vertex_centric_push_bfs<<<blocksPerGrid, threadsPerBlock>>>(deviceSrc, deviceDst, deviceLevel, currentLevel, vertices, edges, deviceVertexVisited);
        else
            vertex_centric_pull_bfs<<<blocksPerGrid, threadsPerBlock>>>(deviceSrc, deviceDst, deviceLevel, currentLevel, vertices, edges, deviceVertexVisited);

        
        // copy the vertexVisited back to the host after the kernel has finished to check whether any vertex has been visited if not the max depth reached
        cudaMemcpy(&vertexVisited, deviceVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
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