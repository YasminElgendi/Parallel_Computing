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


__global__ void edge_centric_bfs(int *src, int *dst, int *level, int currentLevel, int edges, int *vertexVisited)
{
    unsigned int tidx = threadIdx.x;
    unsigned int edge = blockIdx.x * blockDim.x + tidx; // each thread is assigned an edge (element in the array)
    extern __shared__ int sharedLevel[];

    if (threadIdx.x < edges) {
        sharedLevel[tidx] = level[tidx];
    }
    __syncthreads();

    if (edge < edges) {
        unsigned int sourceVertex = src[edge];
        unsigned int destinationVertex = dst[edge];

        if (sharedLevel[sourceVertex] == currentLevel - 1 && sharedLevel[destinationVertex] == -1) {
            sharedLevel[destinationVertex] = currentLevel;
            *vertexVisited = 1;
        }
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

    // Initialize the level of each vertex to -1
    // and the source vertex to 0
    for (int i = 0; i < vertices; i++)
    {
        if (i == srcVertex)
            level[i] = 0;
        else
            level[i] = -1;
    }

    // Create a graph using the CSR representation

    // Construct the graph using the CSR representation
    COOGraph(inputFile, src, dst, edges);

    // printGraph(src, dst, edges, vertices, 0);

    // 2. Allocate device memory for the graph
    timer.start();
    int *deviceSrc;
    int *deviceDst;
    int *deviceLevel;
    int *deviceVertexVisited;

    cudaMalloc((void **)&deviceSrc, edges * sizeof(int));
    cudaMalloc((void **)&deviceDst, edges * sizeof(int));
    cudaMalloc((void **)&deviceLevel, vertices * sizeof(int));
    cudaMalloc((void **)&deviceVertexVisited, sizeof(int));
    cudaDeviceSynchronize();
    // printf("\nDevice memory allocated successfully\n");
    timer.stop();
    double allocationTime = timer.elapsed();
    printf("Time taken to allocate device memory: %f ms\n", allocationTime);

    // 3. Copy memory to the device
    timer.start();
    cudaMemcpy(deviceSrc, src, edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDst, dst, edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLevel, level, vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.stop();

    // printf("Copied to GPU successfully\n");
    double copyingTime = timer.elapsed();

    printf("Time taken to copy memory to the device: %f ms\n", copyingTime);

    // 4. Set the number of threads and blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (edges + threadsPerBlock - 1) / threadsPerBlock;
    unsigned int vertexVisited = 1;

    unsigned int currentLevel = 1; // we start from level 1 since we already set the level of the source vertex to 0

    // printf("Launching Kernel\n");
    // 5. Launch the kernel
    timer.start();
    unsigned int sharedMemorySize = threadsPerBlock * sizeof(int);
    while (vertexVisited)
    {
        vertexVisited = 0;
        cudaMemcpy(deviceVertexVisited, &vertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice); // copy the vertexVisited to the device before the launch of each kernel

        // kernel processes each level
        // the kernel will be called for each level in the graph
        // global synchronisation across different levels
    
        edge_centric_bfs<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(deviceSrc, deviceDst, deviceLevel, currentLevel, edges, deviceVertexVisited);

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
    free(src);
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