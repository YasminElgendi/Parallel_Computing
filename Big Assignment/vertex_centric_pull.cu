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
 Vertex Centric Paralleisation of BFS:
 Bottom-Up Approach:
    - The second approach is the vertex-centric pull-based BFS
    - Which is a bottom-up approach
    - Each thread handles a vertex but instead of updating the levels of the neighbours in the next level, it updates the level of the current vertex by checking the level of  the neighbours of the previous level
    - This approach has less control divergence since it breaks the loop if a single neighbour has been visited and updates the level of the current vertex for a vertex that has very large neighborhoods
*/

__global__ void vertex_centric_pull_bfs(int *srcPtrs, int *dst, int *level, int currentLevel, int vertices, int edges, int *vertexVisited)
{
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex < vertices)
    {
        if (level[vertex] == -1)
        {
            // check if my neighbours are in the previous level
            int start = srcPtrs[vertex];
            int end = srcPtrs[vertex + 1];
            for (int i = start; i < end; i++)
            {
                int neighbour = dst[i];
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

    int *srcPtrs = (int *)malloc((vertices + 1) * sizeof(int));
    int *dst = (int *)malloc(edges * sizeof(int));
    int *level = (int *)malloc(vertices * sizeof(int));
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

    // Create a graph using the CSR representation

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
        vertex_centric_pull_bfs<<<threadsPerBlock, blocksPerGrid>>>(deviceSrc, deviceDst, deviceLevel, currentLevel, vertices, edges, deviceVertexVisited);

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