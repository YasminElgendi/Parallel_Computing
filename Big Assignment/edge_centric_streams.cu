#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include "graph.h"
#include "timer.h"

#define STREAM_COUNT 4 // Number of CUDA streams

/*
 Edge Centric Approach:
    - Each thread is assigned to an edge 
    - We use the COO graph representation for this approach
    - The kernel will be called for each level in the graph
    - The update will happen for the edge that has a vertex in the previous level and a vertex in the current level that was not visited
    - No for loops => less control divergence
*/

__global__ void edge_centric_bfs(int *src, int *dst, int *level, int currentLevel, int edges, int *vertexVisited) {
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x; // each thread is assigned an edge (element in the array)

    if (edge < edges) {
        unsigned int sourceVertex = src[edge];
        unsigned int destinationVertex = dst[edge];

        if (level[sourceVertex] == currentLevel - 1 && level[destinationVertex] == -1) {
            level[destinationVertex] = currentLevel;
            *vertexVisited = 1;
        }
    }
}

int main(int argc, char *argv[]) {

    if (argc < 4) {
        printf("Please provide the paths of the input and output files and the source vertex\n");
        return 1;
    }

    FILE *inputFile;
    FILE *outputFile;
    int srcVertex = atoi(argv[3]);
    Timer timer;

    inputFile = fopen(argv[1], "r");
    outputFile = fopen(argv[2], "w");

    if (!inputFile || !outputFile) {
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

    // Initialize the level of each vertex to -1
    // and the source vertex to 0
    for (int i = 0; i < vertices; i++) {
        if (i == srcVertex)
            level[i] = 0;
        else
            level[i] = -1;
    }

    // Create a graph using the COO representation
    COOGraph(inputFile, src, dst, edges);

    // 2. Allocate device memory for the graph
    timer.start();
    int *deviceSrc, *deviceDst, *deviceLevel, *deviceVertexVisited;

    cudaMalloc((void **)&deviceSrc, edges * sizeof(int));
    cudaMalloc((void **)&deviceDst, edges * sizeof(int));
    cudaMalloc((void **)&deviceLevel, vertices * sizeof(int));
    cudaMalloc((void **)&deviceVertexVisited, sizeof(int));
    cudaDeviceSynchronize();
    timer.stop();
    double allocationTime = timer.elapsed();
    printf("Time taken to allocate device memory: %f ms\n", allocationTime);

    // Create streams
    cudaStream_t streams[STREAM_COUNT];
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // 3. Copy memory to the device asynchronously
    timer.start();
    int edgesPerStream = (edges + STREAM_COUNT - 1) / STREAM_COUNT;

    for (int i = 0; i < STREAM_COUNT; i++) {
        int offset = i * edgesPerStream;
        int edgeCount = (i == STREAM_COUNT - 1) ? (edges - offset) : edgesPerStream;

        cudaMemcpyAsync(deviceSrc + offset, src + offset, edgeCount * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(deviceDst + offset, dst + offset, edgeCount * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
    }
    cudaMemcpyAsync(deviceLevel, level, vertices * sizeof(int), cudaMemcpyHostToDevice, streams[0]);
    cudaDeviceSynchronize();
    timer.stop();
    double copyingTime = timer.elapsed();
    printf("Time taken to copy memory to the device: %f ms\n", copyingTime);

    // 4. Set the number of threads and blocks
    int threadsPerBlock = 256;
    unsigned int vertexVisited = 1;
    unsigned int currentLevel = 1; // Start from level 1

    // 5. Launch the kernel
    timer.start();
    while (vertexVisited) {
        vertexVisited = 0;
        cudaMemcpyAsync(deviceVertexVisited, &vertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice, streams[0]);

        for (int i = 0; i < STREAM_COUNT; i++) {
            int offset = i * edgesPerStream;
            int edgeCount = (i == STREAM_COUNT - 1) ? (edges - offset) : edgesPerStream;
            int blocksPerGrid = (edgeCount + threadsPerBlock - 1) / threadsPerBlock;

            edge_centric_bfs<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(deviceSrc + offset, deviceDst + offset, deviceLevel, currentLevel, edgeCount, deviceVertexVisited);
        }

        for (int i = 0; i < STREAM_COUNT; i++) {
            cudaStreamSynchronize(streams[i]);
        }

        cudaMemcpyAsync(&vertexVisited, deviceVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost, streams[0]);
        cudaStreamSynchronize(streams[0]);
        currentLevel++;
    }

    cudaDeviceSynchronize();
    timer.stop();
    double kernelTime = timer.elapsed();
    printf("\033[0;34m"); // Set color to blue
    printf("Kernel Time: %f ms\n", kernelTime);
    printf("\033[0m"); // Reset color

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // In red
        printf("\033[0;31m");
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        // Reset color
        printf("\033[0m");
    }

    // 6. Copy the result back to the host asynchronously
    cudaMemcpyAsync(level, deviceLevel, vertices * sizeof(int), cudaMemcpyDeviceToHost, streams[0]);
    cudaStreamSynchronize(streams[0]);

    // 7. Write the result to the output file
    for (int i = 0; i < vertices; i++) {
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
    printf("\033[0;32m"); // Set color to green
    printf("Total GPU time: %f ms\n", totalGPUTime);
    printf("\033[0m"); // Reset color

    // Destroy streams
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
