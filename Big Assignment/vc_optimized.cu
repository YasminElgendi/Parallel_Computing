#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <queue>
#include "graph.h"
#include "timer.h"

#define LOCAL_QUEUE_SIZE 2048

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

    The optimized version of the vertex centric push BFS will use the following:
    1. Frontiers: We will make use of frontiers to reduce redundant work:
        - The redundency from the previous implementation results from that each thread will loop over all the neighbours of the vertex to find them
    2. Privatization: shared memory will be used to store the frontier => to deal with atomic operations
        - since the frontier is shared with all threads in the same block we will use shared memory to store the frontier
        - All threads are atomically incrementing the same global counter to insert elements in the queue => high latency (global memory access) and serialization (contention)
        - Each block will commit to a private queue and then commit entrie to global queue
    3. Minimized Launch Overhead
        - The kernel is only called when the number of vertices of two consecutive queues combined is less than the number of threads in a single block
        - This is to minimize the overhead of the kernel launch
        - This will create a local level array for the block and synchronize at the end
*/

// For later use: cudaDeviceSynchronize() => Waits for all kernels in all streams on a CUDA device to complete.

__global__ void vertex_centric_optimized_bfs(unsigned int *srcPtrs, unsigned int *dst, unsigned int *level, unsigned int *currentQueue,
                                             unsigned int *previousQueue, int currentLevel, unsigned int *numberOfCurrentQueue, int numberOfPreviousQueue,
                                             int vertices, int edges, unsigned int *numberOfLevels)
{
    // define the shared memory for the frontier
    __shared__ unsigned int sharedCurrentQueue[LOCAL_QUEUE_SIZE];
    __shared__ unsigned int sharedNumCurrentFrontier;
    __shared__ unsigned int sharedNumLevels;

    // only need a single thread to initialize the shared memory not all but make sure all threads are synchronized
    if (threadIdx.x == 0)
    {
        // printf("Regular kernel\n");
        sharedNumLevels = 1;
        sharedNumCurrentFrontier = 0;
    }

    __syncthreads();

    // Perform BFS on local queue
    // each thread is assigned a vertex
    // since this is considered a 1-D array we will use the the x index to get the vertex for each thread
    // following the basis of a vector addition
    unsigned int vertexIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary conditions => check if the vertex within the previous queue
    if (vertexIndex < numberOfPreviousQueue)
    {
        unsigned int vertex = previousQueue[vertexIndex]; // get the vertex from the previous queue

        unsigned int start = srcPtrs[vertex];   // get the start index of the neighbours of the vertex
        unsigned int end = srcPtrs[vertex + 1]; // get the end index of the neighbours of the vertex

        // iterate over the neighbours of the vertex
        for (unsigned int i = start; i < end; i++)
        {
            unsigned int neighbour = dst[i];
            // the atomic compare and swap doesnt change since the level array is still global and shared across all threads
            // what happens if multiple threads in the previous queue have the same neighbours => race condition (can be added to the current queue multiple times)
            if (atomicCAS(&level[neighbour], UINT_MAX, currentLevel) == UINT_MAX)
            {
                // Add to the local queue
                unsigned int sharedIndex = atomicAdd(&sharedNumCurrentFrontier, 1); // get the index of the local current queue

                // Check if the shared memory is full
                if (sharedIndex < LOCAL_QUEUE_SIZE)
                {
                    sharedCurrentQueue[sharedIndex] = neighbour;
                }
                else
                {
                    // dealing with overflow
                    // since the shared memory is of limited size we will add the neighbour to the global queue if the shared memory is full

                    sharedNumCurrentFrontier = LOCAL_QUEUE_SIZE;             // set the number of the current queue to the max size
                    unsigned int index = atomicAdd(numberOfCurrentQueue, 1); // add the neighbour to the current queue
                    currentQueue[index] = neighbour;
                }
            }
        }
    }
    __syncthreads(); // wait for all threads before copying the shared memory to the global memory

    // copy the shared memory to the global memory
    __shared__ unsigned int sharedStartIndex; // get the start index of the current queue in the global memory

    if (threadIdx.x == 0)
    {
        sharedStartIndex = atomicAdd(numberOfCurrentQueue, sharedNumCurrentFrontier);

        // copy the number of levels to the global memory
        *numberOfLevels = sharedNumLevels;
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < sharedNumCurrentFrontier; i += blockDim.x) // memory coalescing achieved
    {
        currentQueue[sharedStartIndex + i] = sharedCurrentQueue[i];
    }
}

// The kernel is only called when the number of vertices of two consecutive queues combined is less than the number of threads in a single block
// This is to minimize the overhead of the kernel launch
// This will create a local level array for the block and synchronize at the end
__global__ void minimize_overhead_kernel(unsigned int *srcPtrs, unsigned int *dst, unsigned int *level, unsigned int *currentQueue,
                                         unsigned int *previousQueue, int currentLevel, unsigned int *numberOfCurrentQueue, int numberOfPreviousQueue,
                                         int vertices, int edges, unsigned int *numberOfLevels)
{
    __shared__ unsigned int sharedPreviuosQueue[LOCAL_QUEUE_SIZE];
    __shared__ unsigned int sharedCurrentQueue[LOCAL_QUEUE_SIZE];
    __shared__ unsigned int sharedNumPreviousQueue;
    __shared__ unsigned int sharedNumCurrentQueue;
    __shared__ unsigned int sharedNumLevels;

    // Since we have a single block the vertex index in the queue is the thread index
    int threadIndex = threadIdx.x;

    // only need a single thread to initialize the shared memory not all but make sure all threads are synchronized
    if (threadIndex == 0)
    {
        sharedNumPreviousQueue = numberOfPreviousQueue;
        sharedNumCurrentQueue = 0;
        sharedNumLevels = 0;
    }
    __syncthreads();

    // Copy the previous queue to the shared memory
    for (int i = threadIndex; i < sharedNumPreviousQueue; i += blockDim.x)
    {
        sharedPreviuosQueue[i] = previousQueue[i];
    }
    __syncthreads();

    // While the number of vertices in the previous queue is greater than 0 and the number of vertices in the current queue is less than the block size
    // If the number of vertices in the current queue is greates than the block size then we will add the vertices to the global queue


    while (sharedNumPreviousQueue > 0 && sharedNumCurrentQueue <= blockDim.x)
    {
        // Reset the number of vertices in the current queue
        if (threadIndex == 0)
        {
            sharedNumCurrentQueue = 0;
        }
        __syncthreads();

        // ... 

        // Process the vertices in the queue
        for (unsigned int i = threadIndex; i < sharedNumPreviousQueue; i += blockDim.x)
        {
            unsigned int vertex = sharedPreviuosQueue[i];
            unsigned int start = srcPtrs[vertex];
            unsigned int end = srcPtrs[vertex + 1];

            for (unsigned int j = start; j < end; j++)
            {
                unsigned int neighbour = dst[j];
                if (atomicCAS(&level[neighbour], UINT_MAX, currentLevel + sharedNumLevels) == UINT_MAX)
                {
                    unsigned int sharedIndex = atomicAdd(&sharedNumCurrentQueue, 1);
                    if (sharedIndex < LOCAL_QUEUE_SIZE)
                    {
                        sharedCurrentQueue[sharedIndex] = neighbour;
                    }
                    else
                    {
                        sharedNumCurrentQueue = LOCAL_QUEUE_SIZE;
                        unsigned int index = atomicAdd(numberOfCurrentQueue, 1);
                        currentQueue[index] = neighbour;
                    }
                }
            }
        }
        __syncthreads();

        if (threadIndex == 0)
        {
            // Copy the current queue to the previous queue
            // The previous queue becomes the current queue for the next iteration
            sharedNumPreviousQueue = sharedNumCurrentQueue; 

            // Increment the number of levels to indicate a level has been processed
            atomicAdd(&sharedNumLevels, 1);
        }
        __syncthreads();

        // Each thread copies an element from the current queue to the previous queue
        for (int i = threadIndex; i < sharedNumCurrentQueue; i += blockDim.x)
        {
            sharedPreviuosQueue[i] = sharedCurrentQueue[i];
        }
        __syncthreads();


        // if (sharedNumPreviousQueue > LOCAL_QUEUE_SIZE)
        //     break;
    }

    // copy the shared memory to the global memory
    __shared__ unsigned int sharedStartIndex; // get the start index of the current queue in the global memory

    if (threadIdx.x == 0)
    {
        sharedStartIndex = atomicAdd(numberOfCurrentQueue, sharedNumCurrentQueue);

        // copy the number of levels to the global memory
        *numberOfLevels = sharedNumLevels;
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < sharedNumCurrentQueue; i += blockDim.x) // memory coalescing achieved
    {
        currentQueue[sharedStartIndex + i] = sharedCurrentQueue[i];
    }
}

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
    FILE *cpuOutputFile = fopen("output/cpu_output.txt", "w");
    for (int i = 0; i < vertices; i++)
    {
        fprintf(cpuOutputFile, "%d %d\n", i, levelCPU[i]);
    }

    double cpuTime = timer.elapsed();
    printf("\033[0;33m"); // set color to yellow
    printf("CPU Time: %f ms\n", cpuTime);
    printf("\033[0m"); // reset color

    // 2. Allocate device memory for the graph
    timer.start();
    unsigned int *deviceSrc;
    unsigned int *deviceDst;
    unsigned int *deviceLevel;
    unsigned int *deviceTemp1;           // used to swap the current and previous queues
    unsigned int *deviceTemp2;           // used to swap the current and previous queues
    unsigned int *deviceNumCurrentQueue; // the number of vertices in the queue of the current level
    unsigned int *deviceNumLevels;       // the number of levels updated in the kernel

    cudaMalloc((void **)&deviceSrc, (vertices + 1) * sizeof(unsigned int));
    cudaMalloc((void **)&deviceDst, edges * sizeof(unsigned int));
    cudaMalloc((void **)&deviceLevel, vertices * sizeof(unsigned int));
    cudaMalloc((void **)&deviceTemp1, vertices * sizeof(unsigned int));
    cudaMalloc((void **)&deviceTemp2, vertices * sizeof(unsigned int));
    cudaMalloc((void **)&deviceNumCurrentQueue, sizeof(unsigned int));
    cudaMalloc((void **)&deviceNumLevels, sizeof(unsigned int));
    cudaDeviceSynchronize();

    unsigned int *devicePreviosQueue = deviceTemp1;
    unsigned int *deviceCurrentQueue = deviceTemp2;

    // printf("\nDevice memory allocated successfully\n");
    timer.stop();
    double allocationTime = timer.elapsed();
    printf("Time taken to allocate device memory: %f ms\n", allocationTime);

    // 3. Copy memory to the device
    timer.start();
    cudaMemcpy(deviceSrc, srcPtrs, (vertices + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDst, dst, edges * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLevel, level, vertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceCurrentQueue, &srcVertex, sizeof(unsigned int), cudaMemcpyHostToDevice); // the initial queue will have the source vertex only
    cudaDeviceSynchronize();
    timer.stop();

    // printf("Copied to GPU successfully\n");
    double copyingTime = timer.elapsed();

    printf("Time taken to copy memory to the device: %f ms\n", copyingTime);

    // 4. Set the number of threads and blocks
    int threadsPerBlock = 256;
    int blocksPerGrid;
    int numberOfPreviousQueue = 1;

    int currentLevel = 1; // we start from level 1 since we already set the level of the source vertex to 0
    int numberOfLevels = 0;

    // 5. Launch the kernel
    timer.start();

    // Here the stopping condition changed to check the number of vertices in the previous queue
    // If the number of vertices in the previous queue is 0 then no thread has added any vertices to the current queue
    // This means that the max depth has been reached
    while (numberOfPreviousQueue > 0)
    {
        cudaMemset(deviceNumCurrentQueue, 0, sizeof(unsigned int));                      // reset the number of vertices in the current queue


        cudaMemset(deviceNumLevels, 0, sizeof(unsigned int));  // reset the number of vertices in the current queue


        blocksPerGrid = (numberOfPreviousQueue + threadsPerBlock - 1) / threadsPerBlock; // calculate the number of blocks needed for the current queue

        if (numberOfPreviousQueue <= threadsPerBlock)
        {
            minimize_overhead_kernel<<<1, threadsPerBlock>>>(deviceSrc, deviceDst, deviceLevel, deviceCurrentQueue, devicePreviosQueue,
                                                             currentLevel, deviceNumCurrentQueue, numberOfPreviousQueue, vertices, edges, deviceNumLevels);
        }
        else
        {
            vertex_centric_optimized_bfs<<<blocksPerGrid, threadsPerBlock>>>(deviceSrc, deviceDst, deviceLevel, deviceCurrentQueue, devicePreviosQueue,
                                                                             currentLevel, deviceNumCurrentQueue, numberOfPreviousQueue, vertices, edges, deviceNumLevels);
        }

        // Copy the number of the current queue of the device to the number of previous queue of the host
        cudaMemcpy(&numberOfPreviousQueue, deviceNumCurrentQueue, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        // Swap the current and previous queues
        unsigned int *temp = devicePreviosQueue;
        devicePreviosQueue = deviceCurrentQueue;
        deviceCurrentQueue = temp;

        // Copy the number of levels to the host
        cudaMemcpy(&numberOfLevels, deviceNumLevels, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        currentLevel += numberOfLevels;

    }

    cudaDeviceSynchronize(); // wait for all kernels to finish so that the level array is updated
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

    // 6. Copy the result back to the host
    cudaMemcpy(level, deviceLevel, vertices * sizeof(int), cudaMemcpyDeviceToHost);

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
    free(levelCPU);

    // Free device memory
    timer.start();
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
    cudaFree(deviceLevel);
    cudaFree(devicePreviosQueue);
    cudaFree(deviceCurrentQueue);
    cudaFree(deviceNumCurrentQueue);
    cudaFree(deviceNumLevels);
    timer.stop();

    double deallocationTime = timer.elapsed();

    printf("Time taken to deallocate device memory: %f ms\n", deallocationTime);

    double totalGPUTime = allocationTime + copyingTime + kernelTime + deallocationTime;

    printf("\033[0;32m"); // set color to green
    printf("Total GPU time: %f ms\n", totalGPUTime);
    printf("\033[0m"); // reset color

    return 0;
}