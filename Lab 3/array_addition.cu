#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

// __global__ void kernel(float *array, float *sum, int n)
// {

//     int index = threadIdx.x;
//     int stride = blockDim.x;

//     // printf("stride: %d\n", stride);

//     float tempSum = 0;
//     for (int i = index; i < n; i+=stride)
//     {
//         printf("array[%d]: %0.1f\n", i, array[i]);
//         tempSum += array[i];
//     }
//     printf("tempSum: %0.1f\n", tempSum);
//     atomicAdd(sum, tempSum);
// }
__global__ void kernel(float *array, float *sum, int N)
{

    __shared__ float partialSum[1024];

    *sum = 1.0f;

    // Copy input elements to shared memory
    int index = threadIdx.x;
    int start = blockIdx.x * blockDim.x;

    if (start + index < N)
    {
        partialSum[index] = array[start + index];
    }
    else
    {
        partialSum[index] = 0.0f; // Pad for out-of-bounds threads
    }
    __syncthreads();

    // Reduction
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (index % (2 * stride) == 0 && index + stride < blockDim.x)
        {
            partialSum[index] += partialSum[index + stride];
        }
        __syncthreads();
    }

    // Write the result back to global memory
    if (index == 0)
    {
        printf("partialSum[0]: %0.1f\n", partialSum[0]);
        sum = &partialSum[0];
        printf("kernel sum: %0.1f\n", *sum);
    }
}

int getSize(FILE *input)
{
    int size = 1;
    char c;
    while ((c = fgetc(input)) != EOF)
    {
        if (c == '\n')
            size++;
    }
    rewind(input);
    return size;
}

void readInput(FILE *input, float *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        fscanf(input, "%f", &array[i]);
    }
}

int main(char argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Please provide the path of the input file\n");
        return 1;
    }

    FILE *inputFile;

    inputFile = fopen(argv[1], "r");

    if (inputFile == NULL)
    {
        printf("Error opening file\n");
        return 1;
    }

    int size = getSize(inputFile);

    printf("Size: %d\n", size);

    float *array = (float *)malloc(size * sizeof(float));
    float *sum = (float *)malloc(sizeof(float));

    float *d_array;
    float *d_sum;

    readInput(inputFile, array, size);

    for (int i = 0; i < size; i++)
    {
        printf("%0.1f\n", array[i]);
    }

    // Allocate device memory
    cudaMalloc((void **)&d_array, sizeof(float) * size);
    cudaMalloc((void **)&d_sum, sizeof(float));

    // Transfer data from host to device memory
    cudaMemcpy(d_array, array, sizeof(float) * size, cudaMemcpyHostToDevice);

    kernel<<<1, 256>>>(d_array, d_sum, size);

    // Transfer data back to host memory
    cudaMemcpy(sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum: %0.1f\n", sum);

    // 8. Deallocate device memory
    cudaFree(d_array);
    cudaFree(d_sum);

    // 9. Deallocate host memory
    free(array);
    free(sum);

    fclose(inputFile);

    return 0;
}