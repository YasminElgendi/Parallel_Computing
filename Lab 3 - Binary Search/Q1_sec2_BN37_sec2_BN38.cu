#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

__global__ void kernel(float *array, float *sum, int N)
{

    __shared__ float partialSum[256];

    // Copy input elements to shared memory
    int index = threadIdx.x;
    int start = blockIdx.x * blockDim.x;

    if (start + index < N)
    {
        partialSum[index] = array[start + index];
    }

    __syncthreads();

    // Reduction
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (index % (2 * stride) == 0)
        {
            partialSum[index] += partialSum[index + stride];
        }
        __syncthreads();
    }

    // Write the result in global memory
    if (index == 0)
    {
        sum[blockIdx.x] = partialSum[0];
        // printf("kernel sum: %0.1f\n", *sum);
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

    // printf("Size: %d\n", size);

    float *array = (float *)malloc(size * sizeof(float));
    float *sum = (float *)malloc(sizeof(float));

    float *d_array;
    float *d_sum;

    readInput(inputFile, array, size);

    // for (int i = 0; i < size; i++)
    // {
    //     printf("%0.1f\n", array[i]);
    // }

    // Allocate device memory
    cudaMalloc((void **)&d_array, sizeof(float) * size);
    cudaMalloc((void **)&d_sum, sizeof(float));

    // Transfer data from host to device memory
    cudaMemcpy(d_array, array, sizeof(float) * size, cudaMemcpyHostToDevice);

    kernel<<<1, 256>>>(d_array, d_sum, size);

    // Transfer data back to host memory
    cudaMemcpy(sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // Result
    printf("%0.1f", *sum);
    
    // Deallocate device memory
    cudaFree(d_array);
    cudaFree(d_sum);

    // Deallocate host memory
    free(array);
    free(sum);

    fclose(inputFile);

    return 0;
}