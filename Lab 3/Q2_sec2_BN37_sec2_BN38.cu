#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

__global__ void binarySearch(float *array, float target, int *result, int size) 
{
    int index = threadIdx.x;    

    int start_index = 0;
    int end_index = size - 1;

    while (start_index <= end_index) {
        int mid = start_index + (end_index - start_index) / 2;

        if (array[mid] == target) {
            *result = mid;
            return;
        }
        else if (array[mid] < target) {
            start_index = mid + 1;
        }
        else {
            end_index = mid - 1;
        }
    }

    *result = -1; // Not found
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
    if (argc < 3)
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
    float target = atof(argv[2]);

    // printf("Size: %d\n", size);

    float *array;
    int index = -1;

    float *d_array;
    int *d_result;

    array = (float *)malloc(size * sizeof(float));

    readInput(inputFile, array, size);

    // for (int i = 0; i < size; i++)
    // {
    //     printf("%0.1f\n", array[i]);
    // }

    // Allocate device memory
    cudaMalloc((void **)&d_array, sizeof(float) * size);
    cudaMalloc((void **)&d_result, sizeof(int));

    // Transfer data from host to device memory
    cudaMemcpy(d_array, array, sizeof(float) * size, cudaMemcpyHostToDevice);

    thrust::device_ptr<float> dev_ptr(d_array);
    thrust::sort(dev_ptr, dev_ptr + size);
    // for (int i = 0; i < size; i++)
    // {
    //     printf("%0.1f\n", array[i]);
    // }

    binarySearch<<<1, 256>>>(d_array, target, d_result, size);

    // Transfer data back to host memory
    cudaMemcpy(&index, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Result
    printf("%d", index);
    // Deallocate device memory
    cudaFree(d_array);
    cudaFree(d_result);

    // Deallocate host memory
    free(array);

    fclose(inputFile);

    return 0;
}
