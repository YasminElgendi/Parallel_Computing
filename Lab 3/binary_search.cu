#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

__global__ void kernel(float *d_array, float target_value, int size)
{
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
        printf("Please provide the path of the input file and the target element\n");
        return 1;
    }

    FILE *inputFile;
    float target_value = atof(argv[2]);

    inputFile = fopen(argv[1], "r");

    if (inputFile == NULL)
    {
        printf("Error opening file\n");
        return 1;
    }

    int size = getSize(inputFile);

    printf("Size: %d\n", size);

    float *array, *result;

    float *d_array, *d_result;

    array = (float *)malloc(size * sizeof(float));
    result = (float *)malloc(2 * sizeof(float));

    readInput(inputFile, array, size);

    for (int i = 0; i < size; i++)
    {
        printf("%0.1f\n", array[i]);
    }

    // Allocate device memory
    cudaMalloc((void **)&d_array, sizeof(float) * size);
    cudaMalloc((void **)&d_result, sizeof(float) * 2);

    // 8. Deallocate device memory
    cudaFree(d_array);
    cudaFree(d_result);

    // 9. Deallocate host memory
    free(array);
    free(result);

    fclose(inputFile);

    return 0;
}
