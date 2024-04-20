// This file is a part of the OpenSurgSim project.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_ERR 1e-6

__global__ void matrix_multiplication(float *out, float *matrix, float *vector, int rows, int columns)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        for (int i = 0; i < columns; i++)
        {
            out[row] += vector[i] * matrix[(row * columns) + i];
        }
    }
}

void read_matrices(FILE *input, float *a, float *b, int &testcases, int rows, int columns)
{
    char line[200];
    int check;

    // Read first matrix
    for (int i = 0; i < rows * columns; i++)
    {
        if ((i + 1) % columns == 0)
        {
            fscanf(input, "%f", &a[i]);

            check = getc(input);
            if (check == '\n')
            {
                continue;
            }
            fgets(line, 100, input);
            continue;
        }
        fscanf(input, "%f", &a[i]);
    }

    // Read second matrix
    for (int i = 0; i < columns; i++)
    {
        fscanf(input, "%f", &b[i]);
        getc(input);
    }
}

void write_file(FILE *output, float *out, int rows, int columns)
{
    for (int i = 0; i < rows; i++)
    {
        fprintf(output, "%0.1f ", out[i]);
        fprintf(output, "\n");
    }
    fprintf(output, "\n");
}

int main(char argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Please provide the paths of the input and output files\n");
        return 1;
    }

    FILE *inputFile;
    FILE *outputFile;

    inputFile = fopen(argv[1], "r");
    outputFile = fopen(argv[2], "w");

    if (!inputFile || !outputFile)
    {
        printf("Please provide the correct path of both files");
        return 1;
    }

    // Read the input file

    // Get the number of testcases
    int testcasesNumber, rows, columns;
    testcasesNumber = getc(inputFile) - '0';

    printf("Number of testcases: %d\n", testcasesNumber);

    // Skip the comment
    char line[200];
    fgets(line, 200, inputFile);

    for (int i = 0; i < testcasesNumber; i++)
    {
        printf("\nTestcase: %d\n", i + 1);

        float *matrix, *vector, *out;
        float *d_matrix, *d_vector, *d_out;

        fscanf(inputFile, "%d %d", &rows, &columns);
        fgets(line, 200, inputFile);

        printf("Rows: %d, Columns: %d\n\n", rows, columns);

        // 1. Allocate host memory
        matrix = (float *)malloc(sizeof(float) * rows * columns);
        vector = (float *)malloc(sizeof(float) * columns);
        out = (float *)malloc(sizeof(float) * rows);

        // 2. Initialize host memory
        read_matrices(inputFile, matrix, vector, testcasesNumber, rows, columns);

        printf("Matrix:\n");
        for (int i = 0; i < rows * columns; i++)
        {
            printf("%0.1f ", matrix[i]);
            if ((i + 1) % columns == 0)
            {
                printf("\n");
            }
        }

        printf("\nVector:\n");
        for (int i = 0; i < columns; i++)
        {
            printf("%0.1f ", vector[i]);
            printf("\n");
        }

        printf("\n");

        // 3. Allocate device memory
        cudaMalloc((void **)&d_matrix, sizeof(float) * rows * columns);
        cudaMalloc((void **)&d_vector, sizeof(float) * columns);
        cudaMalloc((void **)&d_out, sizeof(float) * rows);

        // 4. Transfer data from host to device memory
        cudaMemcpy(d_matrix, matrix, sizeof(float) * rows * columns, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vector, vector, sizeof(float) * columns, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int gridSize = (columns + blockSize) / blockSize;
        dim3 threadsPerBlock(16, 16);

        // 5. Executing kernel
        matrix_multiplication<<<gridSize, blockSize>>>(d_out, d_matrix, d_vector, rows, columns);

        // 6. Transfer data from device to host memory
        cudaMemcpy(out, d_out, sizeof(float) * rows, cudaMemcpyDeviceToHost);

        printf("Output Vector:\n");
        for (int i = 0; i < rows; i++)
        {
            printf("%0.1f ", out[i]);
            printf("\n");
        }

        write_file(outputFile, out, rows, columns);

        // 7. Deallocate device memory
        cudaFree(d_matrix);
        cudaFree(d_vector);
        cudaFree(d_out);

        // 8. Deallocate host memory
        free(matrix);
        free(vector);
        free(out);
    }

    fclose(inputFile);
    fclose(outputFile);

    return 0;
}