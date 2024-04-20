// C.   kernel3: each thread produces one output matrix column

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

#define MAX_ERR 1e-6

__global__ void matrix_addition(float *out, float *a, float *b, int rows, int columns)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < columns)
    {
        for (int j = 0; j < rows; j++)
        {
            int index = col * rows + j;
            out[index] = a[index] + b[index];
        }
    }
}

void write_file(FILE *output, float *matrix, int rows, int columns)
{
    for (int i = 0; i < rows * columns; i++)
    {
        fprintf(output, "%0.1f ", matrix[i]);
        if ((i + 1) % columns == 0)
        {
            fprintf(output, "\n");
        }
    }
}

void read_matrices(FILE *input, float *a, float *b, int &testcases, int rows, int columns)
{
    char line[100];
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
    for (int i = 0; i < rows * columns; i++)
    {
        if ((i + 1) % columns == 0)
        {
            fscanf(input, "%f", &b[i]);
            check = getc(input);
            continue;
        }
        fscanf(input, "%f", &b[i]);
    }
}

int main(char argc, char *argv[])
{

    if (argc < 3)
    {
        printf(argv[1]);
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

    // Skip the comment
    char line[100];
    fgets(line, 100, inputFile);

    for (int i = 0; i < testcasesNumber; i++)
    {
        // Read number of rows and columns
        fscanf(inputFile, "%d %d", &rows, &columns);
        int N = rows * columns;

        // Skip comments
        fgets(line, 100, inputFile);

        float *a, *b, *out;
        float *d_a, *d_b, *d_out;

        // 1. Allocate host memory
        a = (float *)malloc(sizeof(float) * N);
        b = (float *)malloc(sizeof(float) * N);
        out = (float *)malloc(sizeof(float) * N);

        // 2. Initialize host memory
        read_matrices(inputFile, a, b, testcasesNumber, rows, columns);

        for (int i = 0; i < N; i++)
        {
            printf("%0.1f ", a[i]);
            if ((i + 1) % columns == 0)
            {
                printf("\n");
            }
        }

        printf("\n");

        for (int i = 0; i < N; i++)
        {
            printf("%0.1f ", b[i]);
            if ((i + 1) % columns == 0)
            {
                printf("\n");
            }
        }

        printf("\n");

        // 3. Allocate device memory
        cudaMalloc((void **)&d_a, sizeof(float) * N);
        cudaMalloc((void **)&d_b, sizeof(float) * N);
        cudaMalloc((void **)&d_out, sizeof(float) * N);

        // 4. Transfer data from host to device memory
        cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int gridSize = (columns + blockSize) / blockSize;
        printf("gridSize: %d\n", gridSize); 
        dim3 threadsPerBlock(16, 16);

        // 5. Executing kernel
        matrix_addition<<<gridSize, blockSize>>>(d_out, d_a, d_b, rows, columns);

        // 6. Transfer data from device to host memory
        cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

        // 7. Verification
        for (int i = 0; i < N; i++)
        {
            assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
        }

        for (int i = 0; i < N; i++)
        {
            printf("%0.1f ", out[i]);
            if ((i + 1) % columns == 0)
            {
                printf("\n");
            }
        }

        write_file(outputFile, out, rows, columns);
        printf("\nPASSED\n\n");

        // 8. Deallocate device memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_out);

        // 9. Deallocate host memory
        free(a);
        free(b);
        free(out);
    }

    fclose(inputFile);
    fclose(outputFile);

    return 0;
}