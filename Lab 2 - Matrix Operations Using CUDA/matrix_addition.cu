// Matrix addition using CUDA

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

#define MAX_ERR 1e-6

// A.   kernel1: each thread produces one output matrix element
__global__ void kernel1(float *out, float *a, float *b, int rows, int columns)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("i = %d, row = %d\n", i, row);

    if (col < columns && row < rows)
    {
        out[col + columns * row] = a[col + columns * row] + b[col + columns * row];
    }
}

// B.   kernel2: each thread produces one output matrix row
__global__ void kernel2(float *out, float *a, float *b, int rows, int columns)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < rows)
    {
        for (int i = 0; i < columns; i++)
        {
            int index = j * columns + i;
            out[index] = a[index] + b[index];
        }
    }
}

// C.   kernel3: each thread produces one output matrix column
__global__ void kernel3(float *out, float *a, float *b, int rows, int columns)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < columns)
    {
        for (int j = 0; j < columns; j++)
        {
            int index = j * columns + i;
            out[index] = a[index] + b[index];
        }
    }
}

int main(char argc, char *argv[])
{
    FILE *fp;

    if (argc < 3)
    {
        printf("Please provide the number of rows and columns of the matrix\n");
        return 1;
    }

    fp = fopen(argv[1], "r");

        int rows = atoi(argv[1]);
    int columns = atoi(argv[2]);
    int N = rows * columns;

    if (argc < 3 + N)
    {
        printf("Please provide the matrix elements\n");
        return 1;
    }

    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // 1. Allocate host memory
    a = (float *)malloc(sizeof(float) * N);
    b = (float *)malloc(sizeof(float) * N);
    out = (float *)malloc(sizeof(float) * N);

    // 2. Initialize host memory
    for (int i = 0; i < N; i++)
    {
        a[i] = atof(argv[i + 3]);
        b[i] = atof(argv[i + 3 + N]);
    }

    for (int i = 0; i < N; i++)
    {
        printf("%f ", a[i]);
        if ((i + 1) % columns == 0)
        {
            printf("\n");
        }
    }

    printf("\n");

    for (int i = 0; i < N; i++)
    {
        printf("%f ", b[i]);
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

    dim3 blockDim(16, 16);
    dim3 threadsPerBlock(16, 16);
    // int blockSize = 256;
    dim3 gridDim((columns + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // 5. Executing kernel
    kernel3<<<gridDim, blockDim>>>(d_out, d_a, d_b, rows, columns);

    // 6. Transfer data from device to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // 7. Verification
    for (int i = 0; i < N; i++)
    {
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    for (int i = 0; i < N; i++)
    {
        printf("%f ", out[i]);
        if ((i + 1) % columns == 0)
        {
            printf("\n");
        }
    }
    printf("\nout[0] = %f\n", out[0]);
    printf("PASSED\n");

    // 8. Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // 9. Deallocate host memory
    free(a);
    free(b);
    free(out);

    return 0;
}