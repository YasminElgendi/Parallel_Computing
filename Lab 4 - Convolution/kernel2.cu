#define STB_IMAGE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include "./include/stb/stb_image_write.h"
#include "read_data.h"

#define OUTPUT_TILE_WIDTH 1 // => 16 x 16 = 256

__constant__ float constant_mask[MAX_MASK_SIZE * MAX_MASK_SIZE]; // constant memory for the mask

// 2. kernel2: tiling where each block matches the input tile size.
// The size of the block matches the size of the input tile
__global__ void kernel2(unsigned char *output_image, unsigned char *input_image, int width, int height, int comp, int mask_size, int input_tile_width)
{
    // get the pixel index
    int out_column = blockIdx.x * OUTPUT_TILE_WIDTH + threadIdx.x;
    int out_row = blockIdx.y * OUTPUT_TILE_WIDTH + threadIdx.y;

    // printf("out_column = %d, out_row = %d\n", out_column, out_row);

    // indices to access the input image
    int in_column = out_column - mask_size / 2;
    int in_row = out_row - mask_size / 2;

    // printf("in_column = %d, in_row = %d\n", in_column, in_row);

    // STEPS:
    // 1. Load data into shared memory
    // each thread will load 3 values corresponding to the three channels
    extern __shared__ float shared_input[]; // Example: Allocate the input tile size of shared memory

    if (in_column >= 0 && in_column < width && in_row >= 0 && in_row < height)
    {
        for (int c = 0; c < comp; c++)
        {

            shared_input[(threadIdx.y * input_tile_width + threadIdx.x) * comp + c] = (float)input_image[(in_row * width + in_column) * comp + c];
        }
    }
    else
    {   
        for (int c = 0; c < comp; c++)
        {
            shared_input[(threadIdx.y * input_tile_width + threadIdx.x) * comp + c] = 0.0f;
        }
    }

    __syncthreads();

    // 2. Compute the output tile

    // check if the pixel is within the image boundaries
    if (threadIdx.x < OUTPUT_TILE_WIDTH && threadIdx.y < OUTPUT_TILE_WIDTH) // since the output tile = 1 only thread 0 will compute the pixel value
    {
        float pixel_value = 0.0f;
        for (int c = 0; c < comp; c++)
        {
            // iterate over the mask elements => surrounding box
            for (int mask_row = 0; mask_row < mask_size; mask_row++) // rows
            {
                for (int mask_column = 0; mask_column < mask_size; mask_column++) // columns
                {
                    pixel_value += shared_input[((threadIdx.y + mask_row) * input_tile_width + (threadIdx.x + mask_column)) * comp + c] * constant_mask[mask_row * mask_size + mask_column];
                }
            }
        }

        // 3. Write the output tile to the output image
        output_image[out_row * width + out_column] = (unsigned char)pixel_value;
    }
}

int main(char argc, char *argv[])
{
    // Read and check command line arguments
    // Command line arguments
    char *input_folder_path;
    char *output_folder_path;
    char *mask_file_path;

    int batch_size = readCommandLineArguments(argc, argv, &input_folder_path, &output_folder_path, &mask_file_path);
    printf("mask_file_path = %s\n", mask_file_path);

    // 1. Allocate host memory => read input image / convolution mask

    // 1.1 Read image
    int width, height, comp;

    // get the fulle input path of the image
    char full_input_path[256];
    sprintf(full_input_path, "%s/%s", input_folder_path, "/image.jpg");

    unsigned char *input_image = readImage(full_input_path, &width, &height, &comp);

    if (!input_image)
    {
        printf("Error: failed to read image\n");
        exit(1);
    }
    printf("width = %d, height = %d, comp = %d (channels)\n", width, height, comp); // print image dimensions
    int N = width * height * comp;

    // Allocate memory for the output image
    unsigned char *output_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    // 1.2 Read convolution masks
    FILE *mask_file = fopen(mask_file_path, "r");
    if (!mask_file)
    {
        printf("Error: failed to open mask file\n");
        exit(1);
    }

    // Read mask from file => same mask applied on all channels

    // read the first mask for the first channel
    int mask_size = readMaskSize(mask_file);                              // read mask size
    float *mask = (float *)malloc(mask_size * mask_size * sizeof(float)); // allocate memory for the mask
    readMask(mask_file, mask, mask_size);                                 // read mask elements
    printMask(mask, mask_size);                                           // print mask elements

    // Copy Filter to constant memory
    cudaMemcpyToSymbol(constant_mask, mask, mask_size * mask_size * sizeof(float));

    const int input_tile_width = OUTPUT_TILE_WIDTH + mask_size - 1;
    printf("INPUT TILE WIDTH = %d\n", input_tile_width);
    printf("OUTPUT TILE WIDTH = %d\n", OUTPUT_TILE_WIDTH);

    // 2. Allocate device memory
    unsigned char *d_image, *d_out;
    cudaMalloc((void **)&d_image, sizeof(unsigned char) * N);
    cudaMalloc((void **)&d_out, sizeof(unsigned char) * width * height);

    // 3. Transfer data from host to device memory
    cudaMemcpy(d_image, input_image, sizeof(unsigned char) * N, cudaMemcpyHostToDevice);

    // 4. Launch kernel
    dim3 block_dim(input_tile_width, input_tile_width);
    // int grid_columns = (width - 1) / OUTPUT_TILE_WIDTH + 1;
    // int grid_rows = (height - 1) / OUTPUT_TILE_WIDTH + 1;
    int grid_columns = ceil((float)width / OUTPUT_TILE_WIDTH);
    int grid_rows = ceil((float)height / OUTPUT_TILE_WIDTH);
    dim3 grid_dim(grid_columns, grid_rows);

    int shared_memory_size = input_tile_width * input_tile_width * comp * sizeof(float);

    kernel2<<<grid_dim, block_dim, shared_memory_size>>>(d_out, d_image, width, height, comp, mask_size, input_tile_width);

    // If Error occurs in kernel execution show it using cudaDeviceSynchronize, cudaGetLastError
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        // in red
        printf("\033[1;31m");
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        // reset color
        printf("\033[0m");
    }

    // 5. Transfer data from device to host memory
    cudaMemcpy(output_image, d_out, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

    // 6. Write output image

    stbi_write_jpg("./output/output_image.jpg", width, height, 1, output_image, 100);
    printf("OUTPUT IMAGE\n");
    for (size_t i = 0; i < NUM_PIXELS_TO_PRINT * comp; i++)
    {
        printf("%d%s", output_image[i], ((i + 1) % comp) ? " " : "\n");
    }

    // close the files
    fclose(mask_file);

    // free host memory
    free(mask);
    free(input_image);
    free(output_image);

    // free device memory
    cudaFree(d_image);
    cudaFree(d_out);
    cudaFree(constant_mask);

    return 0;
}