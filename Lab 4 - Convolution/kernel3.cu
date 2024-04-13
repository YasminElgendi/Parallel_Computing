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

#define OUTPUT_TILE_SIZE 16

__constant__ float constant_mask[MAX_MASK_SIZE * MAX_MASK_SIZE]; // constant memory for the mask

// 3. kernel3: tiling where each block matches the output tile size.
// each block calculates an output tile => block size = output tile size
// to calculate the output tile we need (OUTPUT_TILE_SIZE + MASK_SIZE - 1) pixels from the input image => load into shared memory
// All threads will participate in calculating the output elements
// Some threads will load more than one pixel from the input image
__global__ void kernel3(unsigned char *output_image, unsigned char *input_image, int width, int height, int comp, int mask_size)
{
    // get the pixel index
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // STEPS:
    // 1. Load data into shared memory
    extern __shared__ int shared_input[]; // Example: Allocate 256 integers of shared memory

    // 2. Compute the output tile
    // 3. Write the output tile to the output image

    // check if the pixel is within the image boundaries

    // get the start of column and row for the current pixel being convulted

    // iterate over the three channels => for a single thread compute the pixel values for all three channels

    // iterate over the mask elements => surrounding box

    // check if the current pixel is within the image boundaries => no padding was added

    // write the pixel value to the output image
}

int main(char argc, char *argv[])
{
    // Read and check command line arguments
    // Command line arguments
    char *input_folder_path;
    char *output_folder_path;
    char *mask_file_path;

    int batch_size = readCommandLineArguments(argc, argv, &input_folder_path, &output_folder_path, &mask_file_path);

    // 1. Allocate host memory => read input image / convolution mask

    // 1.1 Read image
    int width, height, comp;
    unsigned char *input_image = readImage(strcat(input_folder_path, "/image.jpg"), &width, &height, &comp);
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
        printf("Error: failed to read mask\n");
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

    // 2. Allocate device memory
    unsigned char *d_image, *d_out;
    cudaMalloc((void **)&d_image, sizeof(unsigned char) * N);
    cudaMalloc((void **)&d_out, sizeof(unsigned char) * width * height);

    // 3. Transfer data from host to device memory
    cudaMemcpy(d_image, input_image, sizeof(unsigned char) * N, cudaMemcpyHostToDevice);

    // 4. Launch kernel
    dim3 block_dim(16, 16, 1);
    dim3 grid_dim(ceil(width / 16.0), ceil(height / 16.0), 1);

    kernel3<<<grid_dim, block_dim>>>(d_out, d_image, width, height, comp, mask_size);

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