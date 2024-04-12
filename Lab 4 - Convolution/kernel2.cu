#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include "./stb/stb_image_write.h"
#include "read_data.h"

__constant__ float constant_mask[100]; // constant memory for the mask

__global__ void kernel2(unsigned char *output_image, unsigned char *input_image, int width, int height, int comp, int mask_size)
{
}

int main(char argc, char *argv[])
{

    return 0;
}