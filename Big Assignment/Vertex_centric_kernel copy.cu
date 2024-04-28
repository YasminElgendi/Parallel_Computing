#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

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

    return 0;
}