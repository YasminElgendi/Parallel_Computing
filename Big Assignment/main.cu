// // This file is for testing purposes
// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include <assert.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <fstream>
// #include <vector>
// #include <queue>
// #include "graph.h"

// int main(int argc, char *argv[])
// {
//     // Check for correct number of arguments
//     if (argc < 2)
//     {
//         printf("Usage: %s <filename>\n", argv[0]);
//         exit(1);
//     }

//     // read the filename from the command line
//     char *filename = argv[1];

//     printf("COO Graph Representation\n");
//     // COO graph representation
//     FILE *file = fopen(filename, "r"); // open graph file
//     // read the number of vertices and edges
//     int vertices, edges;
//     fscanf(file, "%d %d", &vertices, &edges);

//     // allocate host memory for the graph
//     int *src = (int *)malloc(edges * sizeof(int));
//     int *dst = (int *)malloc(edges * sizeof(int));

//     COOGraph(file, src, dst, edges); // construct the graph using the COO representation

//     printGraph(src, dst, edges, vertices, 0); // print the graph

//     // close files
//     fclose(file);

//     // CSR graph representation
//     printf("\n\nCSR Graph Representation\n");

//     file = fopen(filename, "r"); // open graph file

//     // read vertices and edges
//     fscanf(file, "%d %d", &vertices, &edges);

//     int *srcPtrs = (int *)malloc(vertices * sizeof(int));
//     int *dstCSR = (int *)malloc(edges * sizeof(int));
//     CSRGraph(file, srcPtrs, dstCSR, edges); // construct the graph using the CSR representation

//     printGraph(srcPtrs, dstCSR, edges, vertices, 1); // print the graph

//     // close files
//     fclose(file);

//     // free memory
//     free(src);
//     free(dst);
//     free(srcPtrs);
//     free(dstCSR);

//     return 0;
// }
