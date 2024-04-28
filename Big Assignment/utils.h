#include <stdio.h>

//Read graph from file of format: n is number of vertices, m is number of edges
// n m
// u1 v1 w1
// u2 v2 w2
// ...
void readGraph(FILE * filename, int *n, int *m, int **edges, int **weights)
{
    fscanf(filename, "%d %d", n, m);
    *edges = (int *)malloc(*m * 2 * sizeof(int));
    *weights = (int *)malloc(*m * sizeof(int));
    for (int i = 0; i < *m; i++)
    {
        fscanf(filename, "%d %d %d", &(*edges)[2 * i], &(*edges)[2 * i + 1], &(*weights)[i]);
    }
}