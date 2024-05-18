#include <vector>
#include <cstdio>
#include <cstdlib>

using namespace std;
// create the graph data structure
struct Graph
{
    int numberOfVertices;
    int numberOfEdges;
    std::vector<std::vector<int>> adjacencyList;
};

struct GraphCSR
{
    int *srcPtrs;
    int *dst;
    int edges;
    int vertices;
    /* data */
};

int getSourceSize(char *filename)
{
    printf("Reading file %s\n", filename);
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error: File not found\n");
        exit(1);
    }

    int currVertex;
    int srcIndex = 0;
    int prevVertex;
    int skip, vertices, edges;
    fscanf(file, "%d %d", &vertices, &edges);

    printf("Number of edges: %d\n", edges);
    for (int i = 0; i < edges; i++)
    {
        fscanf(file, "%d", &currVertex);
        if (i == 0)
        {
            srcIndex++;
        }
        else if (currVertex != prevVertex)
        {
            srcIndex++;
        }

        prevVertex = currVertex;
        fscanf(file, "%d", &skip);
    }

    printf("Number of vertices: %d\n", srcIndex);
    fclose(file);

    // Indicates the end of the last vertex
    return srcIndex;
}

void createGraph(FILE *file, Graph &graph, int &vertices, int &edges)
{
    fscanf(file, "%d %d", &vertices, &edges);
    graph.numberOfVertices = vertices;
    graph.numberOfEdges = edges;
    graph.adjacencyList.resize(vertices);

    for (int i = 0; i < edges; i++)
    {
        int u, v;
        fscanf(file, "%d %d", &u, &v);
        graph.adjacencyList[u].push_back(v);
        graph.adjacencyList[v].push_back(u);
    }
}

// Adjacency matrix representations for sparse graphs

// Coordinate
// Easy access to source and destination vertices of a given edge
void COOGraph(FILE *file, int *src, int *dst, int edges)
{
    // The size of the arrays are the number of edges
    for (int i = 0; i < edges; i++)
    {
        fscanf(file, "%d %d", &src[i], &dst[i]);
    }
}

// Compressed Sparse Row
// Easy access to outgoing edges of a certain vertex
// The rowPtr is the size of the rows in the adjacency matrix
// The rowPtr is the size of the vertices (source)
// The colIndices is the size of the edges
// vertex = row
void CSRGraph(FILE *file, int *srcPtrs, int *dst, int edges)
{
    int currVertex;
    int srcIndex = 0;
    int prevVertex;
    for (int i = 0; i < edges; i++)
    {
        fscanf(file, "%d", &currVertex);
        if (i == 0)
        {
            srcPtrs[srcIndex] = i;
            srcIndex++;
        }
        else if (currVertex != prevVertex)
        {
            srcPtrs[srcIndex] = i;
            srcIndex++;
        }

        prevVertex = currVertex;
        fscanf(file, "%d", &dst[i]);
    }

    // Indicates the end of the last vertex
    srcPtrs[srcIndex] = edges;
}

// Compressed Sparse Column
// Easy access to incoming edges of a certain vertex
// need the graph to be sorted by the destination vertex
void CSCGraph(FILE *file, int *src, int *dstPtrs, int edges)
{
    int currVertex;
    int dstIndex = 0;
    int prevVertex;
    for (int i = 0; i < edges; i++)
    {
        fscanf(file, "%d", &src[i]);

        fscanf(file, "%d", &currVertex);
        if (i == 0)
        {
            dstPtrs[dstIndex] = i;
            dstIndex++;
        }
        else if (currVertex != prevVertex)
        {
            dstPtrs[dstIndex] = i;
            dstIndex++;
        }

        prevVertex = currVertex;
    }
}

// type: 0 - COO, 1 - CSR, 2 - CSC
void printGraph(int *src, int *dst, int edges, int vertices, int type)
{
    switch (type)
    {
    case 0:
        /* code */
        printf("Source: ");
        for (int i = 0; i < edges; i++)
        {
            printf("%d ", src[i]);
        }

        printf("\nDestination: ");
        for (int i = 0; i < edges; i++)
        {
            printf("%d ", dst[i]);
        }
        break;
    case 1:
        /* code */
        printf("Source: ");

        for (int i = 0; i < vertices + 1; i++)
        {
            // printf("%d ", i);
            printf("%d ", src[i]);
        }

        printf("\nDestination: ");
        for (int i = 0; i < edges; i++)
        {
            printf("%d ", dst[i]);
        }
        break;
    case 2:
        /* code */
        printf("Source: ");
        for (int i = 0; i < edges; i++)
        {
            printf("%d ", src[i]);
        }

        printf("\nDestination: ");
        for (int i = 0; i < vertices; i++)
        {
            printf("%d ", dst[i]);
        }
        break;

    default:
        break;
    }
    printf("\n");
}
