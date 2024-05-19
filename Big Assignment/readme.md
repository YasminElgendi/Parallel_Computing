# Graph Parallel Traversing
## Approaches are Vertex-Centric and Edge-Centric for CPU and GPU with further optimizations

### 1. Generate graph using python script
#### Format: python generate.py num_vertices num_edges output_file.txt
    python generate.py 10 10 graph_10.txt

### 2. Change configurations in make file for whch algorithm to use and output file paths

### 3. Build files
    make

### 4. Run
    make run