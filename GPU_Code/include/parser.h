#ifndef PARSER_H
#define PARSER_H

#include <vector>
#include <filesystem>

struct CSR {
    int n_rows;
    int n_cols;
    int nnz; 
    int* row_ptr; // standard CSR row pointers (size: n_rows + 1)
    int* col_indices;
    float* values;
};

struct ELLPACK {
    int n_rows;
    int n_cols;
    int K;         // Width of the ELLPACK matrix storage per row
    int* col_indices; // Flattened 2D array of size (n_rows * K)
    float* values;    // Flattened 2D array of size (n_rows * K)
};

struct MatrixEntry { // Only used in the parser to organize data
    int r, c;
    float v;
};

struct HybridMatrix {
    CSR csr;
    ELLPACK ell;
};

// Updated signature returning both structs wrapped together
HybridMatrix parser_hybrid(const std::filesystem::path& filepath, int K, double* parsing_time);

#endif