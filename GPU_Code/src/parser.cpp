#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <filesystem> 
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <sys/time.h>
#include "parser.h"
#include "time_and_path_management.h"

HybridMatrix parser_hybrid(const std::filesystem::path& filepath, int K, double* parsing_time) {
    TIMER_DEF(parse_timer);
    TIMER_START(parse_timer); 

    HybridMatrix result = {};
    
    FILE* f = fopen(filepath.string().c_str(), "r");
    if (!f) return result;

    char line[1024];
    bool is_symmetric = false;

    // Parse Header
    if (fgets(line, sizeof(line), f)) {
        if (strstr(line, "symmetric")) is_symmetric = true;
    }

    // Skip Comments
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '%') break;
    }

    // Read Dimensions
    int rows, cols;
    size_t file_nnz;
    sscanf(line, "%d %d %zu", &rows, &cols, &file_nnz);

    // Temporary Allocation for sorting
    size_t max_possible = is_symmetric ? (file_nnz * 2) : file_nnz;
    MatrixEntry* entries = (MatrixEntry*)malloc(max_possible * sizeof(MatrixEntry));

    // Data Parsing
    size_t count = 0;
    for (size_t i = 0; i < file_nnz; ++i) {
        int r, c;
        float v;
        if (fscanf(f, "%d %d %f", &r, &c, &v) != 3) break;
        
        r--; c--; // Convert to 0-based index
        entries[count++] = {r, c, v};

        if (is_symmetric && r != c) {
            entries[count++] = {c, r, v};
        }
    }
    fclose(f);

    // Sort entries by row, then by column
    std::sort(entries, entries + count, [](const MatrixEntry& a, const MatrixEntry& b) {
        if (a.r != b.r) return a.r < b.r;
        return a.c < b.c;
    });

    // Setup base dimensions
    result.ell.n_rows = rows;
    result.ell.n_cols = cols;
    result.ell.K = K;
    
    result.csr.n_rows = rows;
    result.csr.n_cols = cols;

    // --- PASS 1: Count NNZ distributions per row ---
    std::vector<int> ell_row_counts(rows, 0);
    std::vector<int> csr_row_counts(rows, 0);
    size_t total_csr_nnz = 0;

    size_t current_entry_idx = 0;
    for (int r = 0; r < rows; ++r) {
        int row_nnz_count = 0;
        // Keep counting elements matching the current row index
        while (current_entry_idx < count && entries[current_entry_idx].r == r) {
            if (row_nnz_count < K) {
                ell_row_counts[r]++;
            } else {
                csr_row_counts[r]++;
                total_csr_nnz++;
            }
            row_nnz_count++;
            current_entry_idx++;
        }
    }

    // --- Allocate Memory ---
    // ELLPACK is a fixed dense layout: rows * K
    size_t total_ell_elements = (size_t)rows * K;
    result.ell.col_indices = (int*)malloc(total_ell_elements * sizeof(int));
    result.ell.values = (float*)malloc(total_ell_elements * sizeof(float));

    // Fill ELLPACK arrays with default padding values (-1 index, 0.0f value)
    std::fill_with_padding_values(result.ell.col_indices, total_ell_elements, -1);
    for (size_t i = 0; i < total_ell_elements; ++i) {
        result.ell.col_indices[i] = -1;
        result.ell.values[i] = 0.0f;
    }

    // CSR allocation
    result.csr.nnz = total_csr_nnz;
    result.csr.row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    result.csr.col_indices = total_csr_nnz > 0 ? (int*)malloc(total_csr_nnz * sizeof(int)) : nullptr;
    result.csr.values = total_csr_nnz > 0 ? (float*)malloc(total_csr_nnz * sizeof(float)) : nullptr;

    // Build CSR row pointers
    result.csr.row_ptr[0] = 0;
    for (int r = 0; r < rows; ++r) {
        result.csr.row_ptr[r + 1] = result.csr.row_ptr[r] + csr_row_counts[r];
    }

    // --- PASS 2: Populate Structures ---
    current_entry_idx = 0;
    size_t csr_inserted = 0;

    for (int r = 0; r < rows; ++r) {
        int ell_idx_in_row = 0;
        
        while (current_entry_idx < count && entries[current_entry_idx].r == r) {
            MatrixEntry e = entries[current_entry_idx];
            
            if (ell_idx_in_row < K) {
                // ELLPACK storage: Row-Major indexing layout [r * K + ell_idx_in_row]
                size_t target_idx = (size_t)r * K + ell_idx_in_row;
                result.ell.col_indices[target_idx] = e.c;
                result.ell.values[target_idx] = e.v;
                ell_idx_in_row++;
            } else {
                // CSR Spillover
                result.csr.col_indices[csr_inserted] = e.c;
                result.csr.values[csr_inserted] = e.v;
                csr_inserted++;
            }
            current_entry_idx++;
        }
    }

    free(entries); // Free sorting array
    TIMER_STOP(parse_timer);
    *parsing_time = TIMER_ELAPSED(parse_timer);
    
    return result;
}