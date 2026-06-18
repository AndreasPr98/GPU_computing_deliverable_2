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


// The parser function
COO parser(const std::string& file_path, double* parsing_time) {

    TIMER_DEF(parse_timer);
    TIMER_START(parse_timer); 

    FILE* f = fopen(file_path.c_str(), "r");
    if (!f) return {0, 0, 0, nullptr, nullptr, nullptr};

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

    // Allocation (Array of Structs for sorting)
    size_t max_possible = is_symmetric ? (file_nnz * 2) : file_nnz;
    MatrixEntry* entries = (MatrixEntry*)malloc(max_possible * sizeof(MatrixEntry));

    // Data Parsing
    size_t count = 0;
    for (size_t i = 0; i < file_nnz; ++i) {
        int r, c;
        float v;

        if (fscanf(f, "%d %d %f", &r, &c, &v) != 3) break;
        
        r--; c--; // 0-based

        entries[count++] = {r, c, v};

        if (is_symmetric && r != c) {
            entries[count++] = {c, r, v};
        }
    }
    fclose(f);

    // Sorting after the "unsymmetrifying" of the matrix
    std::sort(entries, entries + count, [](const MatrixEntry& a, const MatrixEntry& b) {
        if (a.r != b.r) return a.r < b.r;
        return a.c < b.c;
    });

    // Final Allocation and Linear Copy
    COO mtx;
    mtx.n_rows = rows;
    mtx.n_cols = cols;
    mtx.nnz = count;
    mtx.row_indices = (int*)malloc(count * sizeof(int));
    mtx.col_indices = (int*)malloc(count * sizeof(int));
    mtx.values = (float*)malloc(count * sizeof(float));

    for (size_t i = 0; i < count; ++i) {
        mtx.row_indices[i] = entries[i].r;
        mtx.col_indices[i] = entries[i].c;
        mtx.values[i] = entries[i].v;
    }

    free(entries); // Free the temporary sorting array
    TIMER_STOP(parse_timer);
    *parsing_time = TIMER_ELAPSED(parse_timer);
    return mtx;
}

