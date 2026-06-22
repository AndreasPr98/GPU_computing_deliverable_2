#include "utils.h"
#include "time_and_path_management.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <climits>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <sys/time.h>

std::vector<int> stream_row_lengths_from_mtx(const std::filesystem::path& filepath, matrix_data& m_data) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return {};
    }

    std::string line;
    bool is_symmetric = false;
    
    // 1. Parse banner line to find symmetry flag
    if (std::getline(file, line)) {
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
        if (line.find("symmetric") != std::string::npos) {
            is_symmetric = true;
        }
    }

    // Skip remaining comments
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') {
            continue; 
        }
        break; // Reached the geometry line
    }

    // 2. Read the Matrix Geometry
    std::stringstream geometry_stream(line);
    if (!(geometry_stream >> m_data.n_rows >> m_data.n_cols >> m_data.nnz)) {
        std::cerr << "Error: Invalid Matrix Market geometry line." << std::endl;
        return {};
    }

    // 3. Allocate row lengths vector
    std::vector<int> row_lengths(m_data.n_rows, 0);

    // 4. Stream and accumulate lengths
    int row, col;
    while (file >> row >> col) {
        row--; col--; // Convert to 0-based index
        
        if (row >= 0 && row < m_data.n_rows) {
            row_lengths[row]++;
        }
        
        // Handle Symmetric mirroring mapping
        if (is_symmetric && row != col) {
            if (col >= 0 && col < m_data.n_rows) {
                row_lengths[col]++;
            }
        }
        
        file.ignore(256, '\n'); 
    }

    return row_lengths;
}

int find_optimal_ellpack_k(int num_rows, const std::vector<int>& row_lengths) {
    // 1. Find the maximum row length to bound search space
    int max_row_len = 0;
    for (int len : row_lengths) {
        if (len > max_row_len) max_row_len = len;
    }

    // 2. Build a histogram of row lengths
    std::vector<int> histogram(max_row_len + 1, 0);
    for (int len : row_lengths) {
        histogram[len]++;
    }

    // 3. Track best configuration
    int best_K = 0;
    unsigned long long min_total_bytes = ULLONG_MAX;

    // Weights (in bytes)
    const int bytes_per_element = 8; // 4 bytes for float value + 4 bytes for int col_idx
    const int bytes_per_ptr     = 4; // 4 bytes for CSR row_ptr int

    // 4. Evaluate memory footprint for every choice of K
    for (int K = 0; K <= max_row_len; ++K) {
        
        unsigned long long total_ellpack_elements = 0;
        unsigned long long total_csr_elements = 0;

        // Calculate element distribution based on row length split profiles
        for (int len = 0; len <= max_row_len; ++len) {
            long long num_rows_with_this_len = histogram[len];
            if (num_rows_with_this_len == 0) continue;

            if (len <= K) {
                // Entire row fits into ELLPACK
                total_ellpack_elements += num_rows_with_this_len * len;
            } else {
                // Row is split: K elements to ELLPACK, remainder to CSR spillover
                total_ellpack_elements += num_rows_with_this_len * K;
                total_csr_elements += num_rows_with_this_len * (len - K);
            }
        }

        // Compute Footprints
        unsigned long long ellpack_bytes = total_ellpack_elements * bytes_per_element;
        
        // CSR includes overhead for the structural row pointer array (num_rows + 1)
        unsigned long long csr_bytes = (total_csr_elements * bytes_per_element) + 
                                       ((num_rows + 1) * bytes_per_ptr);

        unsigned long long total_bytes = ellpack_bytes + csr_bytes;

        // Capture configuration optimizing total memory overhead
        if (total_bytes < min_total_bytes) {
            min_total_bytes = total_bytes;
            best_K = K;
        }
    }

    return best_K;
}

void get_local_chunk(long int n_points, int* local_size, 
                     int* local_start) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int base_size = n_points / world_size;
    int remainder = n_points % world_size;

    // 1. Calculate local size
    if (world_rank < remainder) {
        *local_size = base_size + 1;
    } else {
        *local_size = base_size;
    }

    // 2. Calculate local start
    if (world_rank < remainder) {
        *local_start = world_rank * (*local_size);
    } else {
        *local_start = world_rank * base_size + remainder;
    }
}

double arithmetic_mean(const double *v, int len) {

    double mu = 0.0;
    for (int i=0; i<len; i++)
        mu += (double)v[i];
    mu /= (double)len;

    return(mu);
}

double geometric_mean(const double *v, int len) {
    
    double mu = 1.0;
    for (int i=0; i<len; i++) {
        mu *= (v[i] > 0) ? ((double)v[i]) : 1;
    }
    mu = pow(mu, 1.0 / len);
    
    return(mu);
}

double sigma_fn_sol(double *v, double mu, int len) {

    double sigma = 0.0;
    for (int i=0; i<len; i++) {
        sigma += ((double)v[i] - mu)*((double)v[i] - mu);
    }
    sigma /= (double)len;

    return(sigma);
}

double working_set_bytes(const COO& m){
    return (double)m.nnz * (2.0 * sizeof(int) + sizeof(float)) +
            (double)m.n_cols * sizeof(float) + (double)m.n_rows * sizeof(float);
}

double spmv_flop_count(int nnz) {
    return 2.0 * (double)nnz;
}

double ns_to_gflops(double flop_count, double runtime_ns) {
    const double runtime_s = runtime_ns * 1e-9;
    if (runtime_s <= 0.0) { return 0.0; }
    return (flop_count / runtime_s) / 1e9;
}

// COO matrix to CSR matrix convertion
CSR coo_to_csr(const COO& COO_matrix, double* convertion_time){

    TIMER_DEF(convertion_timer);
    TIMER_START(convertion_timer);

    CSR CSR_matrix;
    CSR_matrix.n_rows = COO_matrix.n_rows;
    CSR_matrix.n_cols = COO_matrix.n_cols;
    CSR_matrix.nnz = COO_matrix.nnz;

    CSR_matrix.row_indices = (int*)calloc((CSR_matrix.n_rows + 1), sizeof(int));
    CSR_matrix.col_indices = (int*)malloc(CSR_matrix.nnz * sizeof(int));
    CSR_matrix.values = (float*)malloc(CSR_matrix.nnz * sizeof(float));
    if(!CSR_matrix.row_indices || !CSR_matrix.col_indices || !CSR_matrix.values) {
        fprintf(stderr, "COO_malloc_matrix allocation failed");
        return {};
    }

    for (int i = 0; i < CSR_matrix.nnz; i++) {
        CSR_matrix.row_indices[COO_matrix.row_indices[i] + 1]++;
    }
    for (int i = 0; i < CSR_matrix.n_rows; i++) {
        CSR_matrix.row_indices[i + 1] += CSR_matrix.row_indices[i];
    }

    for (int i = 0; i < CSR_matrix.nnz; i++) {
        CSR_matrix.col_indices[i] = COO_matrix.col_indices[i];
        CSR_matrix.values[i] = COO_matrix.values[i];
    }

    TIMER_STOP(convertion_timer);
    *convertion_time = TIMER_ELAPSED(convertion_timer);
    return CSR_matrix;
}

void results_comparing(int n_rows, const char* v1, const float* COO_results, 
                        const char* v2, const float* CSR_results){
    int equal = 1;
    
    const double abs_tol = 1e-2;
    const double rel_tol = 1e-2;
    for(int i = 0; i < n_rows; i++){
        double a = (double)COO_results[i];
        double b = (double)CSR_results[i];
        double diff = fabs(a - b);
        double scale = fmax(fabs(a), fabs(b));

        // Mixed tolerance is robust for both near-zero and large-magnitude values.
        if(diff > (abs_tol + rel_tol * scale)) {
            equal = 0; 
            printf(" Mismatch at row %d: %f vs %f", i, a, b);
            break;
        }
    }
    if(equal == 1) {printf(" %s = %s", v1, v2);} 
    else {printf(" %s != %s", v1, v2);}
}

void save_matrix_number(std::string input, int matrix_number, double workset_bytes,
                        const std::filesystem::path& output_path) {
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream outFile(output_path, std::ios::app);
    if (outFile.is_open()) {
        outFile << matrix_number << " -> " << input << " " 
                << std::fixed << std::setprecision(0) << workset_bytes << "\n";
        outFile.close();
    }
}

void append_spmv_summary_csv(const std::filesystem::path& output_path,
                            const results_struct& summary) {
    std::filesystem::create_directories(output_path.parent_path());
    const bool file_exists = std::filesystem::exists(output_path);
    const bool write_header = !file_exists || std::filesystem::file_size(output_path) == 0;

    std::ofstream out_file(output_path, std::ios::app);
    if (!out_file.is_open()) {
        fprintf(stderr, "Cannot open CSV output file: %s\n", output_path.string().c_str());
        return;
    }

    if (write_header) {
        out_file << "name,working_bytes,parsing_time_ms,"
                 << "coo_time_us,coo_gflops,"
                 << "coo_parallel_time_us,coo_parallel_gflops,"
                 << "coo_cuda_basic_time_us,coo_cuda_basic_gflops,"
                 << "coo_cuda_custom_1_time_us,coo_cuda_custom_1_gflops,"
                 << "coo_cuda_cuSparse_time_us,coo_cuda_cuSparse_gflops,"
                 << "coo_to_csr_time_us,"
                 << "csr_time_us,csr_gflops,"
                 << "csr_parallel_time_us,csr_parallel_gflops,"
                 << "csr_cuda_basic_time_us,csr_cuda_basic_gflops,"
                 << "csr_cuda_custom_1_time_us,csr_cuda_custom_1_gflops,"
                 << "csr_cuda_custom_2_time_us,csr_cuda_custom_2_gflops,csr_custom_2_setup_time_us,"
                 << "csr_cuda_cuSparse_time_us,csr_cuda_cuSparse_gflops,"
                 << "total_time_ms\n";
    }

    out_file << summary.name << ","
             << std::fixed << std::setprecision(0) << summary.working_set_bytes << ","
             << std::setprecision(6) << (summary.parsing_time_ns / 1e6) << ","
             << (summary.coo_time_ns / 1e3) << ","
             << summary.coo_gflops << ","
             << (summary.coo_parallel_time_ns / 1e3) << ","
             << summary.coo_parallel_gflops << ","
             << (summary.coo_cuda_basic_time_ns / 1e3) << ","
             << summary.coo_cuda_basic_gflops << ","
             << (summary.coo_cuda_custom_1_time_ns / 1e3) << ","
             << summary.coo_cuda_custom_1_gflops << ","
             << (summary.coo_cuda_cuSparse_time_ns / 1e3) << ","
             << summary.coo_cuda_cuSparse_gflops << ","
             << (summary.coo_to_csr_time_ns / 1e3) << ","
             << (summary.csr_time_ns / 1e3) << ","
             << summary.csr_gflops << ","
             << (summary.csr_parallel_time_ns / 1e3) << ","
             << summary.csr_parallel_gflops << ","
             << (summary.csr_cuda_basic_time_ns / 1e3) << ","
             << summary.csr_cuda_basic_gflops << ","
             << (summary.csr_cuda_custom_1_time_ns / 1e3) << ","
             << summary.csr_cuda_custom_1_gflops << ","
             << (summary.csr_cuda_custom_2_time_ns / 1e3) << ","
             << summary.csr_cuda_custom_2_gflops << ","
             << summary.csr_custom_2_setup_time_us << ","
             << (summary.csr_cuda_cuSparse_time_ns / 1e3) << ","
             << summary.csr_cuda_cuSparse_gflops << ","
             << (summary.total_time_ns / 1e6) << "\n";
}

void data_printing(results_struct results, results_vectors_struct& res_vector, const COO& m){
    fprintf(stdout, "%s:\n", results.name.c_str());
    fprintf(stdout, "Parsing time: %.1f us (%.0f ms),  ", 
            results.parsing_time_ns/1e3, results.parsing_time_ns/1e6);
    fprintf(stdout, "stats: rows=%d cols=%d nnz=%d workset~%.2f MB\n",
            m.n_rows, m.n_cols, m.nnz, results.working_set_bytes / 1e6);
    fprintf(stdout, "COO SpMV: - serial: (%.1f us) (%.1f GFLOP/s)\n", results.coo_time_ns/1000, results.coo_gflops);
    fprintf(stdout, "          - parallel: (%.1f us) (%.1f GFLOP/s)\n", results.coo_parallel_time_ns/1000, results.coo_parallel_gflops);
    fprintf(stdout, "          - GPU basic: (%.1f us) (%.1f GFLOP/s)\n", results.coo_cuda_basic_time_ns/1000, results.coo_cuda_basic_gflops);
    fprintf(stdout, "          - GPU custom 1: (%.1f us) (%.1f GFLOP/s)\n", results.coo_cuda_custom_1_time_ns/1000, results.coo_cuda_custom_1_gflops);
    fprintf(stdout, "          - GPU cuSParse: (%.1f us) (%.1f GFLOP/s)\n", results.coo_cuda_cuSparse_time_ns/1000, results.coo_cuda_cuSparse_gflops);
    fprintf(stdout, "COO to CSR conversion time: %.1f us\n", results.coo_to_csr_time_ns/1000);
    fprintf(stdout, "CSR SpMV: - serial: (%.1f us) (%.1f GFLOP/s)\n", results.csr_time_ns/1000, results.csr_gflops);
    fprintf(stdout, "          - parallel: (%.1f us) (%.1f GFLOP/s)\n", results.csr_parallel_time_ns/1000, results.csr_parallel_gflops);
    fprintf(stdout, "          - GPU basic: (%.1f us) (%.1f GFLOP/s)\n", results.csr_cuda_basic_time_ns/1000, results.csr_cuda_basic_gflops);
    fprintf(stdout, "          - GPU custom 1: (%.1f us) (%.1f GFLOP/s)\n", results.csr_cuda_custom_1_time_ns/1000, results.csr_cuda_custom_1_gflops);
    fprintf(stdout, "          - GPU custom 2: (%.1f us) (%.1f GFLOP/s) [%.1f us setup time]\n", results.csr_cuda_custom_2_time_ns/1000, results.csr_cuda_custom_2_gflops, results.csr_custom_2_setup_time_us);
    fprintf(stdout, "          - GPU cuSParse: (%.1f us) (%.1f GFLOP/s)\n", results.csr_cuda_cuSparse_time_ns/1000, results.csr_cuda_cuSparse_gflops);
    fprintf(stdout, "Results check: ");
    results_comparing(m.n_rows, "COO", res_vector.COO_results, "COO_parallel", res_vector.COO_parallel_results);
    results_comparing(m.n_rows, "CSR", res_vector.CSR_results, "CSR_parallel", res_vector.CSR_parallel_results);
    results_comparing(m.n_rows, "COO", res_vector.COO_results, "CSR\n", res_vector.CSR_results);
    results_comparing(m.n_rows, "COO", res_vector.COO_results, "COO_cuda_basic", res_vector.COO_results_cuda_basic);
    results_comparing(m.n_rows, "COO", res_vector.COO_results, "COO_cuda_custom_1", res_vector.COO_results_cuda_custom_1);
    results_comparing(m.n_rows, "COO", res_vector.COO_results, "COO_cuda_cuSparse", res_vector.COO_results_cuda_cuSparse);
    results_comparing(m.n_rows, "CSR", res_vector.CSR_results, "CSR_cuda_basic", res_vector.CSR_results_cuda_basic);
    results_comparing(m.n_rows, "CSR", res_vector.CSR_results, "CSR_cuda_custom_1", res_vector.CSR_results_cuda_custom_1);
    results_comparing(m.n_rows, "CSR", res_vector.CSR_results, "CSR_cuda_custom_2", res_vector.CSR_results_cuda_custom_2);
    results_comparing(m.n_rows, "CSR", res_vector.CSR_results, "CSR_cuda_cuSparse", res_vector.CSR_results_cuda_cuSparse);
}






