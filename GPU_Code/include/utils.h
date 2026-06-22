#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <string>
#include <filesystem>
#include "parser.h"

struct results_struct {
    std::string name;
    double working_set_bytes;
    double parsing_time_ns;

    double coo_time_ns;
    double coo_gflops;
    double coo_parallel_time_ns;
    double coo_parallel_gflops;

    double coo_cuda_basic_time_ns;
    double coo_cuda_basic_gflops;
    double coo_cuda_custom_1_time_ns;
    double coo_cuda_custom_1_gflops;
    double coo_cuda_cuSparse_time_ns;
    double coo_cuda_cuSparse_gflops;

    double coo_to_csr_time_ns;

    double csr_time_ns;
    double csr_gflops;
    double csr_parallel_time_ns;
    double csr_parallel_gflops;
    
    double csr_cuda_basic_time_ns;
    double csr_cuda_basic_gflops;
    double csr_cuda_custom_1_time_ns;
    double csr_cuda_custom_1_gflops;
    double csr_cuda_custom_2_time_ns;
    double csr_custom_2_setup_time_us;
    double csr_cuda_custom_2_gflops;
    double csr_cuda_cuSparse_time_ns;
    double csr_cuda_cuSparse_gflops;

    double total_time_ns;
};

struct results_vectors_struct {
    float* COO_results;
    float* COO_parallel_results;
    float* CSR_results;
    float* CSR_parallel_results;
    float* COO_results_cuda_basic;
    float* COO_results_cuda_custom_1;
    float* COO_results_cuda_cuSparse;
    float* CSR_results_cuda_basic;
    float* CSR_results_cuda_custom_1;
    float* CSR_results_cuda_custom_2;
    float* CSR_results_cuda_cuSparse;
};

struct matrix_data {
    int n_rows;
    int n_cols;
    unsigned long long nnz;
};

std::vector<int> stream_row_lengths_from_mtx(const std::filesystem::path& filepath, matrix_data& matrix_data);
int find_optimal_ellpack_k(int num_rows, const std::vector<int>& row_lengths);

double geometric_mean(const double *v, int len);
double arithmetic_mean(const double *v, int len);
double sigma_fn_sol(double *v, double mu, int len);

double working_set_bytes(const COO& m);
double spmv_flop_count(int nnz);
double ns_to_gflops(double flop_count, double runtime_ns);
CSR coo_to_csr(const COO& COO_matrix, double* convertion_time);
void results_comparing(int n_rows, const char* v1, const float* COO_results, 
                        const char* v2, const float* CSR_results);
void save_matrix_number(std::string input, int matrix_number, double workset_bytes,
                        const std::filesystem::path& output_path);
void append_spmv_summary_csv(const std::filesystem::path& output_path,
                            const results_struct& summary);
void data_printing(results_struct results, results_vectors_struct& res_vector, const COO& m);

#endif
