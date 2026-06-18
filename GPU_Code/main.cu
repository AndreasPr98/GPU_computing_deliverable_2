#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>
#include <filesystem> 
#include <fstream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "time_management.h"
#include "parser.h"
#include "utils.h"
#include "CPU_kernels.h"
#include "GPU_kernels.h"

#define WARMUP 2
#define NITER 10


int main(int argc, char *argv[]) {

    /*  Time variables definition  */
    TIMER_DEF(total_timer);
    TIMER_START(total_timer);
    double* runs_time = (double*)malloc(sizeof(double)*NITER);
    
    /*  Input and output paths definition  */
    // Change base path with the path where you have the code and the matrixes folder
    const std::filesystem::path base_path = "/home/name.surname/GPU_computing"; // Change to adapt to you needs
    const std::filesystem::path target_dir = base_path / "Matrixes";
    const std::filesystem::path matrix_numbers_path = base_path / "GPU_Code" / "program_output" / "matrix_numbers.txt";
    const std::filesystem::path spmv_csv_path = base_path / "GPU_Code" / "program_output" / "spmv_results.csv";

    // Check if the matrixes directory exists
    if (!std::filesystem::exists(target_dir) || !std::filesystem::is_directory(target_dir)) {
        std::cerr << "Invalid directory path." << std::endl;
        return 1;
    }
    
    /*  Matrix path and number checks and elaborations  */
    // Collect matrix files and process them in deterministic alphabetical order.
    std::vector<std::filesystem::path> matrix_files;
    for (auto it = std::filesystem::recursive_directory_iterator(target_dir);
         it != std::filesystem::recursive_directory_iterator(); ++it) {
        // Stop the iteration to 1 "level"
        if (it.depth() > 1) { it.disable_recursion_pending(); continue; }

        const auto& entry = *it;
        if (entry.is_regular_file() && entry.path().extension() == ".mtx") {
            matrix_files.push_back(entry.path());
        }
    }

    // Sort the matrixes in alphabetical order
    std::sort(matrix_files.begin(), matrix_files.end(),
        [](const std::filesystem::path& a, const std::filesystem::path& b) {
            return a.filename().string() < b.filename().string();
        });

    // Check if the matrix number has been passed
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_number 1-" << matrix_files.size() << ">" << std::endl;
        free(runs_time);
        return 1;
    }

    // Check if the data passed through argv is actually a number
    char* endptr = nullptr;
    long requested_matrix_number = strtol(argv[1], &endptr, 10);
    if (endptr == argv[1] || *endptr != '\0') {
        std::cerr << "Invalid matrix number: " << argv[1] << std::endl;
        free(runs_time);
        return 1;
    }

    // Check if the number is in range of 1 - total_n_of_matrixes
    if (requested_matrix_number < 1 || (size_t)requested_matrix_number > matrix_files.size()) {
        std::cerr << "Matrix number must be in range [1, " << matrix_files.size() << "]." << std::endl;
        free(runs_time);
        return 1;
    }

    // Create the structs where all the results will be stored
    results_struct results;
    results_vectors_struct res_vector;

    /*  Process only the selected matrix from the sorted list  */
    std::filesystem::directory_entry entry(matrix_files[(size_t)requested_matrix_number - 1]);
        results.name = entry.path().filename().c_str();
        
        /*  Parse the matrix  */
        COO COO_matrix = parser(entry.path().string(), &results.parsing_time_ns);
        if(COO_matrix.row_indices == nullptr){return 1;}
        results.working_set_bytes = working_set_bytes(COO_matrix);
        const double spmv_flops = spmv_flop_count(COO_matrix.nnz);

        // Create and randomize the random vector for the SpMV operation
        float* random_vector = (float*)malloc(COO_matrix.n_cols * sizeof(float));
        for(int i = 0; i < COO_matrix.n_cols; i++){random_vector[i] = (rand() % 100) + 1;}
        
        // Create the vectors for the multiplication and result storing
        res_vector.COO_results = (float*)calloc(COO_matrix.n_rows, sizeof(float));
        res_vector.COO_parallel_results = (float*)calloc(COO_matrix.n_rows, sizeof(float));
        res_vector.CSR_results = (float*)calloc(COO_matrix.n_rows, sizeof(float));
        res_vector.CSR_parallel_results = (float*)calloc(COO_matrix.n_rows, sizeof(float));
        if(!random_vector || !res_vector.COO_results || !res_vector.CSR_results || !res_vector.COO_parallel_results || !res_vector.CSR_parallel_results) {
            fprintf(stderr, "random_vector || COO_results || COO_parallel_results || CSR_results || CSR_parallel_results allocation failed");
            return 1;
        }        

        /*  COO analysis  */
        // Do the COO matrix multiplication NITER times
        COO_SpMV_serial(WARMUP, NITER, COO_matrix, random_vector,
                            res_vector.COO_results, runs_time);          
        results.coo_time_ns = arithmetic_mean(runs_time, NITER);
        results.coo_gflops = ns_to_gflops(spmv_flops, results.coo_time_ns);
        
        // Parallel COO version
        COO_SpMV_parallel(WARMUP, NITER, COO_matrix, random_vector,
                            res_vector.COO_parallel_results, runs_time);
        results.coo_parallel_time_ns = arithmetic_mean(runs_time, NITER);                    
        results.coo_parallel_gflops = ns_to_gflops(spmv_flops, results.coo_parallel_time_ns);
        
        /*  CSR analysis  */
        // COO to CSR convertion
        CSR CSR_matrix = coo_to_csr(COO_matrix, &results.coo_to_csr_time_ns);
        if(CSR_matrix.row_indices == nullptr){return 1;}

        // Do the CSR matrix multiplication NITER times
        CSR_SpMV_serial(WARMUP, NITER, CSR_matrix, random_vector,  
                            res_vector.CSR_results, runs_time);
        results.csr_time_ns = arithmetic_mean(runs_time, NITER);
        results.csr_gflops = ns_to_gflops(spmv_flops, results.csr_time_ns);

        // Parallel CSR version
        CSR_SpMV_parallel(WARMUP, NITER, CSR_matrix, random_vector,  
                            res_vector.CSR_parallel_results, runs_time);
        results.csr_parallel_time_ns = arithmetic_mean(runs_time, NITER);
        results.csr_parallel_gflops = ns_to_gflops(spmv_flops, results.csr_parallel_time_ns);

        // Create a .txt file with the matrix number and name for later identification
        save_matrix_number(entry.path().filename().string(), (int)requested_matrix_number,
                            results.working_set_bytes, matrix_numbers_path);

        /*  GPU Analysis  */
        // Transfer matrix arrays to the GPU. The structs stay on the CPU and
        // their pointer fields reference device memory.
        COO GPU_COO_matrix = coo_upload_to_device(COO_matrix);
        CSR GPU_CSR_matrix = csr_upload_to_device(CSR_matrix);

        // Allocate and copy the random vector on GPU for CUDA SpMV also allocate a results vector
        float* GPU_random_vector = nullptr;
        float* GPU_results = nullptr;
        cuda_check(cudaMalloc(&GPU_random_vector, COO_matrix.n_cols * sizeof(float)));
        cuda_check(cudaMalloc(&GPU_results, COO_matrix.n_rows * sizeof(float)));
        cuda_check(cudaMemcpy(GPU_random_vector, random_vector, COO_matrix.n_cols * sizeof(float), cudaMemcpyHostToDevice));
        
        // Allocate CPU memory for the results
        res_vector.COO_results_cuda_basic = (float*)calloc(COO_matrix.n_rows, sizeof(float));
        res_vector.COO_results_cuda_custom_1 = (float*)calloc(COO_matrix.n_rows, sizeof(float));
        res_vector.COO_results_cuda_cuSparse = (float*)calloc(COO_matrix.n_rows, sizeof(float));
        res_vector.CSR_results_cuda_basic = (float*)calloc(COO_matrix.n_rows, sizeof(float));
        res_vector.CSR_results_cuda_custom_1 = (float*)calloc(COO_matrix.n_rows, sizeof(float));
        res_vector.CSR_results_cuda_custom_2 = (float*)calloc(COO_matrix.n_rows, sizeof(float));
        res_vector.CSR_results_cuda_cuSparse = (float*)calloc(COO_matrix.n_rows, sizeof(float));
        if( !res_vector.COO_results_cuda_basic || !res_vector.COO_results_cuda_custom_1 || !res_vector.COO_results_cuda_cuSparse ||
            !res_vector.CSR_results_cuda_basic || !res_vector.CSR_results_cuda_custom_1 ||
            !res_vector.CSR_results_cuda_custom_2 || !res_vector.CSR_results_cuda_cuSparse) {
            fprintf(stderr, "CUDA result vector allocation failed");
            return 1;
        }  
        
        // Execture SpMV with COO in GPU with basic tecnique 
        SpMV_cuda_basic(WARMUP, NITER, GPU_COO_matrix, GPU_random_vector, GPU_results, runs_time); 
        results.coo_cuda_basic_time_ns = arithmetic_mean(runs_time, NITER);
        results.coo_cuda_basic_gflops = ns_to_gflops(spmv_flops, results.coo_cuda_basic_time_ns);
        cuda_check(cudaMemcpy(res_vector.COO_results_cuda_basic, GPU_results, COO_matrix.n_rows * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Execute SpMV with COO in GPU with custom tecnique 1
        SpMV_cuda_custom_1(WARMUP, NITER, GPU_COO_matrix, GPU_random_vector, GPU_results, runs_time);
        results.coo_cuda_custom_1_time_ns = arithmetic_mean(runs_time, NITER);
        results.coo_cuda_custom_1_gflops = ns_to_gflops(spmv_flops, results.coo_cuda_custom_1_time_ns);
        cuda_check(cudaMemcpy(res_vector.COO_results_cuda_custom_1, GPU_results, COO_matrix.n_rows * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Execture SpMV with COO in GPU with the help of cuSparse nvidia library
        SpMV_cuda_cuSparse(WARMUP, NITER, GPU_COO_matrix, GPU_random_vector, GPU_results, runs_time);
        results.coo_cuda_cuSparse_time_ns = arithmetic_mean(runs_time, NITER);
        results.coo_cuda_cuSparse_gflops = ns_to_gflops(spmv_flops, results.coo_cuda_cuSparse_time_ns);
        cuda_check(cudaMemcpy(res_vector.COO_results_cuda_cuSparse, GPU_results, COO_matrix.n_rows * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Execture SpMV with CSR in GPU with basic tecnique 
        SpMV_cuda_basic(WARMUP, NITER, GPU_CSR_matrix, GPU_random_vector, GPU_results, runs_time); 
        results.csr_cuda_basic_time_ns = arithmetic_mean(runs_time, NITER);
        results.csr_cuda_basic_gflops = ns_to_gflops(spmv_flops, results.csr_cuda_basic_time_ns);
        cuda_check(cudaMemcpy(res_vector.CSR_results_cuda_basic, GPU_results, CSR_matrix.n_rows * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Execute SpMV with CSR in GPU with custom tecnique 1
        SpMV_cuda_custom_1(WARMUP, NITER, GPU_CSR_matrix, GPU_random_vector, GPU_results, runs_time);
        results.csr_cuda_custom_1_time_ns = arithmetic_mean(runs_time, NITER);
        results.csr_cuda_custom_1_gflops = ns_to_gflops(spmv_flops, results.csr_cuda_custom_1_time_ns);
        cuda_check(cudaMemcpy(res_vector.CSR_results_cuda_custom_1, GPU_results, CSR_matrix.n_rows * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Execute SpMV with CSR in GPU with custom tecnique 2
        SpMV_cuda_custom_2(WARMUP, NITER, GPU_CSR_matrix, GPU_random_vector, GPU_results, runs_time, &results.csr_custom_2_setup_time_us);
        results.csr_cuda_custom_2_time_ns = arithmetic_mean(runs_time, NITER);
        results.csr_cuda_custom_2_gflops = ns_to_gflops(spmv_flops, results.csr_cuda_custom_2_time_ns);
        cuda_check(cudaMemcpy(res_vector.CSR_results_cuda_custom_2, GPU_results, CSR_matrix.n_rows * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Execture SpMV with CSR in GPU with the help of cuSparse nvidia library
        SpMV_cuda_cuSparse(WARMUP, NITER, GPU_CSR_matrix, GPU_random_vector, GPU_results, runs_time);
        results.csr_cuda_cuSparse_time_ns = arithmetic_mean(runs_time, NITER);
        results.csr_cuda_cuSparse_gflops = ns_to_gflops(spmv_flops, results.csr_cuda_cuSparse_time_ns);
        cuda_check(cudaMemcpy(res_vector.CSR_results_cuda_cuSparse, GPU_results, CSR_matrix.n_rows * sizeof(float), cudaMemcpyDeviceToHost));

        TIMER_STOP(total_timer);
        results.total_time_ns = TIMER_ELAPSED(total_timer);   

        // Create csv file with all the information about the run
        append_spmv_summary_csv(spmv_csv_path, results);

        // Print all the data on the terminal
        data_printing(results, res_vector, COO_matrix);

    /*  Free memory for the next matrixes  */
    free(random_vector);
    free(runs_time);
    free(res_vector.COO_results);
    free(res_vector.COO_parallel_results);
    free(res_vector.CSR_results);
    free(res_vector.CSR_parallel_results);
    free(res_vector.COO_results_cuda_basic);
    free(res_vector.COO_results_cuda_custom_1);
    free(res_vector.COO_results_cuda_cuSparse);
    free(res_vector.CSR_results_cuda_basic);
    free(res_vector.CSR_results_cuda_custom_1);
    free(res_vector.CSR_results_cuda_custom_2);
    free(res_vector.CSR_results_cuda_cuSparse);
    // Free COO
    free(COO_matrix.row_indices);
    free(COO_matrix.col_indices);
    free(COO_matrix.values);
    cuda_check(cudaFree(GPU_COO_matrix.row_indices));
    cuda_check(cudaFree(GPU_COO_matrix.col_indices));
    cuda_check(cudaFree(GPU_COO_matrix.values));
    cuda_check(cudaFree(GPU_random_vector));
    cuda_check(cudaFree(GPU_results));
    // Free CSR
    cuda_check(cudaFree(GPU_CSR_matrix.row_indices));
    cuda_check(cudaFree(GPU_CSR_matrix.col_indices));
    cuda_check(cudaFree(GPU_CSR_matrix.values));
    free(CSR_matrix.row_indices);
    free(CSR_matrix.col_indices);
    free(CSR_matrix.values);
    
    fprintf(stdout, "\nProgram finished in %.3f s\n\n", TIMER_ELAPSED(total_timer)/1e9);
    return 0;
}
