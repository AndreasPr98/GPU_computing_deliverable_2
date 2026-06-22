#include "CPU_kernels.h"
#include "time_and_path_management.h"
#include <omp.h>

// COO operations
void COO_SpMV_serial (int warmup, int niter, const COO& COO_matrix, const float* random_vector,  
                        float* COO_results, double* runs_time){

    TIMER_DEF(kernel_time);

    // Do the COO matrix multiplication NITER times
    for(int i = -warmup; i < niter; i++){

        for(int j = 0; j<COO_matrix.n_rows; j++){COO_results[j] = 0;}

        if(i >= 0){TIMER_START(kernel_time);}

        for(int j = 0; j<COO_matrix.nnz; j++){
            COO_results[COO_matrix.row_indices[j]] += COO_matrix.values[j] 
                * random_vector[COO_matrix.col_indices[j]];
        }

        if(i >= 0){TIMER_STOP(kernel_time); runs_time[i] = TIMER_ELAPSED(kernel_time);}
    }
}

void COO_SpMV_parallel (int warmup, int niter, const COO& COO_matrix, const float* random_vector,  
                        float* COO_parallel_results, double* runs_time){

    TIMER_DEF(kernel_time);

    // Parallel COO version
    for (int i = -warmup; i < niter; i++) {
        
        #pragma omp parallel for
        for (int j = 0; j < COO_matrix.n_rows; j++) {COO_parallel_results[j] = 0.0f;}
    
        if (i >= 0) { TIMER_START(kernel_time); }
    
        #pragma omp parallel for
        for (int j = 0; j < COO_matrix.nnz; j++) {
            float val = COO_matrix.values[j] * random_vector[COO_matrix.col_indices[j]];
                     
            #pragma omp atomic
            COO_parallel_results[COO_matrix.row_indices[j]] += val;
        }
    
        if (i >= 0) {TIMER_STOP(kernel_time); runs_time[i] = TIMER_ELAPSED(kernel_time);}
    }
}

// CSR operations
void CSR_SpMV_serial (int warmup, int niter, const CSR& CSR_matrix, const float* random_vector,  
                        float* CSR_results, double* runs_time){

    TIMER_DEF(kernel_time);

    for(int i = -warmup; i < niter; i++){
        for(int j = 0; j<CSR_matrix.n_rows; j++){CSR_results[j] = 0;}
        if(i >= 0){TIMER_START(kernel_time);}
        for (int j = 0; j < CSR_matrix.n_rows; j++) {
            int row_start = CSR_matrix.row_indices[j];
            int row_end   = CSR_matrix.row_indices[j + 1];
            
            for (int k = row_start; k < row_end; k++) {
                CSR_results[j] += CSR_matrix.values[k] * random_vector[CSR_matrix.col_indices[k]];
            }
        }
        if(i >= 0){TIMER_STOP(kernel_time); runs_time[i] = TIMER_ELAPSED(kernel_time);}
    }
}

void CSR_SpMV_parallel (int warmup, int niter, const CSR& CSR_matrix, const float* random_vector,  
                        float* CSR_parallel_results, double* runs_time){

    TIMER_DEF(kernel_time);

    for (int i = -warmup; i < niter; i++) {
        #pragma omp parallel for
        for (int j = 0; j < CSR_matrix.n_rows; j++) {
            CSR_parallel_results[j] = 0.0f;
        }
        if (i >= 0) { TIMER_START(kernel_time); }
    
        // 2. Parallelize the row computation
        #pragma omp parallel for
        for (int j = 0; j < CSR_matrix.n_rows; j++) {
            int row_start = CSR_matrix.row_indices[j];
            int row_end   = CSR_matrix.row_indices[j + 1];
        
            for (int k = row_start; k < row_end; k++) {
                CSR_parallel_results[j] += CSR_matrix.values[k] * random_vector[CSR_matrix.col_indices[k]];
            }
        }
    
        if (i >= 0) { TIMER_STOP(kernel_time); runs_time[i] = TIMER_ELAPSED(kernel_time); }
    }
}

void HYB_SpMV_serial (int warmup, int niter, const HybridMatrix& HYB_matrix, const float* random_vector,   
                      float* HYB_results, double* runs_time) {

    TIMER_DEF(kernel_time);
    
    // Deconstruct fields for easier syntax tracking
    const int rows = HYB_matrix.ell.n_rows;
    const int K = HYB_matrix.ell.K;

    for (int i = -warmup; i < niter; i++) {

        // 1. Reset vector values across the rows
        for (int j = 0; j < rows; j++) { 
            HYB_results[j] = 0.0f; 
        }

        if (i >= 0) { TIMER_START(kernel_time); }

        // 2. Compute ELLPACK Portion
        for (int r = 0; r < rows; ++r) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                size_t idx = (size_t)r * K + k;
                int col = HYB_matrix.ell.col_indices[idx];
                
                // -1 indicates ELLPACK structural padding; break or skip
                if (col == -1) continue; 
                
                sum += HYB_matrix.ell.values[idx] * random_vector[col];
            }
            HYB_results[r] = sum;
        }

        // 3. Compute CSR Spillover Portion (Accumulating into results)
        for (int r = 0; r < rows; ++r) {
            int row_start = HYB_matrix.csr.row_ptr[r];
            int row_end   = HYB_matrix.csr.row_ptr[r + 1];
            
            float sum = 0.0f;
            for (int k = row_start; k < row_end; k++) {
                sum += HYB_matrix.csr.values[k] * random_vector[HYB_matrix.csr.col_indices[k]];
            }
            HYB_results[r] += sum; // Accumulate spillover elements
        }

        if (i >= 0) { 
            TIMER_STOP(kernel_time); 
            runs_time[i] = TIMER_ELAPSED(kernel_time); 
        }
    }
}

void HYB_SpMV_parallel (int warmup, int niter, const HybridMatrix& HYB_matrix, const float* random_vector,  
                        float* HYB_parallel_results, double* runs_time) {

    TIMER_DEF(kernel_time);
    
    const int rows = HYB_matrix.ell.n_rows;
    const int K = HYB_matrix.ell.K;

    for (int i = -warmup; i < niter; i++) {
        
        // 1. Parallel Initialization
        #pragma omp parallel for
        for (int j = 0; j < rows; j++) { 
            HYB_parallel_results[j] = 0.0f; 
        }
    
        if (i >= 0) { TIMER_START(kernel_time); }
    
        // 2 & 3. Fused Parallel Computation Loop
        // Because both layouts separate data by row mapping, we can execute them
        // inside the same parallel workshare loop safely without race conditions!
        #pragma omp parallel for
        for (int r = 0; r < rows; ++r) {
            float row_accumulator = 0.0f;

            // Compute ELL segment for row 'r'
            for (int k = 0; k < K; ++k) {
                size_t idx = (size_t)r * K + k;
                int col = HYB_matrix.ell.col_indices[idx];
                if (col == -1) continue;
                
                row_accumulator += HYB_matrix.ell.values[idx] * random_vector[col];
            }

            // Compute CSR spillover segment for row 'r'
            int row_start = HYB_matrix.csr.row_ptr[r];
            int row_end   = HYB_matrix.csr.row_ptr[r + 1];
            for (int k = row_start; k < row_end; k++) {
                row_accumulator += HYB_matrix.csr.values[k] * random_vector[HYB_matrix.csr.col_indices[k]];
            }

            // Single unified write per row out to global RAM memory
            HYB_parallel_results[r] = row_accumulator;
        }
    
        if (i >= 0) { 
            TIMER_STOP(kernel_time); 
            runs_time[i] = TIMER_ELAPSED(kernel_time); 
        }
    }
}













