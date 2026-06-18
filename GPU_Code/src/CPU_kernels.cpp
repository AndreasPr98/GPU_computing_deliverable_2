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
