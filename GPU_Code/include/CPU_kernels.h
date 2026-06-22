#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H
#include "parser.h"

void COO_SpMV_serial (int warmup, int niter, const COO& COO_matrix, const float* random_vector,
                        float* COO_results, double* runs_time);
void COO_SpMV_parallel (int warmup, int niter, const COO& COO_matrix, const float* random_vector,  
                        float* COO_parallel_results, double* runs_time);

void CSR_SpMV_serial (int warmup, int niter, const CSR& CSR_matrix, const float* random_vector,  
                        float* CSR_results, double* runs_time);
void CSR_SpMV_parallel (int warmup, int niter, const CSR& CSR_matrix, const float* random_vector,  
                        float* CSR_parallel_results, double* runs_time);

void HYB_SpMV_serial (int warmup, int niter, const HybridMatrix& HYB_matrix, const float* random_vector, 
                      float* HYB_results, double* runs_time);
void HYB_SpMV_parallel (int warmup, int niter, const HybridMatrix& HYB_matrix, const float* random_vector,  
                        float* HYB_parallel_results, double* runs_time);
                        
#endif


