#include "GPU_kernels.h"
#include "time_and_path_management.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>


// Functions for moving the data from CPU to GPU

/*  
    Allocates GPU memory and transfers the bulk data (row indices, column indices, and values) 
    of a COO matrix from the host to the device. Returns a COO struct by value that resides on the CPU, 
    but contains pointers to the newly allocated data in GPU memory.
*/
COO coo_upload_to_device(const COO& host_matrix) {
    COO device_data = host_matrix;
    cuda_check(cudaMalloc(&device_data.row_indices, host_matrix.nnz * sizeof(int)));
    cuda_check(cudaMalloc(&device_data.col_indices, host_matrix.nnz * sizeof(int)));
    cuda_check(cudaMalloc(&device_data.values, host_matrix.nnz * sizeof(float)));

    cuda_check(cudaMemcpy(device_data.row_indices, host_matrix.row_indices,
                          host_matrix.nnz * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(device_data.col_indices, host_matrix.col_indices,
                          host_matrix.nnz * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(device_data.values, host_matrix.values,
                          host_matrix.nnz * sizeof(float), cudaMemcpyHostToDevice));

    return device_data;
}

/*  
    Allocates GPU memory and transfers the bulk data (row indices, column indices, and values) 
    of a CSR matrix from the host to the device. Returns a CSR struct by value that resides on the CPU, 
    but contains pointers to the newly allocated data in GPU memory.
*/
CSR csr_upload_to_device(const CSR& host_matrix) {
    CSR device_data = host_matrix;
    cuda_check(cudaMalloc(&device_data.row_indices, (host_matrix.n_rows + 1) * sizeof(int)));
    cuda_check(cudaMalloc(&device_data.col_indices, host_matrix.nnz * sizeof(int)));
    cuda_check(cudaMalloc(&device_data.values, host_matrix.nnz * sizeof(float)));

    cuda_check(cudaMemcpy(device_data.row_indices, host_matrix.row_indices,
                          (host_matrix.n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(device_data.col_indices, host_matrix.col_indices,
                          host_matrix.nnz * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(device_data.values, host_matrix.values,
                          host_matrix.nnz * sizeof(float), cudaMemcpyHostToDevice));

    return device_data;
}


// SpMV kernels

/*  COO basic SpMV kernel (without the use of cache)  */
__global__ void coo_spmv_basic_kernel(int nnz, 
                                const int* __restrict__ row_indices, 
                                const int* __restrict__ col_indices, 
                                const float* __restrict__ values, 
                                const float* __restrict__ random_vector, 
                                float* result_vector) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        atomicAdd(&result_vector[row_indices[i]], values[i] * random_vector[col_indices[i]]);
    }
}

/*  COO custom SpMV kernel  */
__global__ void coo_spmv_custom_1_kernel(
    int nnz, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ col_indices, 
    const float* __restrict__ values, 
    const float* __restrict__ random_vector, 
    float* __restrict__ result_vector) 
{
    // Global thread index and lane index within the warp
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;

    int my_row = -1;
    float my_val = 0.0f;

    // Load data and compute the basic multiplication
    if (i < nnz) {
        my_row = row_indices[i];
        my_val = values[i] * random_vector[col_indices[i]];
    }

    // Reduction inside the threads of the same warp that have the same row index (result is on the far right thread)
    for (int offset = 1; offset < 32; offset *= 2) {
        // Get row and value from the thread 'offsetted' behind us
        int src_row = __shfl_up_sync(0xFFFFFFFF, my_row, offset);
        float src_val = __shfl_up_sync(0xFFFFFFFF, my_val, offset);

        // If the thread behind us is valid and belongs to the same row, add its value
        if (lane_id >= offset && src_row == my_row) {
            my_val += src_val;
        }
    }
    // Now the last thread within the ones with the same row value has the reduction result

    // Get the row index of the thread on the right
    int next_row = __shfl_down_sync(0xFFFFFFFF, my_row, 1);
    
    // See if the thread is the one on the far right that cumulated the result
    // a) It is the last lane in the warp (lane 31)
    // b) The next thread belongs to a different row
    // c) It is the absolute last element in the entire matrix (i == nnz - 1)
    bool is_tail = (lane_id == 31) || (next_row != my_row) || (i == nnz - 1);

    // If the thread is actually the one on the far right, than write the result with atomic to avoid crash
    if (i < nnz && is_tail) {
        atomicAdd(&result_vector[my_row], my_val);
    }
}

/*  CSR basic SpMV kernel (without the use of cache)  */
__global__ void csr_spmv_basic_kernel(int n_rows,
                                    const int* __restrict__ row_indices,
                                    const int* __restrict__ col_indices,
                                    const float* __restrict__ values,
                                    const float* __restrict__ random_vector,
                                    float* result_vector) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n_rows) {
        float sum = 0.0f;
        for (int k = row_indices[row]; k < row_indices[row + 1]; k++) {
            sum += values[k] * random_vector[col_indices[k]];
        }
        result_vector[row] = sum;
    }
}

/*  First CSR custom SpMV kernel  */
__global__ void csr_spmv_custom_1_kernel(int n_rows,
                                     const int* __restrict__ row_ptr,
                                     const int* __restrict__ col_indices,
                                     const float* __restrict__ values,
                                     const float* __restrict__ x,
                                     float* y) {

    // Determine which row this warp is responsible for
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32; // Thread's index within the warp 

    if (warp_id < n_rows) {
        int row_start = row_ptr[warp_id];
        int row_end   = row_ptr[warp_id + 1];
        float sum = 0.0f;

        // All 32 threads in the warp process the row in parallel
        // If row has 100 elements, each thread handles ~3 elements
        for (int i = row_start + lane; i < row_end; i += 32) {
            sum += values[i] * x[col_indices[i]];
            //int col = __ldg(&col_indices[i]); // Using read only shared memory doenst change the speed (i tried)
            //sum += __ldg(&values[i]) * __ldg(&x[col]); // Probably because values are __restrict__ so the
        }                                                // SM already does that on it's own

        // Parallel Reduction: Sum the 'sum' variables of all 32 threads
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Thread 0 of the warp now holds the final result for the row
        if (lane == 0) {
            y[warp_id] = sum;
        }
    }
}


/*  Preparation kernel for the second custom SpMV CSR kernel,  
    counts the chunks of data that there are in each row, 
    it depends on how big is the choosen chunk.
*/
template<int CHUNK_SIZE>
__global__ void csr_row_chunk_count_kernel(int n_rows,
                                           const int* __restrict__ row_ptr,
                                           int* __restrict__ chunk_counts) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n_rows) {
        int row_len = row_ptr[row + 1] - row_ptr[row];
        chunk_counts[row] = (row_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
    }
}

/* 
   Preparation kernel for the second custom CSR SpMV kernel.
   It reads the prefix-summed 'chunk_offsets' array to populate:
     - chunk_rows[]: Maps each global chunk ID to its corresponding row ID 
                     (to help the assigned thread/warp find its target row).
     - chunk_starts[]: Stores the local starting element offset inside that row 
                       (to find the precise starting point inside that row).
*/
template<int CHUNK_SIZE>
__global__ void csr_chunk_map_kernel(int n_rows,
                                     const int* __restrict__ row_ptr,
                                     const int* __restrict__ chunk_offsets,
                                     int* __restrict__ chunk_rows,
                                     int* __restrict__ chunk_starts) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n_rows) {
        int row_len = row_ptr[row + 1] - row_ptr[row];
        int row_chunks = (row_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
        int first_chunk = chunk_offsets[row];

        for (int chunk = 0; chunk < row_chunks; chunk++) {
            int chunk_id = first_chunk + chunk;
            chunk_rows[chunk_id] = row;
            chunk_starts[chunk_id] = chunk * CHUNK_SIZE;
        }
    }
}

/*  Second CSR custom SpMV kernel  */
template<int V, int CHUNK_SIZE>
__global__ void csr_spmv_custom_2_kernel(
    int total_chunks,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const int* __restrict__ chunk_rows,
    const int* __restrict__ chunk_starts,
    const float* __restrict__ x,
    float* y)
{
    // Get the chung id and the thread id inside each compute unit
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_id = global_thread_id / V;
    int vector_lane = threadIdx.x % V;

    if (chunk_id >= total_chunks) {
        return;
    }

    // Get the necessary parameters for each compute unit
    int row = chunk_rows[chunk_id];
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];
    int row_len = row_end - row_start;
    int chunk_start = row_start + chunk_starts[chunk_id];
    int chunk_end = min(chunk_start + CHUNK_SIZE, row_end);

    // Each thread computes CHUNK_SIZE/V multiplications with a stride of V between each value
    float sum = 0.0f;
    for (int i = chunk_start + vector_lane; i < chunk_end; i += V) {
        sum += values[i] * x[col_indices[i]];
    }

    // Reduction of the V threads to the first
    for (int offset = V / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset, V);
    }

    // If the row lenght is shoter then the CHUNK_SIZE there is no need for a heavy atomicAdd
    if (vector_lane == 0) {
        if (row_len <= CHUNK_SIZE) {
            y[row] = sum;
        } else {
            atomicAdd(&y[row], sum);
        }
    }
}


// Functions for the setup and use of the kernels

/*  Function for the use of basic SpMV kernels (both COO and CSR)  */
template <typename MatrixT>
void SpMV_cuda_basic(int warmup, int niter, const MatrixT& matrix,
                     const float* random_vector,
                     float* result_vector,
                     double* runs_time) {
    
    int threadsPerBlock = 256;
    int blocksPerGrid;

    if constexpr (std::is_same_v<MatrixT, COO>) {
        blocksPerGrid = (matrix.nnz + threadsPerBlock - 1) / threadsPerBlock;
    } else if constexpr (std::is_same_v<MatrixT, CSR>) {
        // Use one warp (32 threads) per row to match the warp-based kernel
        threadsPerBlock = 256;
        blocksPerGrid = (matrix.n_rows + threadsPerBlock - 1) / threadsPerBlock;
    }

    cudaEvent_t start_evt, stop_evt;
    cuda_check(cudaEventCreate(&start_evt));
    cuda_check(cudaEventCreate(&stop_evt));

    for (int i = -warmup; i < niter; i++) {
        cuda_check(cudaMemset(result_vector, 0, matrix.n_rows * sizeof(float)));

        if (i >= 0) {
            cuda_check(cudaEventRecord(start_evt));
        }

        if constexpr (std::is_same_v<MatrixT, COO>) {
            coo_spmv_basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                matrix.nnz,
                matrix.row_indices,
                matrix.col_indices,
                matrix.values,
                random_vector,
                result_vector
            );
        } else if constexpr (std::is_same_v<MatrixT, CSR>) {
            csr_spmv_basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                matrix.n_rows,
                matrix.row_indices,
                matrix.col_indices,
                matrix.values,
                random_vector,
                result_vector
            );
        }

        cuda_check(cudaGetLastError());

        if (i >= 0) {
            cuda_check(cudaEventRecord(stop_evt));
            cuda_check(cudaEventSynchronize(stop_evt));

            float elapsed_ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
            runs_time[i] = static_cast<double>(elapsed_ms) * 1e6;
        } else {
            cuda_check(cudaDeviceSynchronize());
        }
    }

    cuda_check(cudaEventDestroy(start_evt));
    cuda_check(cudaEventDestroy(stop_evt));
}

template void SpMV_cuda_basic<COO>(
    int, int, const COO&, const float*, float*, double*
);

template void SpMV_cuda_basic<CSR>(
    int, int, const CSR&, const float*, float*, double*
);


/*  Function for the use of the first custom SpMV kernels (custom_1) (both COO and CSR)  */
template <typename MatrixT>
void SpMV_cuda_custom_1(int warmup, int niter, const MatrixT& matrix,
                        const float* random_vector,
                        float* result_vector,
                        double* runs_time) {
    
    int threadsPerBlock = 256;
    int blocksPerGrid;

    if constexpr (std::is_same_v<MatrixT, COO>) {
        blocksPerGrid = (matrix.nnz + threadsPerBlock - 1) / threadsPerBlock;
    } else if constexpr (std::is_same_v<MatrixT, CSR>) {
        blocksPerGrid = (matrix.n_rows * 32 + threadsPerBlock - 1) / threadsPerBlock;
    }

    cudaEvent_t start_evt, stop_evt;
    cuda_check(cudaEventCreate(&start_evt));
    cuda_check(cudaEventCreate(&stop_evt));

    for (int i = -warmup; i < niter; i++) {
        cuda_check(cudaMemset(result_vector, 0, matrix.n_rows * sizeof(float)));

        if (i >= 0) {
            cuda_check(cudaEventRecord(start_evt));
        }

        if constexpr (std::is_same_v<MatrixT, COO>) {
            coo_spmv_custom_1_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                matrix.nnz,
                matrix.row_indices,
                matrix.col_indices,
                matrix.values,
                random_vector,
                result_vector
            );
        } else if constexpr (std::is_same_v<MatrixT, CSR>) {
            csr_spmv_custom_1_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                matrix.n_rows,
                matrix.row_indices,
                matrix.col_indices,
                matrix.values,
                random_vector,
                result_vector
            );
        }

        cuda_check(cudaGetLastError());

        if (i >= 0) {
            cuda_check(cudaEventRecord(stop_evt));
            cuda_check(cudaEventSynchronize(stop_evt));

            float elapsed_ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
            runs_time[i] = static_cast<double>(elapsed_ms) * 1e6;
        } else {
            cuda_check(cudaDeviceSynchronize());
        }
    }

    cuda_check(cudaEventDestroy(start_evt));
    cuda_check(cudaEventDestroy(stop_evt));
}

template void SpMV_cuda_custom_1<COO>(
    int, int, const COO&, const float*, float*, double*
);

template void SpMV_cuda_custom_1<CSR>(
    int, int, const CSR&, const float*, float*, double*
);



/*  Function for the use of the second custom SpMV kernel (custom_2) (CSR only)  */
void SpMV_cuda_custom_2(int warmup, int niter, const CSR& matrix,
                        const float* random_vector, float* result_vector,
                        double* runs_time, double* setup_time) {
    
    int threadsPerBlock = 256;
    int blocksPerGrid;

    /*  Parameters for the Work-Chunk logic, test:
        4-64 always better than 4-32 and of 8-64 and almost always of 8-128,
        2-32 also very good (better than 4-64 half the times)
    */
    constexpr int V = 4;           // Vector size (threads per row/chunk unit)(compute unit size)
    constexpr int CHUNK_SIZE = 64; // (max) NNZ elements per chunk

    int total_chunks = 0;
    int* d_chunk_counts = nullptr;
    int* d_chunk_offsets = nullptr;
    int* d_chunk_rows = nullptr;
    int* d_chunk_starts = nullptr;

    cudaEvent_t setup_time_start, setup_time_end;
    cuda_check(cudaEventCreate(&setup_time_start));
    cuda_check(cudaEventCreate(&setup_time_end));
    cuda_check(cudaEventRecord(setup_time_start));

    if (matrix.n_rows > 0 && matrix.nnz > 0) {
        
        int row_blocks = (matrix.n_rows + threadsPerBlock - 1) / threadsPerBlock;

        cuda_check(cudaMalloc(&d_chunk_counts, matrix.n_rows * sizeof(int)));
        cuda_check(cudaMalloc(&d_chunk_offsets, matrix.n_rows * sizeof(int)));

        csr_row_chunk_count_kernel<CHUNK_SIZE><<<row_blocks, threadsPerBlock>>>(
            matrix.n_rows, matrix.row_indices, d_chunk_counts
        );
        cuda_check(cudaGetLastError());

        // Perform a parallel prefix sum to know the starting point of each chunk
        thrust::exclusive_scan(thrust::device,
                               thrust::device_pointer_cast(d_chunk_counts),
                               thrust::device_pointer_cast(d_chunk_counts + matrix.n_rows),
                               thrust::device_pointer_cast(d_chunk_offsets));

        int last_count = 0;
        int last_offset = 0;
        cuda_check(cudaMemcpy(&last_count, d_chunk_counts + matrix.n_rows - 1,
                              sizeof(int), cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(&last_offset, d_chunk_offsets + matrix.n_rows - 1,
                              sizeof(int), cudaMemcpyDeviceToHost));
        total_chunks = last_offset + last_count;

        cuda_check(cudaMalloc(&d_chunk_rows, total_chunks * sizeof(int)));
        cuda_check(cudaMalloc(&d_chunk_starts, total_chunks * sizeof(int)));

        csr_chunk_map_kernel<CHUNK_SIZE><<<row_blocks, threadsPerBlock>>>(
            matrix.n_rows, matrix.row_indices, d_chunk_offsets,
            d_chunk_rows, d_chunk_starts
        );
        cuda_check(cudaGetLastError());
        cuda_check(cudaDeviceSynchronize());
    } else {    
        cuda_check(cudaEventDestroy(setup_time_start));
        cuda_check(cudaEventDestroy(setup_time_end));
        return;
    }

    cuda_check(cudaEventRecord(setup_time_end));
    cuda_check(cudaEventSynchronize(setup_time_end));

    float setup_ms = 0.0f; 
    cuda_check(cudaEventElapsedTime(&setup_ms, setup_time_start, setup_time_end)); 
    *setup_time = static_cast<double>(setup_ms*1000);
    
    cuda_check(cudaEventDestroy(setup_time_start));
    cuda_check(cudaEventDestroy(setup_time_end));

    int vectors_per_block = threadsPerBlock / V;
    blocksPerGrid = (total_chunks + vectors_per_block - 1) / vectors_per_block;


    cudaEvent_t start_evt, stop_evt;
    cuda_check(cudaEventCreate(&start_evt));
    cuda_check(cudaEventCreate(&stop_evt));

    for (int i = -warmup; i < niter; i++) {
        // Clear result vector before each timed/warmup run.
        cuda_check(cudaMemset(result_vector, 0, matrix.n_rows * sizeof(float)));

        if (matrix.nnz == 0 || matrix.n_rows == 0) {
            if (i >= 0) {
                runs_time[i] = 0.0;
            }
            continue;
        }

        if (i >= 0) {
            cuda_check(cudaEventRecord(start_evt));
        }

        csr_spmv_custom_2_kernel<V, CHUNK_SIZE><<<blocksPerGrid, threadsPerBlock>>>(
            total_chunks,
            matrix.row_indices, 
            matrix.col_indices,
            matrix.values,
            d_chunk_rows,
            d_chunk_starts,
            random_vector,
            result_vector
        );

        cuda_check(cudaGetLastError());

        if (i >= 0) {
            cuda_check(cudaEventRecord(stop_evt));
            cuda_check(cudaEventSynchronize(stop_evt));
            float elapsed_ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
            runs_time[i] = static_cast<double>(elapsed_ms) * 1e6;
        } else {
            cuda_check(cudaDeviceSynchronize());
        }
    }

    cuda_check(cudaEventDestroy(start_evt));
    cuda_check(cudaEventDestroy(stop_evt));

    cuda_check(cudaFree(d_chunk_counts));
    cuda_check(cudaFree(d_chunk_offsets));
    cuda_check(cudaFree(d_chunk_rows));
    cuda_check(cudaFree(d_chunk_starts));

}



/*  SpMV with the use of the cuSparse library from Nvidia (both COO and CSR)  */
template <typename MatrixT>
void SpMV_cuda_cuSparse(int warmup, int niter, const MatrixT& matrix,
                        const float* random_vector,
                        float* result_vector,
                        double* runs_time) {
    cusparseHandle_t handle;
    cuda_check(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    if constexpr (std::is_same_v<MatrixT, COO>) {
        cuda_check(cusparseCreateCoo(
            &matA,
            static_cast<int64_t>(matrix.n_rows),
            static_cast<int64_t>(matrix.n_cols),
            static_cast<int64_t>(matrix.nnz),
            matrix.row_indices,
            matrix.col_indices,
            matrix.values,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F));
    } else if constexpr (std::is_same_v<MatrixT, CSR>) {
        cuda_check(cusparseCreateCsr(
            &matA,
            static_cast<int64_t>(matrix.n_rows),
            static_cast<int64_t>(matrix.n_cols),
            static_cast<int64_t>(matrix.nnz),
            matrix.row_indices,
            matrix.col_indices,
            matrix.values,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F));
    }

    cuda_check(cusparseCreateDnVec(
        &vecX,
        static_cast<int64_t>(matrix.n_cols),
        const_cast<float*>(random_vector),
        CUDA_R_32F));

    cuda_check(cusparseCreateDnVec(
        &vecY,
        static_cast<int64_t>(matrix.n_rows),
        result_vector,
        CUDA_R_32F));

    float alpha = 1.0f;
    float beta = 0.0f;
    size_t bufferSize = 0;
    cuda_check(cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize));

    void* dBuffer = nullptr;
    if (bufferSize > 0) {
        cuda_check(cudaMalloc(&dBuffer, bufferSize));
    }

    cudaEvent_t start_evt, stop_evt;
    cuda_check(cudaEventCreate(&start_evt));
    cuda_check(cudaEventCreate(&stop_evt));

    for (int i = -warmup; i < niter; i++) {
        cuda_check(cudaMemset(result_vector, 0, matrix.n_rows * sizeof(float)));
        if (i >= 0) { cuda_check(cudaEventRecord(start_evt)); }

        cuda_check(cusparseSpMV(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            vecX,
            &beta,
            vecY,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            dBuffer));

        if (i >= 0) {
            cuda_check(cudaEventRecord(stop_evt));
            cuda_check(cudaEventSynchronize(stop_evt));
        } else {
            cuda_check(cudaDeviceSynchronize());
        }

        if (i >= 0) {
            float elapsed_ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
            runs_time[i] = static_cast<double>(elapsed_ms) * 1e6;
        }
    }

    cuda_check(cudaEventDestroy(start_evt));
    cuda_check(cudaEventDestroy(stop_evt));

    if (dBuffer != nullptr) {
        cuda_check(cudaFree(dBuffer));
    }
    cuda_check(cusparseDestroySpMat(matA));
    cuda_check(cusparseDestroyDnVec(vecX));
    cuda_check(cusparseDestroyDnVec(vecY));
    cuda_check(cusparseDestroy(handle));
}

template void SpMV_cuda_cuSparse<COO>(
    int, int, const COO&, const float*, float*, double*
);

template void SpMV_cuda_cuSparse<CSR>(
    int, int, const CSR&, const float*, float*, double*
);
