/**
 * @file safecuda.h
 * @brief SafeCUDA expose for real CUDA functions
 * 
 * Header file containing expose to the real cuda functions
 * 
 * @author Navin <navinkumar.ao2022@vitstudent.ac.in>
 * @date 2025-07-05
 * @version 0.1.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 * 
 * Change Log:
 * - 2025-10-23: Added support for cudaGetLastError
 * - 2025-10-22: Removed redundancies, fixed styling added cudaMalloc
 * - 2025-10-18: Added basic lifecycle components to the header
 * - 2025-07-05: Initial implementation
 */

#ifndef SAFECUDA_H
#define SAFECUDA_H

#include <cuda_runtime.h>
#include <cstddef>

namespace safecuda
{

using cudaMalloc_t = cudaError_t (*)(void **, std::size_t);
using cudaMallocManaged_t = cudaError_t (*)(void **, std::size_t, unsigned int);
using cudaFree_t = cudaError_t (*)(void *);
using cudaDeviceSynchronize_t = cudaError_t (*)();
using cudaGetLastError_t = cudaError_t (*)();

extern cudaMalloc_t real_cudaMalloc;
extern cudaMallocManaged_t real_cudaMallocManaged;
extern cudaFree_t real_cudaFree;
extern cudaDeviceSynchronize_t real_cudaDeviceSynchronize;
extern cudaGetLastError_t real_cudaGetLastError;

void init_safecuda();
void shutdown_safecuda();

}

#endif // SAFECUDA_H
