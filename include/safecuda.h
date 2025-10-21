/**
 * @file safecuda.h
 * @brief SafeCUDA host-side implementation (placeholder)
 * 
 * Header file for safecuda.cpp
 * 
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-07-05
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 * 
 * Change Log:
 * - 2025-07-05: Initial implementation
 * - 2025-10-18: Added basic lifecycle components to the header
 */
#include "safecache.cuh"
#ifndef SAFECUDA_H
#define SAFECUDA_H

namespace safecuda{
	using cudaMallocManaged_t = cudaError_t (*)(void **, size_t, unsigned int);
    	using cudaFree_t = cudaError_t (*)(void *);

    	extern cudaMallocManaged_t real_cudaMallocManaged;
    	extern cudaFree_t real_cudaFree;

	void init_symbols() __attribute__((constructor));
	void shutdown() __attribute__((destructor));

	extern __managed__ safecuda::cache::DynamicCache* dynamic_cache;
}

#endif // SAFECUDA_H