//
// Created by navin on 10/16/25.
//
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

int main() {
	void *ptr = nullptr;
	size_t size = 1024 * 1024; // 1 MB

	std::cout << "Calling cudaMallocManaged..." << std::endl;
	cudaError_t err = cudaMallocManaged(&ptr, size);
	if (err != cudaSuccess) {
		std::cerr << "cudaMallocManaged failed: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}

	std::cout << "cudaMallocManaged succeeded." << std::endl;
	cudaFree(ptr);
	return 0;
}