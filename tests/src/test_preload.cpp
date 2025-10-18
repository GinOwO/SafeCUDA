//
// Created by navin on 10/16/25.
//
// g++ tests/src/test_preload.cpp -o test_preload -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
// LD_PRELOAD=./cmake-build-Debug/libsafecuda.so ./test_preload

#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

int main() {
	int *ptr = nullptr;
	size_t size = 5 * sizeof(int);

	std::cout << "[Test] Calling cudaMallocManaged..." << std::endl;
	cudaError_t err = cudaMallocManaged(&ptr, size);
	if (err != cudaSuccess) {
		std::cerr << "[Test] cudaMallocManaged failed: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	else {
		std::cout << "[Test] cudaMallocManaged succeeded." << std::endl;
	}

	for(int i = 0; i < 5; i++) {
		ptr[i] = i;
		std::cout << ptr[i] << std::endl;
	}

	cudaFree(ptr);
	std::cout << "[Test] cudaFree succeeded." << std::endl;
	return 0;
}