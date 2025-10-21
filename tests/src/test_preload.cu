//
// Created by navin on 10/16/25.
//
// g++ tests/src/test_preload.cu -o test_preload -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
// LD_PRELOAD=./cmake-build-Debug/libsafecuda.so ./test_preload

#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

__global__ void incrementKernel(int* data, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		data[idx] += 10;
	}
}

int main() {
	int *ptr = nullptr;
	size_t n = 5;
	size_t size = n * sizeof(int);

	std::cout << "[Test] Calling cudaMallocManaged..." << std::endl;
	cudaError_t err = cudaMallocManaged(&ptr, size);
	if (err != cudaSuccess) {
		std::cerr << "[Test] cudaMallocManaged failed: " << cudaGetErrorString(err) << std::endl;
		return 1;
	} else {
		std::cout << "[Test] cudaMallocManaged succeeded." << std::endl;
	}

	for (int i = 0; i < n; i++) {
		ptr[i] = i;
	}

	std::cout << "[Test] Before GPU kernel:" << std::endl;
	for (int i = 0; i < n; i++) {
		std::cout << "  ptr[" << i << "] = " << ptr[i] << std::endl;
	}

	incrementKernel<<<1, 32>>>(ptr, n);
	cudaDeviceSynchronize();

	std::cout << "[Test] After GPU kernel:" << std::endl;
	for (int i = 0; i < n; i++) {
		std::cout << "  ptr[" << i << "] = " << ptr[i] << std::endl;
	}

	cudaFree(ptr);
	std::cout << "[Test] cudaFree succeeded." << std::endl;
	return 0;
}
