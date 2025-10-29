#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

extern "C" void launchVecAdd(float *data, int n);
extern "C" void launchVecMul(float *data, int n);
extern "C" void launchVecScale(float *data, int n);

// 1. Large vector ops

int main()
{
	const int n = 1024 * 1024;
	float *d_data;
	cudaMalloc(&d_data, n * sizeof(float));
	launchVecAdd(d_data, n);
	launchVecMul(d_data, n);
	launchVecScale(d_data, n);
	cudaDeviceSynchronize();
	cudaFree(d_data);
	return 0;
}
