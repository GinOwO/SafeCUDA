#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

extern "C" void launchCopy(float *data, int n);
extern "C" void launchScale2(float *data, int n);

int main()
{
	const int n = 1024 * 1024;
	float *d_data;
	cudaMalloc(&d_data, n * sizeof(float));
	cudaMemset(d_data, 1, n * sizeof(float));
	launchCopy(d_data, n);
	launchScale2(d_data, n);
	cudaDeviceSynchronize();
	cudaFree(d_data);
	return 0;
}
