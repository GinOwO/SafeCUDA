#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

extern "C" void launchInit(float *data, int n);
extern "C" void launchSumReduce(float *data, int n);

int main()
{
	const int n = 1024 * 1024;
	float *d_data;
	cudaMalloc(&d_data, n * sizeof(float));
	launchInit(d_data, n);
	cudaDeviceSynchronize();
	launchSumReduce(d_data, n);
	cudaDeviceSynchronize();
	cudaFree(d_data);
	return 0;
}
