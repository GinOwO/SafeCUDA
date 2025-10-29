#include <cuda_runtime.h>

extern "C" void launchMemHammer(float *buffer, int n);

// 5. purely synthetic test, has around 4gb of traffic

int main()
{
	const int n = 1024 * 256;
	float *d_data;

	cudaMalloc(&d_data, n * sizeof(float));
	cudaMemset(d_data, 1, n * sizeof(float));

	launchMemHammer(d_data, n);
	cudaDeviceSynchronize();

	cudaFree(d_data);

	return 0;
}
