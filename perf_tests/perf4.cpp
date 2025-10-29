#include <cuda_runtime.h>
#include <iostream>

extern "C" void launchComputeHeavy(float *data, int n);

// 4. realistic higher end HPC, CFD, or deep learning loops

int main()
{
	const int n = 1024 * 1024 * 8; // 8M floats (~32MB)
	float *d_data;
	cudaMalloc(&d_data, n * sizeof(float));
	cudaMemset(d_data, 1, n * sizeof(float));

	launchComputeHeavy(d_data, n);
	cudaDeviceSynchronize();

	cudaFree(d_data);
	std::cout << "Perf6 realistic compute test done." << std::endl;
	return 0;
}
