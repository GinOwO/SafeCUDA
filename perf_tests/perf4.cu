#include <cuda_runtime.h>

__global__ void computeHeavyKernel(float *data, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < n; i += stride) {
		float x = data[i];
#pragma unroll 5
		for (int k = 0; k < 100; ++k) {
			x = x * 1.0001f - 0.0003f * x * x + 0.5f * sinf(x);
		}
		data[i] = x;
	}
}

extern "C" void launchComputeHeavy(float *data, int n)
{
	int threads = 256;
	int blocks = (n + threads - 1) / threads;
	if (blocks > 65535)
		blocks = 65535; // limit grid size
	computeHeavyKernel<<<blocks, threads>>>(data, n);
}
