#include <cuda_runtime.h>

__global__ void memHammerKernel(float *data, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < n; i += stride) {
		float v = data[i];
		v = v * 1.01f + 0.1f;
		data[i] = v;
	}
}

extern "C" void launchMemHammer(float *buffer, int n)
{
	int threads = 256;
	int blocks = 256;
	memHammerKernel<<<blocks, threads>>>(buffer, n);
}
