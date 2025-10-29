#include <cuda_runtime.h>

__global__ void memHammerKernel(float *data, int n)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int iter = 0; iter < 1500; ++iter) {
		for (int i = tid; i < n; i += stride) {
			float v = data[i];
			v = v * 1.001f + 0.0001f * (iter & 255);
			data[i] = v;
		}
		__syncthreads();
	}
}

extern "C" void launchMemHammer(float *buffer, int n)
{
	int threads = 256;
	int blocks = 512;
	memHammerKernel<<<blocks, threads>>>(buffer, n);
}
