#include <cuda_runtime.h>

extern "C" void launchCopy(float *data, int n);
extern "C" void launchScale2(float *data, int n);

__global__ void copyKernel(float *data, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n / 2)
		data[idx + n / 2] = data[idx];
}

__global__ void scaleKernel(float *data, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		data[idx] *= 3.2f;
}

void launchCopy(float *data, int n)
{
	int threads = 256, blocks = ((n / 2) + threads - 1) / threads;
	copyKernel<<<blocks, threads>>>(data, n);
}
void launchScale2(float *data, int n)
{
	int threads = 256, blocks = (n + threads - 1) / threads;
	scaleKernel<<<blocks, threads>>>(data, n);
}
