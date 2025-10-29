#include <cuda_runtime.h>

extern "C" void launchVecAdd(float *data, int n);
extern "C" void launchVecMul(float *data, int n);
extern "C" void launchVecScale(float *data, int n);

__global__ void vecAdd(float *data, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		data[idx] = data[idx] + 2.0f;
	}
}

__global__ void vecMul(float *data, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		data[idx] = data[idx] * 2.0f;
	}
}

__global__ void vecScale(float *data, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		data[idx] = data[idx] * 1.5f + 4.0f;
	}
}

void launchVecAdd(float *data, int n)
{
	int threads = 256;
	int blocks = (n + threads - 1) / threads;
	vecAdd<<<blocks, threads>>>(data, n);
}

void launchVecMul(float *data, int n)
{
	int threads = 256;
	int blocks = (n + threads - 1) / threads;
	vecMul<<<blocks, threads>>>(data, n);
}

void launchVecScale(float *data, int n)
{
	int threads = 256;
	int blocks = (n + threads - 1) / threads;
	vecScale<<<blocks, threads>>>(data, n);
}
