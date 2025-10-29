#include <cuda_runtime.h>

extern "C" void launchInit(float *data, int n);
extern "C" void launchSumReduce(float *data, int n);

__global__ void initKernel(float *data, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		data[idx] = idx * 1.1f;
}

__global__ void sumReduceKernel(float *data, int n)
{
	float sum = 0;
	for (int i = 0; i < n; i += blockDim.x * gridDim.x) {
		int idx = i + blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < n)
			sum += data[idx];
	}
	if (blockIdx.x == 0 && threadIdx.x == 0)
		data[0] = sum;
}

void launchInit(float *data, int n)
{
	int threads = 256, blocks = (n + threads - 1) / threads;
	initKernel<<<blocks, threads>>>(data, n);
}
void launchSumReduce(float *data, int n)
{
	int threads = 256, blocks = (n + threads - 1) / threads;
	sumReduceKernel<<<blocks, threads>>>(data, n);
}
