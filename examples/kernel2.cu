#include <cuda_runtime.h>

__global__ void addOne(float *data, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		data[idx] += 1.0f;
}

extern "C" void launchAddOne(float *d_data, int n)
{
	int blockSize = 256;
	int gridSize = (n + blockSize - 1) / blockSize;
	addOne<<<gridSize, blockSize>>>(d_data, n);
	cudaDeviceSynchronize();
}
