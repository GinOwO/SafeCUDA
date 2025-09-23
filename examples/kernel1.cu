#include <cuda_runtime.h>

__global__ void scaleArray(float *data, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		data[idx] *= 2.0f;
}

extern "C" void launchScaleArray(float *d_data, int n)
{
	int blockSize = 256;
	int gridSize = (n + blockSize - 1) / blockSize;
	scaleArray<<<gridSize, blockSize>>>(d_data, n);
	cudaDeviceSynchronize();
}
