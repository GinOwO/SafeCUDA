#include <cuda_runtime.h>
#include <stdio.h>

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

__global__ void computeSeriesKernel(unsigned long long int *out, int samples)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	unsigned long long int local = 0;
	for (int i = idx; i < samples; i += stride) {
		float x = (float)i / samples;
		local += (int)(1000.0f *
			       (x * x + 2.0f * x + 1.0f)); // simple polynomial
	}

	atomicAdd(out, local);
}

extern "C" void launchComputeSeries(unsigned long long int *d_out, int samples)
{
	int threads = 256;
	int blocks = 256;
	computeSeriesKernel<<<blocks, threads>>>(d_out, samples);
	cudaDeviceSynchronize();
}

__global__ void outOfBoundsKernel(float *data, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	data[idx + n + 100] = 42.0f;
}

extern "C" __host__ void launchOutOfBoundsKernel(float *d_data, int n)
{
	printf("Launching out-of-bounds kernel (this should trigger SafeCUDA error)...\n");
	outOfBoundsKernel<<<1, 1>>>(d_data, n);
	cudaDeviceSynchronize();
	printf("Kernel completed (if you see this, bounds check may not have triggered)\n");
}
