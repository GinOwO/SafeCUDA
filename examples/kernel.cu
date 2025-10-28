#include <cuda_runtime.h>
#include <stdio.h>

#include <curand_kernel.h>
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

__global__ void monteCarloKernel(int *count, int samples, unsigned long seed)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	curandState state;
	curand_init(seed, idx, 0, &state);

	int local_count = 0;
	for (int i = idx; i < samples; i += stride) {
		float x = curand_uniform(&state);
		float y = curand_uniform(&state);
		if (x * x + y * y <= 1.0f)
			local_count++;
	}

	atomicAdd(count, local_count);
}

extern "C" void monteCarloPi(int *d_count, int samples, int blocks, int threads,
			     unsigned long seed)
{
	monteCarloKernel<<<blocks, threads>>>(d_count, samples, seed);
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
