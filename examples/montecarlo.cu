#include <curand_kernel.h>
#include <cuda_runtime.h>

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
