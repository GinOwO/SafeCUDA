#include <cuda_runtime.h>
#include <stdio.h>

extern "C" __device__ void __bounds_check_safecuda(void *);

__global__ void outOfBoundsKernel(float *data, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__bounds_check_safecuda(data + idx + n + 100);
	data[idx + n + 100] = 42.0f;
}

extern "C" __host__ void launchOutOfBoundsKernel(float *d_data, int n)
{
	printf("Launching out-of-bounds kernel (this should trigger SafeCUDA error)...\n");
	outOfBoundsKernel<<<1, 1>>>(d_data, n);
	cudaDeviceSynchronize();
	printf("Kernel completed (if you see this, bounds check may not have triggered)\n");
}
