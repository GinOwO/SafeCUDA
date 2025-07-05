#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Simple CUDA kernel that doubles each element
__global__ void test_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
	data[idx] *= 2.0f;
    }
}

// Host function to launch the kernel
extern "C" void launch_test_kernel(float* d_data, int size) {
    if (d_data == nullptr || size <= 0) {
	printf("Invalid parameters to launch_test_kernel\n");
	return;
    }
    
    // Calculate grid and block dimensions
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    // Launch kernel
    test_kernel<<<grid_size, block_size>>>(d_data, size);
    
    // Optional: Check for kernel launch errors immediately
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
	printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// Additional utility kernels for future SafeCUDA functionality

// Kernel to initialize memory with a pattern (useful for testing)
__global__ void init_memory_pattern(float* data, int size, float pattern) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
	data[idx] = pattern + static_cast<float>(idx);
    }
}

// Host wrapper for memory pattern initialization
extern "C" void safecuda_init_pattern(float* d_data, int size, float pattern) {
    if (d_data == nullptr || size <= 0) {
	printf("Invalid parameters to safecuda_init_pattern\n");
	return;
    }
    
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    init_memory_pattern<<<grid_size, block_size>>>(d_data, size, pattern);
}

// Bounds checking prototype (placeholder for future implementation)
__device__ bool check_bounds(void* ptr, size_t offset, size_t size) {
    // This is a placeholder - in the full implementation,
    // this would check against the shadow metadata table
    return true;
}

// Example of a protected memory access (future SafeCUDA feature)
__global__ void protected_access_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
	// Future: This would include bounds checking
	if (check_bounds(data, idx * sizeof(float), sizeof(float))) {
	    data[idx] += 1.0f;
	} else {
	    // Future: Signal bounds violation
	    printf("Bounds violation at index %d\n", idx);
	}
    }
}
