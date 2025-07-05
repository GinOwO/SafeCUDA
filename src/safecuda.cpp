#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <memory>

// Forward declaration of CUDA kernel
extern "C" void launch_test_kernel(float *d_data, int size);

class SafeCUDA {
    private:
	bool initialized;
	int device_count;

    public:
	SafeCUDA()
		: initialized(false)
		, device_count(0)
	{
	}

	bool initialize()
	{
		cudaError_t err = cudaGetDeviceCount(&device_count);
		if (err != cudaSuccess) {
			std::cerr << "CUDA Error: " << cudaGetErrorString(err)
				  << std::endl;
			return false;
		}

		if (device_count == 0) {
			std::cerr << "No CUDA devices found!" << std::endl;
			return false;
		}

		// Print device information
		for (int i = 0; i < device_count; ++i) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			std::cout << "Device " << i << ": " << prop.name
				  << " (Compute " << prop.major << "."
				  << prop.minor << ")" << std::endl;
		}

		initialized = true;
		return true;
	}

	bool test_memory_operations()
	{
		if (!initialized) {
			std::cerr << "SafeCUDA not initialized!" << std::endl;
			return false;
		}

		const int size = 1024;
		const size_t bytes = size * sizeof(float);

		// Allocate host memory
		std::vector<float> h_data(size);
		for (int i = 0; i < size; ++i) {
			h_data[i] = static_cast<float>(i);
		}

		// Allocate device memory
		float *d_data = nullptr;
		cudaError_t err = cudaMalloc(&d_data, bytes);
		if (err != cudaSuccess) {
			std::cerr << "cudaMalloc failed: "
				  << cudaGetErrorString(err) << std::endl;
			return false;
		}

		// Copy host to device
		err = cudaMemcpy(d_data, h_data.data(), bytes,
				 cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "cudaMemcpy H2D failed: "
				  << cudaGetErrorString(err) << std::endl;
			cudaFree(d_data);
			return false;
		}

		// Launch kernel
		launch_test_kernel(d_data, size);

		// Check for kernel launch errors
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "Kernel launch failed: "
				  << cudaGetErrorString(err) << std::endl;
			cudaFree(d_data);
			return false;
		}

		// Wait for kernel to complete
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "cudaDeviceSynchronize failed: "
				  << cudaGetErrorString(err) << std::endl;
			cudaFree(d_data);
			return false;
		}

		// Copy device to host
		std::vector<float> h_result(size);
		err = cudaMemcpy(h_result.data(), d_data, bytes,
				 cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			std::cerr << "cudaMemcpy D2H failed: "
				  << cudaGetErrorString(err) << std::endl;
			cudaFree(d_data);
			return false;
		}

		// Verify results (kernel should have doubled each value)
		bool success = true;
		for (int i = 0; i < size; ++i) {
			float expected = 2.0f * static_cast<float>(i);
			if (std::abs(h_result[i] - expected) > 1e-6f) {
				std::cerr << "Mismatch at index " << i
					  << ": expected " << expected
					  << ", got " << h_result[i]
					  << std::endl;
				success = false;
				break;
			}
		}

		// Cleanup
		cudaFree(d_data);

		if (success) {
			std::cout << "✓ CUDA memory operations test passed!"
				  << std::endl;
		}

		return success;
	}

	void cleanup()
	{
		if (initialized) {
			cudaDeviceReset();
			initialized = false;
		}
	}

	~SafeCUDA()
	{
		cleanup();
	}
};

// Test function for library linkage
extern "C" int test_safecuda_linking()
{
	std::cout << "=== SafeCUDA Linking Test ===" << std::endl;

	SafeCUDA safecuda;

	if (!safecuda.initialize()) {
		std::cerr << "Failed to initialize SafeCUDA" << std::endl;
		return -1;
	}

	if (!safecuda.test_memory_operations()) {
		std::cerr << "Memory operations test failed" << std::endl;
		return -1;
	}

	std::cout << "✓ All tests passed! CUDA linking is working correctly."
		  << std::endl;
	return 0;
}

// Main function for standalone testing
int main()
{
	return test_safecuda_linking();
}
