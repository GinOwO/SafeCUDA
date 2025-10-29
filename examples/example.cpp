#include <cuda_runtime.h>
#include <iostream>
#include <vector>

extern "C" void launchScaleArray(float *d_data, int n);
extern "C" void launchAddOne(float *d_data, int n);
extern "C" void monteCarloPi(int *d_count, int samples, int blocks, int threads,
			     unsigned long seed);
extern "C" void launchOutOfBoundsKernel(float *d_data, int n);

int main()
{
	std::cout << "\n=== Test 1: Valid access (should work) ==="
		  << std::endl;
	constexpr int n = 10;
	std::vector<float> h_data(n);
	for (int i = 0; i < n; i++)
		h_data[i] = i;
	float *d_data;
	std::cout << "\nLaunch ScaleArray > AddOne > ScaleArray\n\n";
	cudaMalloc(&d_data, n * sizeof(float));
	cudaMemcpy(d_data, h_data.data(), n * sizeof(float),
		   cudaMemcpyHostToDevice);
	launchScaleArray(d_data, n);
	launchAddOne(d_data, n);
	launchScaleArray(d_data, n);
	cudaMemcpy(h_data.data(), d_data, n * sizeof(float),
		   cudaMemcpyDeviceToHost);
	cudaFree(d_data);
	for (auto val : h_data)
		std::cout << val << " ";
	std::cout << "\n\n";
	// std::cout << "\n\nMonteCarloPi\n";
	int *d_count;
	cudaMalloc(&d_count, sizeof(int));
	cudaMemset(d_count, 0, sizeof(int));
	// monteCarloPi(d_count, 1000000, 256, 256, time(nullptr));
	int h_count = 0;
	cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_count);

	// double pi = 4.0 * h_count / 1000000.0f;
	// std::cout << "Estimated π = " << pi << std::endl;

	std::cout << "Valid access completed successfully!" << std::endl;

	std::cout
		<< "\n=== Test 2: Out-of-bounds access (should trigger error) ==="
		<< std::endl;
	constexpr int m = 1024;
	constexpr size_t bytes = m * sizeof(float);

	auto *h_data_err = new float[m];
	for (int i = 0; i < m; ++i) {
		h_data_err[i] = static_cast<float>(i);
	}

	float *d_data_err;
	cudaMalloc(&d_data_err, bytes);
	cudaMemcpy(d_data_err, h_data_err, bytes, cudaMemcpyHostToDevice);

	launchOutOfBoundsKernel(d_data_err, m);
	std::cout << "If you see this, the bounds check didn't work!"
		  << std::endl;

	cudaFree(d_data_err);
	delete[] h_data_err;

	return 0;
}
