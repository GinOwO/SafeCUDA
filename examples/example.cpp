#include <iostream>
#include <vector>
#include <cuda_runtime.h>

extern "C" void launchScaleArray(float *d_data, int n);
extern "C" void launchAddOne(float *d_data, int n);
extern "C" void monteCarloPi(int *d_count, int samples, int blocks, int threads,
			     unsigned long seed);

int main()
{
	constexpr int n = 10;
	std::vector<float> h_data(n);
	for (int i = 0; i < n; i++)
		h_data[i] = i;

	float *d_data;
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

	std::cout << "\n\nMonteCarloPi\n";

	int *d_count;
	cudaMalloc(&d_count, sizeof(int));
	cudaMemset(d_count, 0, sizeof(int));

	monteCarloPi(d_count, 1000000, 256, 256, time(nullptr));

	int h_count;
	cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_count);

	double pi = 4.0 * h_count / 1000000.0f;
	std::cout << "Estimated π = " << pi << std::endl;
}
