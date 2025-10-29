#include <cuda_runtime.h>
#include <vector>

extern "C" void launchMemHammer(float *buffer, int n);

// 6. purely synthetic test table = 1000 and only modifies the last item but
// smaller no of ops in the kernel with ~1gb of traffic

int main()
{
	const int count = 1000;
	const int n = 1024 * 128;

	std::vector<float *> d_bufs(count);

	for (int i = 0; i < count; ++i) {
		cudaMalloc(&d_bufs[i], n * sizeof(float));
		cudaMemset(d_bufs[i], 0, n * sizeof(float));
	}

	for (int i = 0; i < count; i += 10) {
		cudaFree(d_bufs[i]);
		d_bufs[i] = nullptr;
	}

	float *lastValid = nullptr;
	for (int i = count - 1; i >= 0; --i) {
		if (d_bufs[i]) {
			lastValid = d_bufs[i];
			break;
		}
	}

	launchMemHammer(lastValid, n);
	cudaDeviceSynchronize();

	for (int i = 0; i < count; ++i)
		if (d_bufs[i])
			cudaFree(d_bufs[i]);

	return 0;
}
