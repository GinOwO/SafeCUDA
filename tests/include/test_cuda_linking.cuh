/**
 * @file test_cuda_linking.cuh
 * @brief CUDA device function declarations for linking tests
 * 
 * Contains CUDA kernel and device function declarations for testing
 * GPU compilation and execution pipeline.
 * 
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-07-06
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 * 
 * Change Log:
 * - 2025-07-06: Initial implementation
 */

#ifndef TEST_CUDA_LINKING_CUH
#define TEST_CUDA_LINKING_CUH

namespace cuda_linking_tests
{
extern "C" void launch_test_kernel(float *d_data, int size);
extern "C" void launch_memory_pattern_kernel(float *d_data, int size,
					     float pattern);
extern "C" void launch_advanced_test_kernel(float *d_data, int size);

} // namespace cuda_linking_tests

#endif // TEST_CUDA_LINKING_CUH
