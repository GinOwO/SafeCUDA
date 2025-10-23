/**
* @file test_safecuda_runtime.cuh
 * @brief Device-side kernel declarations for SafeCUDA runtime tests
 *
 * Contains CUDA kernel and device function declarations for testing
 * bounds checking behavior with valid, invalid, and edge-case
 * memory access patterns.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-10-23
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-10-23: Initial implementation
 */

#ifndef TEST_SAFECUDA_RUNTIME_CUH
#define TEST_SAFECUDA_RUNTIME_CUH

namespace safecuda_runtime_tests
{

extern "C" void launch_valid_access_kernel(float *d_data, int size);
extern "C" void launch_out_of_bounds_kernel(float *d_data, int size);
extern "C" void launch_freed_memory_kernel(float *d_data, int size);
extern "C" void launch_interior_pointer_kernel(float *d_data, int size,
					       int offset);
extern "C" void launch_manual_bounds_check(float *d_data, int size,
					   int should_fail);

}

#endif // TEST_SAFECUDA_RUNTIME_CUH
