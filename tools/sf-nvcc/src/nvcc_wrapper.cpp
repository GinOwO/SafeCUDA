/**
 * @file nvcc_wrapper.cpp
 * @brief Implementation of NVCC compiler wrapper
 *
 * Coordinates the PTX modification pipeline by intercepting NVCC calls,
 * generating PTX, modifying it for bounds checking, and resuming compilation.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-12
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-08-12: Initial File
 */

#include "nvcc_wrapper.h"
