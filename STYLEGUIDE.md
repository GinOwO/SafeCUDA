# SafeCUDA Style Guide

This document defines the complete coding style and conventions for the project.
The main language used is C++ with CUDA extensions, and the style is inspired by
the Linux kernel coding style.

## Table of Contents

1. [TLDR](#tldr)
2. [Code Formatting](#code-formatting)
3. [Naming Conventions](#naming-conventions)
4. [Versioning](#versioning)
5. [File Organization](#file-organization)
6. [Documentation Standards](#documentation-standards)
7. [C++ Specific Guidelines](#c-specific-guidelines)
8. [CUDA Specific Guidelines](#cuda-specific-guidelines)
9. [Error Handling](#error-handling)
10. [Build Integration Style](#build-integration-style)
11. [Common Design Patterns](#common-design-patterns)

## TLDR:

- Always run `./scripts/format.sh` before committing code which applies
  formatting using `clang-format`.
- Follow LK style for C++ code which is basically:
    - Follow the K&R style.
    - Function braces should be on a new line.
    - Use 8-space tabs for indentation and a maximum line length of 80
      characters.
    - Use snake_case for variables and functions, PascalCase for classes and
      types.
    - Use UPPER_CASE for constants and macros.
- Use stdint.h types for fixed-width integers (e.g., `int32_t`, `uint64_t`).
- Use `nullptr` for null pointers.
- Use `const` and `constexpr` where applicable.
- Always use namespaces to avoid name collisions.
- Use Doxygen-style comments for documentation.
- Use RAII for resource management and exception safety.
- Read the documentation standards for more details on how to document code.

## Code Formatting

### Basic Formatting Rules

- **Indentation**: 8-space tabs only (`UseTab: Always`)
- **Line length**: Maximum 80 characters (`ColumnLimit: 80`)
- **Encoding**: UTF-8 with Unix line endings (LF)
- **Trailing whitespace**: Not allowed

### Braces and Brackets

```cpp
// Function braces: New line
void function_name()
{
        // Function body
}

// Control statement braces: Same line
if (condition) {
        // Body
} else {
        // Body
}

// Class/struct braces: Same line
class ClassName {
    private:
        int member_;
    public:
        void method();
};

// Namespace braces: New line
namespace safecuda
{
        // Namespace content
}
```

### Spacing and Alignment

```cpp
// Pointer alignment: Right-aligned
int *pointer;
float &reference;
const char *string;

// Function calls: No space before parentheses
function_call(arg1, arg2);
if (condition) {
        // Control statements: Space before parentheses
}

// Operators: Spaces around binary operators
int result = a + b * c;
bool flag = (x == y) && (z != w);

// Assignments: Space around assignment operators
variable = value;
ptr += offset;
```

### Line Breaking

Let clang-format handle this automatically, but here are some guidelines:

```cpp
// Long function declarations
bool very_long_function_name(const std::vector<float>& input_data,
                                                   MetadataEntry *metadata,
                                                   size_t buffer_size);

// Long function calls
result = some_function(first_parameter,
                                          second_parameter,
                                          third_parameter);

// Chain calls: Break after 80 character limit
auto result = object().short_m()
                      .very_long_method()
                      .short_m()
                      .very_long_method()
                      .short_m();
```

## Naming Conventions

### Variables and Functions

```cpp
// snake_case for variables and functions
int buffer_size;
float *device_memory;
bool is_initialized;
const int max_threads = 1024;

void initialize_cuda();
void check_bounds(void *ptr, size_t size);
bool validate_memory_access(void *ptr);
```

### Classes, Structs, and Types

```cpp
// PascalCase for user-defined types
class SafeCUDA {
        // ...
};

struct MetadataEntry {
        void *base_address;
        size_t size;
};

enum class ErrorCode {
        SUCCESS,
        INVALID_POINTER,
        OUT_OF_BOUNDS
};

using DevicePtr = void*;
using MetadataTable = std::unordered_map<void*, MetadataEntry>;
```

### Constants and Macros

```cpp
// UPPER_CASE for constants and macros
#define MAX_BUFFER_SIZE 1024
#define SAFECUDA_VERSION_MAJOR 1

const int DEFAULT_BLOCK_SIZE = 256;
const float EPSILON = 1e-6f;

// Preprocessor directives
#ifdef DEBUG
        #define LOG_LEVEL 3
#else
        #define LOG_LEVEL 1
#endif
```

### CUDA-Specific Naming

```cpp
// Kernel functions: snake_case with _kernel suffix
__global__ void bounds_check_kernel(float *data, int size);
__global__ void memory_init_kernel(void *ptr, size_t bytes);

// Device functions: snake_case (with optional _device suffix)
__device__ bool check_metadata_device(void *ptr);
__device__ inline float compute_distance(float x, float y);

// CUDA variables: descriptive prefixes
float *d_input;           // device memory
float *h_output;          // host memory
cudaStream_t stream;      // CUDA objects
dim3 grid_size;           // kernel launch parameters
dim3 block_size;
```

### File Naming

```cpp
// Header files
safecuda.h                // Main public header
safecuda_types.h          // Type definitions
safecuda_internal.cuh     // Internal CUDA declarations

// Source files
safecuda.cpp              // Host-side implementation
safecuda.cu               // CUDA kernel implementation
interception.cpp          // Memory allocation interception
test_bounds_checking.cpp  // Test files
```

## Versioning

SafeCUDA follows [Semantic Versioning 2.0.0](https://semver.org/).

### Version Format

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
```

**Examples:**

- `1.0.0` - Initial stable release
- `1.2.3` - Standard release
- `2.0.0-alpha.1` - Pre-release version
- `1.0.0+build.123` - Release with build metadata

### Version Components

#### MAJOR Version

Increment when making **incompatible API changes**:

```cpp
// Version 1.x.x
bool initialize_safecuda(int device_id);

// Version 2.0.0 - Breaking change (signature changed)
bool initialize_safecuda(const SafeCudaConfig& config);
```

**Examples of MAJOR changes:**

- Removing public functions or classes
- Changing function signatures
- Modifying public struct/class layouts
- Changing behavior that could break existing code
- Dropping support for CUDA versions

#### MINOR Version

Increment when adding **backward-compatible functionality**:

```cpp
// Version 1.0.0
class SafeCUDA {
public:
        bool initialize();
        void cleanup();
};

// Version 1.1.0 - New backward-compatible feature
class SafeCUDA {
public:
        bool initialize();
        void cleanup();
        
        // New function - existing code still works
        bool enable_async_checking(); 
};
```

**Examples of MINOR changes:**

- Adding new public functions
- Adding new classes or namespaces
- Adding optional parameters with defaults
- Adding new CUDA architecture support
- Performance improvements without API changes

#### PATCH Version

Increment for **backward-compatible bug fixes**:

```cpp
// Version 1.1.0 - Bug in bounds checking
__device__ bool validate_access(void *ptr, size_t size) {
        return ptr != nullptr && size > 0; // BUG: Missing actual bounds check
}

// Version 1.1.1 - Bug fix
__device__ bool validate_access(void *ptr, size_t size) {
        if (ptr == nullptr || size == 0) return false;
        
        // Proper bounds checking implementation
        return check_allocation_bounds(ptr, size);
}
```

**Examples of PATCH changes:**

- Bug fixes that don't change API
- Security patches
- Documentation corrections
- Internal refactoring
- Build system fixes

### Pre-release Versions

Use pre-release identifiers for unstable versions:

```
1.2.0-alpha.1    # Early alpha
1.2.0-beta.2     # Beta release
1.2.0-rc.1       # Release candidate
```

**Pre-release naming:**

- `alpha` - Early development, unstable API
- `beta` - Feature complete, API stable, testing phase
- `rc` (release candidate) - Final testing before release

### Build Metadata

Append build information after `+`:

```
1.0.0+build.123
1.2.0-beta.1+git.abcd123
```

**Build metadata does not affect version precedence.**

### Version Precedence

```
1.0.0-alpha.1 20% or algorithmic changes
- **MINOR**: Performance improvements or new optimization features
- **PATCH**: Minor performance fixes

### Version Management in Code

#### CMakeLists.txt
```cmake
project(SafeCUDA 
    VERSION 1.2.3
    LANGUAGES CXX CUDA
)

# Generate version define
target_compile_definitions(safecuda PRIVATE
        PROJECT_VERSION_MAJOR="${PROJECT_VERSION_MAJOR}"
        PROJECT_VERSION_MINOR="${PROJECT_VERSION_MINOR}"
        PROJECT_VERSION_PATCH="${PROJECT_VERSION_PATCH}"
        PROJECT_VERSION="${PROJECT_VERSION}"
)
```

### Release Process

#### Git Tagging

```bash
# Create annotated tag for release
git tag -a v1.2.3 -m "SafeCUDA v1.2.3: Bug fixes and performance improvements"

# Push tags
git push origin v1.2.3
```

#### Documentation Updates

Update version in all documentation files:

- README.md
- API documentation
- Installation guides
- Changelog/Release notes

#### Version Validation

```cpp
// Compile-time assertions for version consistency
static_assert(SAFECUDA_VERSION_MAJOR >= 1, "Major version must be >= 1");
static_assert(SAFECUDA_VERSION_STRING[0] != '\0', "Version string cannot be empty");
```

### Examples

#### Breaking Change (MAJOR)

```cpp
// v1.x.x
bool initialize_safecuda(int device_id = 0);

// v2.0.0 - API breaking change
struct SafeCudaConfig {
        int device_id = 0;
        bool enable_async = false;
        size_t cache_size = 1024;
};

bool initialize_safecuda(const SafeCudaConfig& config);
```

#### New Feature (MINOR)

```cpp
// v1.0.0
namespace safecuda {
        bool initialize();
        void cleanup();
}

// v1.1.0 - New feature
namespace safecuda {
        bool initialize();
        void cleanup();
        
        // New backward-compatible feature
        class MemoryTracker {
            public:
                size_t get_allocated_bytes() const;
                size_t get_peak_usage() const;
        };
}
```

#### Bug Fix (PATCH)

```cpp
// v1.1.0 - Has bug
__device__ bool check_bounds(void* ptr, size_t size) {
        return size > 0; // BUG: Not checking ptr validity
}

// v1.1.1 - Bug fixed
__device__ bool check_bounds(void* ptr, size_t size) {
        return ptr != nullptr && size > 0;
}
```

## File Organization

### Header File Structure

```cpp
/**
 * @file safecuda.h
 * @brief Software-only memory safety toolkit for CUDA applications
 * 
 * This header defines the core SafeCUDA API for providing spatial memory
 * safety on commodity NVIDIA GPUs without hardware modifications. The
 * implementation uses PTX instrumentation and managed shadow metadata
 * to detect out-of-bounds memory accesses in real-time.
 * 
 * @author Name <example@example.com>
 * @date 2025-01-15
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 * 
 * Change Log:
 * - 2025-01-15: Initial implementation
 */

#ifndef SAFECUDA_H
#define SAFECUDA_H

// Standard C++ includes (alphabetical order)
#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>

// System includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Project includes
#include "safecuda_types.h"

// Forward declarations
class MetadataManager;
struct AllocationInfo;

// Type aliases
using ErrorCallback = std::function<void(const char*)>;

// Constants
extern const int SAFECUDA_VERSION_MAJOR;
extern const int SAFECUDA_VERSION_MINOR;

// Function declarations
namespace safecuda
{
        bool initialize();
        void cleanup();
        
        // Class declarations
        class SafeCUDA {
                // ...
        };
        
} // namespace safecuda

#endif // SAFECUDA_H
```

### Source File Structure

```cpp
/**
 * @file safecuda.cpp
 * @brief Implementation of SafeCUDA core functionality
 * 
 * Contains the host-side implementation of the SafeCUDA memory safety
 * system, including CUDA allocation interception, metadata management,
 * and exception handling mechanisms.
 * 
 * @author Name <example@example.com>
 * @date 2025-01-20
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 * 
 * Change Log:
 * - 2025-01-20: Initial implementation
 */

#include "safecuda.h"

// Additional includes
#include <cassert>
#include <dlfcn.h>

// Anonymous namespace for internal functions
namespace
{
        // File-scope constants
        constexpr size_t METADATA_CACHE_SIZE = 1024;
        
        // Static variables
        static bool g_initialized = false;
        static MetadataTable *g_shadow_table = nullptr;
        
        // Internal helper functions
        static bool validate_cuda_context();
        static void cleanup_metadata_table();
        
} // anonymous namespace

// Public function implementations
namespace safecuda
{
        bool initialize()
        {
                // Implementation...
        }
        
} // namespace safecuda

// Static function implementations
namespace
{
        static bool validate_cuda_context()
        {
                // Implementation...
        }
        
} // anonymous namespace
```

## Documentation Standards

### Doxygen Style Requirements

SafeCUDA uses **Doxygen-style comments** exclusively. All public APIs, classes,
and non-trivial functions must be documented.

### File-Level Documentation

```cpp
/**
 * @file filename.h
 * @brief One-line description of file purpose
 * 
 * Detailed explanation of what this file contains, its role in the
 * system architecture, and any important implementation details or
 * design decisions that affect the entire file.
 * 
 * @author Primary Author <email@university.edu>
 * @author Secondary Author <email@university.edu>
 * @date YYYY-MM-DD
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 * 
 * Change Log:
 * - YYYY-MM-DD: Initial implementation
 */
```

### Class Documentation

```cpp
/**
 * @class ClassName
 * @brief One-line description of class purpose
 * 
 * Detailed explanation of the class's role, design rationale, and
 * important usage patterns. Focus on WHY the class exists and HOW
 * it fits into the overall system architecture.
 * 
 * Thread safety, performance characteristics, and lifecycle
 * management should be documented here.
 */
class ClassName {
        // ...
};
```

### Function Documentation

**Focus on WHY and HOW, not WHAT the function does:**

```cpp
/**
 * @brief One-line summary focusing on implementation approach
 * 
 * Detailed explanation of WHY this implementation approach was chosen
 * and HOW it works internally. Discuss performance considerations,
 * trade-offs, and any non-obvious implementation details.
 * 
 * Avoid simply restating what the function name already tells us.
 * Instead, explain the reasoning behind the implementation strategy.
 * 
 * @param param_name Description of parameter purpose and constraints
 * @param output_param Description of output parameter (use for out params)
 * @return Description of return value and possible error conditions
 * @throws ExceptionType When and why this exception might be thrown
 * @note Important implementation notes or performance implications
 * @warning Critical usage warnings or thread safety considerations
 * @author Function Author <email@university.edu>
 */
ReturnType function_name(ParamType param_name, OutputType *output_param);
```

### Documentation Examples

#### Poor Documentation (Describes WHAT)

```cpp
/**
 * @brief Checks if pointer is valid
 * @param ptr Pointer to check
 * @return true if valid, false otherwise
 */
bool is_valid_pointer(void *ptr);
```

#### Good Documentation (Describes WHY/HOW)

```cpp
/**
 * @brief Validates pointer against shadow metadata using cached lookup strategy
 * 
 * We perform validation by consulting the thread-local metadata cache first,
 * falling back to global memory only on cache misses. This approach minimizes
 * memory bandwidth usage during the common case where kernels access the same
 * allocations repeatedly within a thread block.
 * 
 * The validation uses a two-tier lookup: first checking if the pointer falls
 * within any known allocation range, then verifying the specific access offset
 * doesn't exceed the allocation boundaries.
 * 
 * @param ptr Memory address to validate against tracked allocations
 * @return true if pointer maps to a tracked allocation, false otherwise
 * @note Cache hit rate typically >90% for well-behaved kernels
 * @author Name <example@example.com>
 */
bool is_valid_pointer(void *ptr);
```

## C++ Specific Guidelines

### Class Design Patterns

```cpp
/**
 * @class SafeCUDA
 * @brief Main singleton interface for GPU memory safety operations
 * 
 * Designed as a singleton to maintain global state for memory tracking
 * across the entire application. Uses RAII principles for automatic
 * cleanup and provides exception-safe initialization.
 */
class SafeCUDA {
    private:
        // Member variables: underscore suffix
        bool initialized_;
        int device_count_;
        std::unique_ptr<MetadataTable> shadow_table_;
        
        // Private helper methods
        bool setup_cuda_context();
        void cleanup_internal();

    public:
        // Constructor/Destructor
        SafeCUDA();
        ~SafeCUDA();
        
        // Delete copy operations for singleton behavior
        SafeCUDA(const SafeCUDA&) = delete;
        SafeCUDA& operator=(const SafeCUDA&) = delete;
        
        // Move operations allowed
        SafeCUDA(SafeCUDA&&) = default;
        SafeCUDA& operator=(SafeCUDA&&) = default;
        
        // Public interface
        bool initialize();
        void cleanup();
        
        // Const getters
        bool is_initialized() const noexcept { return initialized_; }
        int device_count() const noexcept { return device_count_; }
};
```

### Memory Management

```cpp
// RAII wrapper for CUDA memory
class CudaMemory {
    private:
        void *ptr_;
        size_t size_;

    public:
        explicit CudaMemory(size_t bytes) : ptr_(nullptr), size_(bytes)
        {
                cudaError_t err = cudaMalloc(&ptr_, bytes);
                if (err != cudaSuccess) {
                        throw std::runtime_error("CUDA allocation failed");
                }
        }
        
        ~CudaMemory()
        {
                if (ptr_) {
                        cudaFree(ptr_);  // Note: cudaFree handles nullptr gracefully
                }
        }
        
        // Non-copyable
        CudaMemory(const CudaMemory&) = delete;
        CudaMemory& operator=(const CudaMemory&) = delete;
        
        // Movable
        CudaMemory(CudaMemory&& other) noexcept
                : ptr_(other.ptr_), size_(other.size_)
        {
                other.ptr_ = nullptr;
                other.size_ = 0;
        }
        
        // Accessors
        void* get() const noexcept { return ptr_; }
        size_t size() const noexcept { return size_; }
};
```

### Function Parameters

```cpp
// Pass by const reference for complex types
void process_data(const std::vector<float>& input_data,
                  const MetadataEntry& metadata);

// Pass by value for primitives and small objects
void set_dimensions(int width, int height, float scale);

// Output parameters: use pointers, not references
bool get_device_info(int device_id, cudaDeviceProp *prop);
void compute_result(const float *input, float *output, size_t count);

// Optional parameters: use std::optional or default values
void configure_kernel(int block_size = 256,
                      std::optional<cudaStream_t> stream = std::nullopt);
```

### Exception Safety

```cpp
/**
 * @brief Initializes SafeCUDA with strong exception safety guarantee
 * 
 * Uses RAII and careful ordering to ensure that if initialization fails
 * at any point, all previously allocated resources are properly cleaned
 * up. The function either succeeds completely or leaves the object in
 * its original state.
 * 
 * @throws std::runtime_error if CUDA context creation fails
 * @throws std::bad_alloc if metadata table allocation fails
 * @author System Architect <architect@university.edu>
 */
void SafeCUDA::initialize()
{
        // Use local variables for exception safety
        std::unique_ptr<MetadataTable> temp_table;
        int temp_device_count = 0;
        
        try {
                // Allocate resources
                temp_table = std::make_unique<MetadataTable>();
                
                // Setup CUDA context
                cudaError_t err = cudaGetDeviceCount(&temp_device_count);
                if (err != cudaSuccess) {
                        throw std::runtime_error("Failed to get CUDA device count");
                }
                
                // Only modify member variables after all operations succeed
                shadow_table_ = std::move(temp_table);
                device_count_ = temp_device_count;
                initialized_ = true;
                
        } catch (...) {
                // Cleanup is automatic due to RAII
                throw;  // Re-throw original exception
        }
}
```

## CUDA Specific Guidelines

### Kernel Launch Patterns

```cpp
/**
 * @brief Launches bounds checking kernel with optimal occupancy
 * 
 * We calculate grid and block dimensions to maximize GPU occupancy while
 * respecting shared memory constraints for metadata caching. The block
 * size is chosen to ensure each block can cache its required metadata
 * entries in shared memory without exceeding hardware limits.
 * 
 * @param data Device memory pointer to validate
 * @param size Number of elements in the data array
 * @param stream CUDA stream for asynchronous execution
 * @author Performance Engineer <perf@university.edu>
 */
void launch_bounds_check(float *data, int size, cudaStream_t stream)
{
        // Calculate optimal block size based on shared memory requirements
        int block_size = 256;  // Optimal for most architectures
        int grid_size = (size + block_size - 1) / block_size;
        
        // Ensure we don't exceed maximum grid dimensions
        int max_grid_size = 65535;  // Common limit across architectures
        if (grid_size > max_grid_size) {
                // Handle large arrays by processing in chunks
                throw std::runtime_error("Array too large for single kernel launch");
        }
        
        // Launch kernel
        bounds_check_kernel<<<grid_size, block_size, 0, stream>>>(data, size);
        
        // Always check for launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
                throw std::runtime_error(std::string("Kernel launch failed: ") + 
                                   cudaGetErrorString(err));
        }
}
```

### Device Code Style

```cpp
/**
 * @brief GPU kernel for validating memory accesses with shared memory caching
 * 
 * This kernel design prioritizes shared memory utilization over global memory
 * bandwidth because GPU architectures provide much higher throughput for
 * shared memory operations. We preload relevant metadata entries into shared
 * memory at kernel launch time to amortize the global memory access cost.
 * 
 * @param data Device pointer to memory buffer being validated
 * @param size Number of elements in the buffer
 * @author Name <example@example.com>
 */
__global__ void bounds_check_kernel(float *data, int size)
{
        // Shared memory for metadata caching
        __shared__ MetadataEntry cache[MAX_CACHE_ENTRIES];
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int tid = threadIdx.x;
        
        // Cooperative loading of metadata into shared memory
        if (tid < MAX_CACHE_ENTRIES) {
                cache[tid] = load_metadata_entry(blockIdx.x, tid);
        }
        __syncthreads();
        
        // Bounds check with early exit
        if (idx >= size) {
                return;
        }
        
        // Perform bounds validation using cached metadata
        bool is_valid = validate_access(&data[idx], cache, MAX_CACHE_ENTRIES);
        if (!is_valid) {
                // Signal violation - details depend on error handling strategy
                signal_bounds_violation(idx, &data[idx]);
        }
        
        // Continue with actual computation
        data[idx] *= 2.0f;
}

/**
 * @brief Device function for efficient bounds validation
 * 
 * Implements bounds checking using binary search through cached metadata
 * rather than hash table lookup to avoid divergent branching that would
 * occur with hash collisions. Binary search provides more predictable
 * performance across different allocation patterns.
 * 
 * @param ptr Memory address being checked
 * @param cache Shared memory cache of allocation metadata
 * @param cache_size Number of valid entries in cache
 * @return true if access is within bounds, false otherwise
 * @author Name <example@example.com>
 */
__device__ bool validate_access(void *ptr, const MetadataEntry *cache, int cache_size)
{
        // Binary search through cached metadata
        int left = 0;
        int right = cache_size - 1;
        
        while (left <= right) {
                int mid = (left + right) / 2;
                const MetadataEntry& entry = cache[mid];
                
                if (ptr >= entry.base_address && 
                        ptr < (char*)entry.base_address + entry.size) {
                        return true;  // Found valid allocation
                }
                
                if (ptr < entry.base_address) {
                        right = mid - 1;
                } else {
                        left = mid + 1;
                }
        }
        
        return false;  // Not found in cache
}
```

### CUDA Error Handling

```cpp
/**
 * @brief RAII wrapper for CUDA error checking
 * 
 * Provides exception-based error handling for CUDA operations while
 * maintaining compatibility with existing C-style error code patterns.
 * The wrapper automatically checks return codes and converts them to
 * exceptions with descriptive error messages.
 */
class CudaErrorChecker {
    public:
        static void check(cudaError_t error, const char *operation)
        {
                if (error != cudaSuccess) {
                        std::string message =
                                std::string(operation) +
                                " failed: " + cudaGetErrorString(error);
                        throw std::runtime_error(message);
                }
        }
};

// Usage macro for concise error checking
#define CUDA_CHECK(call) CudaErrorChecker::check(call, #call)

// Example usage
void allocate_device_memory(void **ptr, size_t bytes)
{
        CUDA_CHECK(cudaMalloc(ptr, bytes));
        CUDA_CHECK(cudaMemset(*ptr, 0, bytes));
}
```

## Error Handling

### Error Code Patterns

```cpp
/**
 * @enum SafeCudaError
 * @brief Comprehensive error codes for SafeCUDA operations
 * 
 * Designed to provide detailed diagnostic information while maintaining
 * compatibility with both exception-based and return-code-based error
 * handling patterns.
 */
enum class SafeCudaError {
        SUCCESS = 0,
        INVALID_POINTER,
        OUT_OF_BOUNDS,
        CUDA_ERROR,
        INITIALIZATION_FAILED,
        DEVICE_NOT_SUPPORTED
};

/**
 * @brief Converts error codes to human-readable messages
 * 
 * Centralized error message formatting ensures consistent error reporting
 * across the entire SafeCUDA system. Messages include both the immediate
 * cause and suggested remediation steps where appropriate.
 * 
 * @param error Error code to convert
 * @return Descriptive error message string
 * @author Name <example@example.com>
 */
const char* get_error_string(SafeCudaError error)
{
        switch (error) {
        case SafeCudaError::SUCCESS:
                return "Operation completed successfully";
        case SafeCudaError::INVALID_POINTER:
                return "Invalid pointer: address not found in allocation table";
        case SafeCudaError::OUT_OF_BOUNDS:
                return "Out of bounds access: offset exceeds allocation size";
        case SafeCudaError::CUDA_ERROR:
                return "CUDA runtime error: check device status and drivers";
        default:
                return "Unknown error code";
        }
}
```

### Exception Handling Strategy

```cpp
/**
 * @class SafeCudaException
 * @brief Custom exception class for SafeCUDA-specific errors
 * 
 * Provides structured exception information including error codes,
 * contextual information, and suggested recovery actions. Designed
 * to integrate with existing C++ exception handling mechanisms.
 */
class SafeCudaException : public std::runtime_error {
    private:
        SafeCudaError error_code_;
        std::string context_;

    public:
        SafeCudaException(SafeCudaError code, const std::string& context)
                : std::runtime_error(get_error_string(code))
                , error_code_(code)
                , context_(context)
        {
        }
        
        SafeCudaError error_code() const noexcept { return error_code_; }
        const std::string& context() const noexcept { return context_; }
};
```

## Build Integration Style

### CMake Style

```cmake
# Use lowercase for commands and consistent indentation (4 spaces)
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project configuration
project(SafeCUDA
        VERSION 1.0.0
        LANGUAGES CXX CUDA
        DESCRIPTION "Software-only memory safety for CUDA applications"
)

# Modern C++ standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Clear variable naming with descriptive comments
set(CUDA_ARCH_LIST 60 61 75 86)  # Support Maxwell through Ampere
set(SAFECUDA_VERSION_MAJOR 1)
set(SAFECUDA_VERSION_MINOR 0)

# Organized target definitions
add_library(safecuda SHARED
        # Host-side implementation
        src/safecuda.cpp
        src/interception.cpp
        src/metadata_manager.cpp

        # Device-side implementation
        src/safecuda.cu
        src/bounds_checking.cu
)

# Clear dependency specification
target_link_libraries(safecuda
        PUBLIC
        CUDA::cudart
        PRIVATE
        CUDA::cuda_driver
        ${CMAKE_DL_LIBS}
)
```

### Shell Script Style

```bash
#!/bin/bash
# SafeCUDA build script
# Usage: ./scripts/build.sh [Debug|Release] [clean]

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Configuration with clear defaults
BUILD_TYPE=${1:-Debug}
BUILD_DIR="build"
CLEAN_BUILD=${2:-""}

# Input validation with descriptive error messages
if [[ "$BUILD_TYPE" != "Debug" && "$BUILD_TYPE" != "Release" ]]; then
        echo "Error: Build type must be 'Debug' or 'Release'"
        echo "Usage: $0 [Debug|Release] [clean]"
        exit 1
fi

# Clear status reporting
echo "Building SafeCUDA in $BUILD_TYPE mode..."

# Clean build if requested
if [[ "$CLEAN_BUILD" == "clean" ]]; then
        echo "Cleaning previous build..."
        rm -rf "$BUILD_DIR"
fi

# Build process with error checking
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -G Ninja \
          -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          ..

ninja

echo "Build complete! Binaries are in $BUILD_DIR/"
echo "Run tests with ctest in the build directory."
```

## Common Design Patterns

### Initialization Pattern

```cpp
/**
 * @class Component
 * @brief Base pattern for SafeCUDA components requiring initialization
 * 
 * Provides a consistent initialization pattern across all SafeCUDA components.
 * Uses a two-phase construction approach: constructor sets up basic state,
 * initialize() performs resource allocation and validation.
 */
class Component {
    private:
        bool initialized_;
        std::string last_error_;

    protected:
        // Virtual methods for derived classes
        virtual bool setup_resources() = 0;
        virtual void cleanup_resources() = 0;

    public:
        Component() : initialized_(false) {}
        
        virtual ~Component()
        {
                cleanup();
        }
        
        /**
         * @brief Two-phase initialization with exception safety
         * 
         * Uses RAII principles to ensure that failed initialization doesn't
         * leave the object in an inconsistent state. All resource allocation
         * is performed through RAII wrappers to guarantee cleanup on exceptions.
         * 
         * @return true if initialization successful, false otherwise
         * @author Name <example@example.com>
         */
        bool initialize()
        {
                if (initialized_) {
                        return true;  // Already initialized
                }
                
                try {
                        if (!setup_resources()) {
                                last_error_ = "Resource setup failed";
                                return false;
                        }
                        
                        initialized_ = true;
                        return true;
                        
                } catch (const std::exception& e) {
                        last_error_ = e.what();
                        return false;
                }
        }
        
        void cleanup()
        {
                if (initialized_) {
                        cleanup_resources();
                        initialized_ = false;
                }
        }
        
        // Status accessors
        bool is_initialized() const noexcept { return initialized_; }
        const std::string& last_error() const noexcept { return last_error_; }
};
```

### Resource Management Pattern

```cpp
/**
 * @class ResourceManager
 * @brief RAII-based management of GPU resources
 * 
 * Centralizes GPU resource management to prevent leaks and ensure proper
 * cleanup order. Uses smart pointers and custom deleters to integrate
 * CUDA resource management with standard C++ RAII patterns.
 */
template<typename T>
class ResourceManager {
    private:
        std::unique_ptr<T, std::function<void(T*)>> resource_;

    public:
        template<typename Deleter>
        ResourceManager(T* resource, Deleter deleter)
                : resource_(resource, deleter)
        {
        }
        
        T* get() const noexcept { return resource_.get(); }
        T* release() noexcept { return resource_.release(); }
        
        explicit operator bool() const noexcept { return resource_ != nullptr; }
};

// Factory functions for common CUDA resources
auto make_cuda_memory(size_t bytes)
{
        void *ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
                throw SafeCudaException(SafeCudaError::CUDA_ERROR, "Memory allocation failed");
        }
        
        return ResourceManager<void>(ptr, [](void* p) { cudaFree(p); });
}

auto make_cuda_stream()
{
        cudaStream_t stream;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
                throw SafeCudaException(SafeCudaError::CUDA_ERROR, "Stream creation failed");
        }
        
        return ResourceManager<cudaStream_t>(
                new cudaStream_t(stream),
                [](cudaStream_t* s) { 
                        cudaStreamDestroy(*s); 
                        delete s; 
                }
        );
}
```
