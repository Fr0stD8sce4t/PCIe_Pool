#include "turbobus/dummy_compute.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace turbobus {

namespace {

void CheckCuda(cudaError_t result, const char* message) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " +
                             cudaGetErrorString(result));
  }
}

void IgnoreCuda(cudaError_t result) {
  (void)result;
}

__global__ void DummyComputeKernel(float* data, std::size_t elements, int iterations) {
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += stride) {
    float value = data[index];
    for (int iteration = 0; iteration < iterations; ++iteration) {
      value = value * 1.000001f + 0.000001f;
      value = value - 0.0000005f;
    }
    data[index] = value;
  }
}

}  // namespace

DummyComputeStats RunDummyCompute(int device, void* device_ptr, std::size_t elements,
                                  int iterations) {
  if (elements == 0) {
    throw std::runtime_error("dummy compute elements must be positive");
  }
  if (iterations <= 0) {
    throw std::runtime_error("dummy compute iterations must be positive");
  }
  if (device_ptr == nullptr) {
    throw std::runtime_error("dummy compute device pointer must not be null");
  }

  int previous_device = 0;
  CheckCuda(cudaGetDevice(&previous_device), "cudaGetDevice dummy compute failed");
  CheckCuda(cudaSetDevice(device), "cudaSetDevice dummy compute failed");

  float* data = static_cast<float*>(device_ptr);
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  try {
    CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreate dummy compute failed");
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate dummy compute start failed");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate dummy compute stop failed");

    const int threads = 256;
    const int max_blocks = 65535;
    int blocks = static_cast<int>((elements + threads - 1) / threads);
    if (blocks < 1) {
      blocks = 1;
    }
    if (blocks > max_blocks) {
      blocks = max_blocks;
    }

    CheckCuda(cudaEventRecord(start, stream),
              "cudaEventRecord dummy compute start failed");
    DummyComputeKernel<<<blocks, threads, 0, stream>>>(data, elements, iterations);
    CheckCuda(cudaGetLastError(), "dummy compute kernel launch failed");
    CheckCuda(cudaEventRecord(stop, stream),
              "cudaEventRecord dummy compute stop failed");
    CheckCuda(cudaEventSynchronize(stop),
              "cudaEventSynchronize dummy compute failed");

    float milliseconds = 0.0f;
    CheckCuda(cudaEventElapsedTime(&milliseconds, start, stop),
              "cudaEventElapsedTime dummy compute failed");

    IgnoreCuda(cudaEventDestroy(start));
    IgnoreCuda(cudaEventDestroy(stop));
    IgnoreCuda(cudaStreamDestroy(stream));
    CheckCuda(cudaSetDevice(previous_device), "cudaSetDevice dummy restore failed");

    DummyComputeStats stats;
    stats.elements = elements;
    stats.iterations = iterations;
    stats.cuda_elapsed_ms = static_cast<double>(milliseconds);
    return stats;
  } catch (...) {
    if (start != nullptr) {
      IgnoreCuda(cudaEventDestroy(start));
    }
    if (stop != nullptr) {
      IgnoreCuda(cudaEventDestroy(stop));
    }
    if (stream != nullptr) {
      IgnoreCuda(cudaStreamDestroy(stream));
    }
    IgnoreCuda(cudaSetDevice(previous_device));
    throw;
  }
}

}  // namespace turbobus
