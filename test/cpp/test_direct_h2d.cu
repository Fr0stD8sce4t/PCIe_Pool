#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "turbobus/runtime.h"

namespace {

void CheckCuda(cudaError_t result, const char* message) {
  if (result != cudaSuccess) {
    std::cerr << message << ": " << cudaGetErrorString(result) << "\n";
    std::exit(1);
  }
}

int EnvInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return std::atoi(value);
}

std::size_t EnvSize(const char* name, std::size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return static_cast<std::size_t>(std::strtoull(value, nullptr, 10));
}

}  // namespace

int main() {
  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed");
  if (device_count < 1) {
    std::cerr << "requires at least one CUDA GPU\n";
    return 77;
  }

  const int target = EnvInt("TURBOBUS_TARGET_GPU", 0);
  if (target < 0 || target >= device_count) {
    std::cerr << "TURBOBUS_TARGET_GPU is out of range\n";
    return 2;
  }
  const std::size_t bytes = EnvSize("TURBOBUS_TEST_BYTES", 16ull * 1024ull * 1024ull);
  auto* host = static_cast<std::uint8_t*>(nullptr);
  auto* dst = static_cast<std::uint8_t*>(nullptr);
  std::vector<std::uint8_t> back(bytes);

  CheckCuda(cudaSetDevice(target), "cudaSetDevice failed");
  CheckCuda(cudaMallocHost(&host, bytes), "cudaMallocHost failed");
  CheckCuda(cudaMalloc(&dst, bytes), "cudaMalloc failed");

  for (std::size_t i = 0; i < bytes; ++i) {
    host[i] = static_cast<std::uint8_t>(i % 251);
  }

  turbobus::RuntimeOptions options;
  options.chunk_bytes = EnvSize("TURBOBUS_CHUNK_BYTES", 4ull * 1024ull * 1024ull);
  options.staging_slots = 2;

  turbobus::TurboBusRuntime runtime(options);
  runtime.Init(target, {});
  runtime.Profile(32ull * 1024ull * 1024ull);
  auto handle = runtime.FetchToGpu(host, dst, bytes);
  runtime.Wait(handle);

  CheckCuda(cudaMemcpy(back.data(), dst, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H failed");

  for (std::size_t i = 0; i < bytes; ++i) {
    assert(back[i] == host[i]);
  }

  cudaFree(dst);
  cudaFreeHost(host);
  std::cout << "direct h2d test passed\n";
  return 0;
}
