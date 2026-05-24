#include <cassert>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "turbobus/executor.h"
#include "turbobus/types.h"

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

  turbobus::TransferPlan plan;
  plan.total_bytes = bytes;
  plan.chunk_bytes = options.chunk_bytes;

  turbobus::PathAssignment assignment;
  assignment.path.kind = turbobus::PathKind::DirectH2D;
  assignment.path.direction = turbobus::TransferDirection::H2D;
  assignment.path.target_device = target;
  assignment.path.relay_device = turbobus::kHostDevice;
  assignment.path.h2d_bw_gbps = 1.0;
  assignment.path.d2h_bw_gbps = 1.0;
  assignment.path.p2p_bw_gbps = 0.0;
  assignment.path.effective_bw_gbps = 1.0;
  assignment.path.enabled = true;
  for (std::size_t offset = 0; offset < bytes; offset += options.chunk_bytes) {
    turbobus::Chunk chunk;
    chunk.src_offset = offset;
    chunk.dst_offset = offset;
    chunk.bytes = std::min(options.chunk_bytes, bytes - offset);
    assignment.chunks.push_back(chunk);
  }
  plan.assignments.push_back(assignment);

  turbobus::BufferView src_view;
  src_view.ptr = host;
  src_view.bytes = bytes;
  src_view.kind = turbobus::MemoryKind::HostPinned;
  src_view.device = turbobus::kHostDevice;

  turbobus::BufferView dst_view;
  dst_view.ptr = dst;
  dst_view.bytes = bytes;
  dst_view.kind = turbobus::MemoryKind::Device;
  dst_view.device = target;

  turbobus::CudaRelayExecutor executor;
  executor.Init(target, {}, options);
  auto handle = executor.Submit(src_view, dst_view, plan);
  executor.Wait(handle);

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
