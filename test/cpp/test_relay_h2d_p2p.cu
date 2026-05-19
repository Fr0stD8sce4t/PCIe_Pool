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

bool PickP2PPair(int device_count, int* target, int* relay) {
  const int env_target = EnvInt("TURBOBUS_TARGET_GPU", -1);
  const int env_relay = EnvInt("TURBOBUS_RELAY_GPU", -1);
  if (env_target >= 0 && env_relay >= 0) {
    *target = env_target;
    *relay = env_relay;
    return true;
  }

  for (int dst = 0; dst < device_count; ++dst) {
    for (int src = 0; src < device_count; ++src) {
      if (src == dst) {
        continue;
      }
      int can_access = 0;
      CheckCuda(cudaDeviceCanAccessPeer(&can_access, src, dst),
                "cudaDeviceCanAccessPeer failed");
      if (can_access) {
        *target = dst;
        *relay = src;
        return true;
      }
    }
  }
  return false;
}

}  // namespace

int main() {
  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed");
  if (device_count < 2) {
    std::cerr << "requires at least two CUDA GPUs\n";
    return 77;
  }

  int target = -1;
  int relay = -1;
  if (!PickP2PPair(device_count, &target, &relay)) {
    std::cerr << "no P2P-capable GPU pair found\n";
    return 77;
  }
  if (target < 0 || target >= device_count || relay < 0 || relay >= device_count ||
      target == relay) {
    std::cerr << "invalid target/relay GPU pair\n";
    return 2;
  }

  int can_access = 0;
  CheckCuda(cudaDeviceCanAccessPeer(&can_access, relay, target),
            "cudaDeviceCanAccessPeer failed");
  if (!can_access) {
    std::cerr << "relay GPU cannot access target GPU through P2P\n";
    return 77;
  }

  CheckCuda(cudaSetDevice(relay), "cudaSetDevice relay failed");
  auto enable_result = cudaDeviceEnablePeerAccess(target, 0);
  if (enable_result == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
  } else {
    CheckCuda(enable_result, "cudaDeviceEnablePeerAccess relay->target failed");
  }

  std::cout << "using target GPU " << target << ", relay GPU " << relay << "\n";

  const std::size_t bytes = EnvSize("TURBOBUS_TEST_BYTES", 32ull * 1024ull * 1024ull);
  auto* host = static_cast<std::uint8_t*>(nullptr);
  auto* dst = static_cast<std::uint8_t*>(nullptr);
  std::vector<std::uint8_t> back(bytes);

  CheckCuda(cudaMallocHost(&host, bytes), "cudaMallocHost failed");
  CheckCuda(cudaSetDevice(target), "cudaSetDevice target failed");
  CheckCuda(cudaMalloc(&dst, bytes), "cudaMalloc target failed");

  for (std::size_t i = 0; i < bytes; ++i) {
    host[i] = static_cast<std::uint8_t>((i * 17) % 251);
  }

  turbobus::RuntimeOptions options;
  options.chunk_bytes = EnvSize("TURBOBUS_CHUNK_BYTES", 4ull * 1024ull * 1024ull);
  options.staging_slots = 2;

  turbobus::CudaRelayExecutor executor;
  executor.Init(target, {relay}, options);

  turbobus::TransferPlan plan;
  plan.total_bytes = bytes;
  plan.chunk_bytes = options.chunk_bytes;

  turbobus::PathAssignment assignment;
  assignment.path.kind = turbobus::PathKind::RelayH2DThenP2P;
  assignment.path.target_device = target;
  assignment.path.relay_device = relay;
  assignment.path.effective_bw_gbps = 1.0;

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

  auto handle = executor.Submit(src_view, dst_view, plan);
  executor.Wait(handle);

  CheckCuda(cudaSetDevice(target), "cudaSetDevice target readback failed");
  CheckCuda(cudaMemcpy(back.data(), dst, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H failed");

  for (std::size_t i = 0; i < bytes; ++i) {
    assert(back[i] == host[i]);
  }

  cudaFree(dst);
  cudaFreeHost(host);
  std::cout << "relay h2d+p2p test passed\n";
  return 0;
}
