#include <cassert>
#include <algorithm>
#include <cstddef>
#include <cstdint>
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

}  // namespace

int main() {
  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed");
  if (device_count < 2) {
    std::cerr << "requires at least two CUDA GPUs\n";
    return 77;
  }

  const int target = 0;
  const int relay = 1;
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

  const std::size_t bytes = 96ull * 1024ull * 1024ull;
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
  options.chunk_bytes = 16ull * 1024ull * 1024ull;
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
