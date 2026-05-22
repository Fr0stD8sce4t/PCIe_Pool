#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
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

std::vector<int> ParseRelays(const char* value) {
  std::vector<int> relays;
  if (value == nullptr || value[0] == '\0') {
    return relays;
  }
  std::string text(value);
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t comma = text.find(',', start);
    const auto token =
        text.substr(start, comma == std::string::npos ? comma : comma - start);
    if (!token.empty()) {
      relays.push_back(std::atoi(token.c_str()));
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return relays;
}

bool EnableRelayToTarget(int relay, int target) {
  int can_access = 0;
  CheckCuda(cudaDeviceCanAccessPeer(&can_access, relay, target),
            "cudaDeviceCanAccessPeer failed");
  if (!can_access) {
    return false;
  }
  CheckCuda(cudaSetDevice(relay), "cudaSetDevice relay failed");
  auto enable_result = cudaDeviceEnablePeerAccess(target, 0);
  if (enable_result == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
  } else {
    CheckCuda(enable_result, "cudaDeviceEnablePeerAccess relay->target failed");
  }
  return true;
}

std::vector<int> PickRelays(int device_count, int target) {
  std::vector<int> requested = ParseRelays(std::getenv("TURBOBUS_RELAY_GPUS"));
  if (requested.empty()) {
    const int relay = EnvInt("TURBOBUS_RELAY_GPU", -1);
    if (relay >= 0) {
      requested.push_back(relay);
    }
  }
  if (requested.empty()) {
    for (int relay = 0; relay < device_count; ++relay) {
      requested.push_back(relay);
    }
  }

  std::vector<int> relays;
  for (const int relay : requested) {
    if (relay < 0 || relay >= device_count || relay == target ||
        std::find(relays.begin(), relays.end(), relay) != relays.end()) {
      continue;
    }
    if (EnableRelayToTarget(relay, target)) {
      relays.push_back(relay);
    }
    if (relays.size() == 2) {
      break;
    }
  }
  return relays;
}

turbobus::PathAssignment MakeAssignment(turbobus::PathKind kind, int target,
                                        int relay) {
  turbobus::PathAssignment assignment;
  assignment.path.kind = kind;
  assignment.path.direction = turbobus::TransferDirection::H2D;
  assignment.path.target_device = target;
  assignment.path.relay_device = relay;
  assignment.path.h2d_bw_gbps = 1.0;
  assignment.path.d2h_bw_gbps = 1.0;
  assignment.path.p2p_bw_gbps = relay == turbobus::kHostDevice ? 0.0 : 1.0;
  assignment.path.effective_bw_gbps = 1.0;
  assignment.path.enabled = true;
  return assignment;
}

std::size_t RelayBytes(const turbobus::TransferStats& stats, int relay) {
  for (std::size_t i = 0; i < stats.relay_devices.size(); ++i) {
    if (stats.relay_devices[i] == relay) {
      return stats.relay_device_bytes[i];
    }
  }
  return 0;
}

std::size_t RelayChunks(const turbobus::TransferStats& stats, int relay) {
  for (std::size_t i = 0; i < stats.relay_devices.size(); ++i) {
    if (stats.relay_devices[i] == relay) {
      return stats.relay_device_chunks[i];
    }
  }
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed");
  if (device_count < 3) {
    std::cerr << "requires at least three CUDA GPUs\n";
    return 77;
  }

  int target = EnvInt("TURBOBUS_TARGET_GPU", -1);
  std::vector<int> relays;
  if (target >= 0) {
    if (target >= device_count) {
      std::cerr << "invalid target GPU\n";
      return 2;
    }
    relays = PickRelays(device_count, target);
  } else {
    for (int candidate = 0; candidate < device_count; ++candidate) {
      relays = PickRelays(device_count, candidate);
      if (relays.size() >= 2) {
        target = candidate;
        break;
      }
    }
  }
  if (relays.size() < 2) {
    std::cerr << "requires at least two P2P-capable relay GPUs\n";
    return 77;
  }

  const std::size_t bytes =
      EnvSize("TURBOBUS_TEST_BYTES", 48ull * 1024ull * 1024ull);
  const std::size_t chunk_bytes =
      EnvSize("TURBOBUS_CHUNK_BYTES", 4ull * 1024ull * 1024ull);
  if (bytes < chunk_bytes * 3) {
    std::cerr << "test bytes must cover at least three chunks\n";
    return 2;
  }

  auto* host = static_cast<std::uint8_t*>(nullptr);
  auto* dst = static_cast<std::uint8_t*>(nullptr);
  CheckCuda(cudaMallocHost(&host, bytes), "cudaMallocHost failed");
  CheckCuda(cudaSetDevice(target), "cudaSetDevice target failed");
  CheckCuda(cudaMalloc(&dst, bytes), "cudaMalloc target failed");

  for (std::size_t i = 0; i < bytes; ++i) {
    host[i] = static_cast<std::uint8_t>((i * 19) % 251);
  }

  turbobus::RuntimeOptions options;
  options.chunk_bytes = chunk_bytes;
  options.staging_slots = EnvInt("TURBOBUS_STAGING_SLOTS", 2);

  turbobus::CudaRelayExecutor executor;
  executor.Init(target, relays, options);

  turbobus::TransferPlan plan;
  plan.total_bytes = bytes;
  plan.chunk_bytes = chunk_bytes;
  plan.assignments.push_back(
      MakeAssignment(turbobus::PathKind::DirectH2D, target, turbobus::kHostDevice));
  plan.assignments.push_back(
      MakeAssignment(turbobus::PathKind::RelayH2DThenP2P, target, relays[0]));
  plan.assignments.push_back(
      MakeAssignment(turbobus::PathKind::RelayH2DThenP2P, target, relays[1]));

  for (std::size_t offset = 0, chunk_index = 0; offset < bytes;
       offset += chunk_bytes, ++chunk_index) {
    turbobus::Chunk chunk;
    chunk.src_offset = offset;
    chunk.dst_offset = offset;
    chunk.bytes = std::min(chunk_bytes, bytes - offset);
    plan.assignments[chunk_index % plan.assignments.size()].chunks.push_back(chunk);
  }

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
  const auto stats = executor.GetStats(handle);

  CheckCuda(cudaSetDevice(target), "cudaSetDevice target readback failed");
  std::vector<std::uint8_t> back(bytes);
  CheckCuda(cudaMemcpy(back.data(), dst, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy readback failed");
  for (std::size_t i = 0; i < bytes; ++i) {
    assert(back[i] == host[i]);
  }

  assert(stats.direct_chunks == plan.assignments[0].chunks.size());
  assert(stats.relay_devices.size() == 2);
  assert(RelayChunks(stats, relays[0]) == plan.assignments[1].chunks.size());
  assert(RelayChunks(stats, relays[1]) == plan.assignments[2].chunks.size());
  assert(RelayBytes(stats, relays[0]) > 0);
  assert(RelayBytes(stats, relays[1]) > 0);
  assert(stats.path_stats.size() == 3);

  cudaFree(dst);
  cudaFreeHost(host);
  std::cout << "multi-relay pool test passed\n";
  return 0;
}
