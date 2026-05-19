#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "turbobus/executor.h"
#include "turbobus/planner.h"
#include "turbobus/profiler.h"
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

double Gbps(std::size_t bytes, float milliseconds) {
  const double seconds = static_cast<double>(milliseconds) / 1000.0;
  const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  return gib / seconds;
}

double Median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void FillHost(std::uint8_t* host, std::size_t bytes) {
  for (std::size_t i = 0; i < bytes; ++i) {
    host[i] = static_cast<std::uint8_t>((i * 31) % 251);
  }
}

void Verify(const std::uint8_t* expected, const std::vector<std::uint8_t>& actual) {
  for (std::size_t i = 0; i < actual.size(); ++i) {
    assert(actual[i] == expected[i]);
  }
}

turbobus::ProfileResult FilterProfile(const turbobus::ProfileResult& profile,
                                      const std::vector<int>& relays) {
  turbobus::ProfileResult filtered;
  filtered.target_device = profile.target_device;
  filtered.direct_h2d_bw_gbps = profile.direct_h2d_bw_gbps;
  for (const auto& relay : profile.relays) {
    if (std::find(relays.begin(), relays.end(), relay.relay_device) != relays.end()) {
      filtered.relays.push_back(relay);
    }
  }
  return filtered;
}

turbobus::TransferPlan MakeDirectPlan(std::size_t bytes, std::size_t chunk_bytes, int target) {
  turbobus::TransferPlan plan;
  plan.total_bytes = bytes;
  plan.chunk_bytes = chunk_bytes;

  turbobus::PathAssignment direct;
  direct.path.kind = turbobus::PathKind::DirectH2D;
  direct.path.target_device = target;
  direct.path.relay_device = turbobus::kHostDevice;
  direct.path.effective_bw_gbps = 1.0;

  for (std::size_t offset = 0; offset < bytes; offset += chunk_bytes) {
    turbobus::Chunk chunk;
    chunk.src_offset = offset;
    chunk.dst_offset = offset;
    chunk.bytes = std::min(chunk_bytes, bytes - offset);
    direct.chunks.push_back(chunk);
  }

  plan.assignments.push_back(direct);
  return plan;
}

turbobus::TransferPlan MakeRelayOnlyPlan(std::size_t bytes, std::size_t chunk_bytes,
                                         int target, const std::vector<int>& relays) {
  turbobus::TransferPlan plan;
  plan.total_bytes = bytes;
  plan.chunk_bytes = chunk_bytes;
  if (relays.empty()) {
    return plan;
  }

  std::vector<turbobus::PathAssignment> assignments;
  assignments.reserve(relays.size());
  for (const int relay : relays) {
    turbobus::PathAssignment assignment;
    assignment.path.kind = turbobus::PathKind::RelayH2DThenP2P;
    assignment.path.target_device = target;
    assignment.path.relay_device = relay;
    assignment.path.effective_bw_gbps = 1.0;
    assignments.push_back(assignment);
  }

  for (std::size_t offset = 0, chunk_idx = 0; offset < bytes;
       offset += chunk_bytes, ++chunk_idx) {
    turbobus::Chunk chunk;
    chunk.src_offset = offset;
    chunk.dst_offset = offset;
    chunk.bytes = std::min(chunk_bytes, bytes - offset);
    assignments[chunk_idx % assignments.size()].chunks.push_back(chunk);
  }

  for (auto& assignment : assignments) {
    if (!assignment.chunks.empty()) {
      plan.assignments.push_back(assignment);
    }
  }
  return plan;
}

double RunMode(const char* mode, turbobus::CudaRelayExecutor& executor, std::uint8_t* host,
               std::uint8_t* dst, const turbobus::TransferPlan& plan, int target,
               bool verify) {
  turbobus::BufferView src_view;
  src_view.ptr = host;
  src_view.bytes = plan.total_bytes;
  src_view.kind = turbobus::MemoryKind::HostPinned;
  src_view.device = turbobus::kHostDevice;

  turbobus::BufferView dst_view;
  dst_view.ptr = dst;
  dst_view.bytes = plan.total_bytes;
  dst_view.kind = turbobus::MemoryKind::Device;
  dst_view.device = target;

  const auto start = std::chrono::steady_clock::now();
  auto handle = executor.Submit(src_view, dst_view, plan);
  executor.Wait(handle);
  const auto stop = std::chrono::steady_clock::now();
  const auto microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  const double milliseconds = static_cast<double>(microseconds) / 1000.0;

  CheckCuda(cudaSetDevice(target), "cudaSetDevice benchmark target readback failed");

  if (verify) {
    std::vector<std::uint8_t> back(plan.total_bytes);
    CheckCuda(cudaMemcpy(back.data(), dst, plan.total_bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy benchmark readback failed");
    Verify(host, back);
  }

  const double bandwidth = Gbps(plan.total_bytes, static_cast<float>(milliseconds));
  std::cout << mode << "_milliseconds=" << milliseconds << "\n";
  std::cout << mode << "_gib_per_second=" << bandwidth << "\n";
  return bandwidth;
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
    const auto token = text.substr(start, comma == std::string::npos ? comma : comma - start);
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

}  // namespace

int main() {
  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed");

  const int target = EnvInt("TURBOBUS_TARGET_GPU", 6);
  if (target < 0 || target >= device_count) {
    std::cerr << "invalid target GPU\n";
    return 2;
  }

  std::vector<int> requested_relays = ParseRelays(std::getenv("TURBOBUS_RELAY_GPUS"));
  if (requested_relays.empty()) {
    const int relay = EnvInt("TURBOBUS_RELAY_GPU", -1);
    if (relay >= 0) {
      requested_relays.push_back(relay);
    }
  }

  std::vector<int> enabled_relays;
  for (const int relay : requested_relays) {
    if (relay < 0 || relay >= device_count || relay == target) {
      continue;
    }
    int can_access = 0;
    CheckCuda(cudaDeviceCanAccessPeer(&can_access, relay, target),
              "cudaDeviceCanAccessPeer failed");
    if (!can_access) {
      continue;
    }
    CheckCuda(cudaSetDevice(relay), "cudaSetDevice relay failed");
    auto enable_result = cudaDeviceEnablePeerAccess(target, 0);
    if (enable_result == cudaErrorPeerAccessAlreadyEnabled) {
      cudaGetLastError();
    } else {
      CheckCuda(enable_result, "cudaDeviceEnablePeerAccess relay->target failed");
    }
    enabled_relays.push_back(relay);
  }

  if (enabled_relays.empty()) {
    for (int relay = 0; relay < device_count; ++relay) {
      if (relay == target) {
        continue;
      }
      int can_access = 0;
      CheckCuda(cudaDeviceCanAccessPeer(&can_access, relay, target),
                "cudaDeviceCanAccessPeer scan failed");
      if (!can_access) {
        continue;
      }
      CheckCuda(cudaSetDevice(relay), "cudaSetDevice relay scan failed");
      auto enable_result = cudaDeviceEnablePeerAccess(target, 0);
      if (enable_result == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else {
        CheckCuda(enable_result, "cudaDeviceEnablePeerAccess scan failed");
      }
      enabled_relays.push_back(relay);
    }
  }

  if (enabled_relays.empty()) {
    std::cerr << "no P2P-capable relay GPU found for target " << target << "\n";
    return 77;
  }

  const std::size_t bytes =
      EnvSize("TURBOBUS_BENCH_BYTES", 256ull * 1024ull * 1024ull);
  const std::size_t chunk_bytes =
      EnvSize("TURBOBUS_CHUNK_BYTES", 16ull * 1024ull * 1024ull);
  const bool verify = EnvInt("TURBOBUS_VERIFY", 1) != 0;
  const int iterations = EnvInt("TURBOBUS_BENCH_ITERS", 5);
  const std::size_t profile_bytes =
      EnvSize("TURBOBUS_PROFILE_BYTES", 16ull * 1024ull * 1024ull);

  auto* host = static_cast<std::uint8_t*>(nullptr);
  auto* dst = static_cast<std::uint8_t*>(nullptr);
  CheckCuda(cudaMallocHost(&host, bytes), "cudaMallocHost failed");
  CheckCuda(cudaSetDevice(target), "cudaSetDevice target failed");
  CheckCuda(cudaMalloc(&dst, bytes), "cudaMalloc target failed");
  FillHost(host, bytes);

  turbobus::RuntimeOptions options;
  options.chunk_bytes = chunk_bytes;
  options.staging_slots = EnvInt("TURBOBUS_STAGING_SLOTS", 2);

  turbobus::CudaRelayExecutor executor;
  executor.Init(target, enabled_relays, options);

  turbobus::BandwidthProfiler profiler;
  auto profile = profiler.Profile(target, enabled_relays, profile_bytes);
  profile = FilterProfile(profile, enabled_relays);
  turbobus::ChunkPlanner planner;
  const auto direct_plan = MakeDirectPlan(bytes, chunk_bytes, target);
  const auto relay_plan = MakeRelayOnlyPlan(bytes, chunk_bytes, target, enabled_relays);
  const auto pool_plan = planner.Plan(bytes, chunk_bytes, profile);

  std::cout << "target_gpu=" << target << "\n";
  std::cout << "relay_gpus=";
  for (const int relay : enabled_relays) {
    std::cout << relay << ",";
  }
  std::cout << "\n";
  std::cout << "bytes=" << bytes << "\n";
  std::cout << "chunk_bytes=" << chunk_bytes << "\n";
  std::cout << "iterations=" << iterations << "\n";

  std::vector<double> direct_values;
  std::vector<double> relay_values;
  std::vector<double> pool_values;
  for (int i = 0; i < iterations; ++i) {
    direct_values.push_back(RunMode("direct", executor, host, dst, direct_plan, target, verify));
    relay_values.push_back(RunMode("relay", executor, host, dst, relay_plan, target, verify));
    pool_values.push_back(RunMode("pool", executor, host, dst, pool_plan, target, verify));
  }

  const double direct = Median(direct_values);
  const double relay_bw = Median(relay_values);
  const double pool = Median(pool_values);
  std::cout << "direct_median_gib_per_second=" << direct << "\n";
  std::cout << "relay_median_gib_per_second=" << relay_bw << "\n";
  std::cout << "pool_median_gib_per_second=" << pool << "\n";
  std::cout << "pool_over_direct_median=" << (pool / direct) << "\n";
  std::cout << "pool_over_relay_median=" << (pool / relay_bw) << "\n";

  cudaFree(dst);
  cudaFreeHost(host);
  return 0;
}
