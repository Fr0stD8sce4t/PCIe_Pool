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

turbobus::TransferPlan MakePlan(std::size_t bytes, std::size_t chunk_bytes, int target,
                                int relay, const char* mode) {
  turbobus::TransferPlan plan;
  plan.total_bytes = bytes;
  plan.chunk_bytes = chunk_bytes;

  turbobus::PathAssignment direct;
  direct.path.kind = turbobus::PathKind::DirectH2D;
  direct.path.target_device = target;
  direct.path.relay_device = turbobus::kHostDevice;
  direct.path.effective_bw_gbps = 1.0;

  turbobus::PathAssignment relay_assignment;
  relay_assignment.path.kind = turbobus::PathKind::RelayH2DThenP2P;
  relay_assignment.path.target_device = target;
  relay_assignment.path.relay_device = relay;
  relay_assignment.path.effective_bw_gbps = 1.0;

  for (std::size_t offset = 0, chunk_idx = 0; offset < bytes;
       offset += chunk_bytes, ++chunk_idx) {
    turbobus::Chunk chunk;
    chunk.src_offset = offset;
    chunk.dst_offset = offset;
    chunk.bytes = std::min(chunk_bytes, bytes - offset);

    if (std::string(mode) == "direct") {
      direct.chunks.push_back(chunk);
    } else if (std::string(mode) == "relay") {
      relay_assignment.chunks.push_back(chunk);
    } else {
      if (chunk_idx % 2 == 0) {
        direct.chunks.push_back(chunk);
      } else {
        relay_assignment.chunks.push_back(chunk);
      }
    }
  }

  if (!direct.chunks.empty()) {
    plan.assignments.push_back(direct);
  }
  if (!relay_assignment.chunks.empty()) {
    plan.assignments.push_back(relay_assignment);
  }
  return plan;
}

double RunMode(const char* mode, turbobus::CudaRelayExecutor& executor, std::uint8_t* host,
               std::uint8_t* dst, std::size_t bytes, std::size_t chunk_bytes, int target,
               int relay, bool verify) {
  auto plan = MakePlan(bytes, chunk_bytes, target, relay, mode);

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

  const auto start = std::chrono::steady_clock::now();
  auto handle = executor.Submit(src_view, dst_view, plan);
  executor.Wait(handle);
  const auto stop = std::chrono::steady_clock::now();
  const auto microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  const double milliseconds = static_cast<double>(microseconds) / 1000.0;

  CheckCuda(cudaSetDevice(target), "cudaSetDevice benchmark target readback failed");

  if (verify) {
    std::vector<std::uint8_t> back(bytes);
    CheckCuda(cudaMemcpy(back.data(), dst, bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy benchmark readback failed");
    Verify(host, back);
  }

  const double bandwidth = Gbps(bytes, static_cast<float>(milliseconds));
  std::cout << mode << "_milliseconds=" << milliseconds << "\n";
  std::cout << mode << "_gib_per_second=" << bandwidth << "\n";
  return bandwidth;
}

}  // namespace

int main() {
  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed");

  const int target = EnvInt("TURBOBUS_TARGET_GPU", 6);
  const int relay = EnvInt("TURBOBUS_RELAY_GPU", 5);
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

  const std::size_t bytes =
      EnvSize("TURBOBUS_BENCH_BYTES", 256ull * 1024ull * 1024ull);
  const std::size_t chunk_bytes =
      EnvSize("TURBOBUS_CHUNK_BYTES", 16ull * 1024ull * 1024ull);
  const bool verify = EnvInt("TURBOBUS_VERIFY", 1) != 0;

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
  executor.Init(target, {relay}, options);

  std::cout << "target_gpu=" << target << "\n";
  std::cout << "relay_gpu=" << relay << "\n";
  std::cout << "bytes=" << bytes << "\n";
  std::cout << "chunk_bytes=" << chunk_bytes << "\n";

  const double direct = RunMode("direct", executor, host, dst, bytes, chunk_bytes, target,
                                relay, verify);
  const double relay_bw = RunMode("relay", executor, host, dst, bytes, chunk_bytes, target,
                                  relay, verify);
  const double pool = RunMode("pool", executor, host, dst, bytes, chunk_bytes, target,
                              relay, verify);

  std::cout << "pool_over_direct=" << (pool / direct) << "\n";
  std::cout << "pool_over_relay=" << (pool / relay_bw) << "\n";

  cudaFree(dst);
  cudaFreeHost(host);
  return 0;
}
