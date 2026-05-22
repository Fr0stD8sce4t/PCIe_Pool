#include <cassert>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#include "turbobus/profiler.h"

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

  std::vector<int> relays;
  const int env_relay = EnvInt("TURBOBUS_RELAY_GPU", -1);
  if (env_relay >= 0) {
    if (env_relay >= device_count || env_relay == target) {
      std::cerr << "TURBOBUS_RELAY_GPU is invalid\n";
      return 2;
    }
    relays.push_back(env_relay);
  } else if (device_count >= 2) {
    for (int device = 0; device < device_count; ++device) {
      if (device != target) {
        relays.push_back(device);
        break;
      }
    }
  }

  turbobus::BandwidthProfiler profiler;
  const auto profile = profiler.Profile(
      target, relays, EnvSize("TURBOBUS_PROFILE_BYTES", 16ull * 1024ull * 1024ull));

  std::cout << "direct_h2d_bw_gbps=" << profile.direct_h2d_bw_gbps << "\n";
  std::cout << "direct_d2h_bw_gbps=" << profile.direct_d2h_bw_gbps << "\n";
  assert(profile.direct_h2d_bw_gbps > 0.0);
  assert(profile.direct_d2h_bw_gbps > 0.0);

  for (const auto& relay : profile.relays) {
    std::cout << "relay=" << relay.relay_device
              << " h2d=" << relay.h2d_bw_gbps
              << " d2h=" << relay.d2h_bw_gbps
              << " p2p=" << relay.p2p_bw_gbps
              << " effective=" << relay.effective_bw_gbps
              << " effective_d2h=" << relay.effective_d2h_bw_gbps
              << " p2p_enabled=" << relay.p2p_enabled << "\n";
    assert(relay.h2d_bw_gbps > 0.0);
    assert(relay.d2h_bw_gbps > 0.0);
    if (relay.p2p_enabled) {
      assert(relay.p2p_bw_gbps > 0.0);
      assert(relay.effective_bw_gbps > 0.0);
      assert(relay.effective_d2h_bw_gbps > 0.0);
    }
  }

  return 0;
}
