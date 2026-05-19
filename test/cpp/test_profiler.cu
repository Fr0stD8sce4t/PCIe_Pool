#include <cassert>
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

}  // namespace

int main() {
  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed");
  if (device_count < 1) {
    std::cerr << "requires at least one CUDA GPU\n";
    return 77;
  }

  std::vector<int> relays;
  if (device_count >= 2) {
    relays.push_back(1);
  }

  turbobus::BandwidthProfiler profiler;
  const auto profile = profiler.Profile(0, relays, 32ull * 1024ull * 1024ull);

  std::cout << "direct_h2d_bw_gbps=" << profile.direct_h2d_bw_gbps << "\n";
  assert(profile.direct_h2d_bw_gbps > 0.0);

  for (const auto& relay : profile.relays) {
    std::cout << "relay=" << relay.relay_device
              << " h2d=" << relay.h2d_bw_gbps
              << " p2p=" << relay.p2p_bw_gbps
              << " effective=" << relay.effective_bw_gbps
              << " p2p_enabled=" << relay.p2p_enabled << "\n";
    assert(relay.h2d_bw_gbps > 0.0);
    if (relay.p2p_enabled) {
      assert(relay.p2p_bw_gbps > 0.0);
      assert(relay.effective_bw_gbps > 0.0);
    }
  }

  return 0;
}

