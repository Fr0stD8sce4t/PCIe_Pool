#include "turbobus/runtime.h"

#include <stdexcept>

namespace turbobus {

TurboBusRuntime::TurboBusRuntime(RuntimeOptions options) : options_(options) {}

TurboBusRuntime::~TurboBusRuntime() = default;

void TurboBusRuntime::Init(int target_device, const std::vector<int>& relay_devices) {
  target_device_ = target_device;
  requested_relays_ = relay_devices;

  topology_ = topology_manager_.Discover(target_device_, requested_relays_,
                                         options_.enable_peer_access);

  enabled_relays_.clear();
  enabled_relays_.reserve(topology_.relay_candidates.size());
  for (const auto& relay : topology_.relay_candidates) {
    enabled_relays_.push_back(relay.device_id);
  }

  executor_.Init(target_device_, enabled_relays_, options_);
  initialized_ = true;
}

ProfileResult TurboBusRuntime::Profile(std::size_t bytes) {
  if (!initialized_) {
    throw std::runtime_error("runtime is not initialized");
  }
  profile_ = profiler_.Profile(target_device_, enabled_relays_, bytes);
  return profile_;
}

TransferHandle TurboBusRuntime::FetchToGpu(void* host_ptr, void* target_gpu_ptr,
                                           std::size_t bytes) {
  if (!initialized_) {
    throw std::runtime_error("runtime is not initialized");
  }
  if (profile_.direct_h2d_bw_gbps <= 0.0 && profile_.relays.empty()) {
    Profile();
  }

  BufferView host;
  host.ptr = host_ptr;
  host.bytes = bytes;
  host.kind = MemoryKind::HostPinned;
  host.device = kHostDevice;

  BufferView target;
  target.ptr = target_gpu_ptr;
  target.bytes = bytes;
  target.kind = MemoryKind::Device;
  target.device = target_device_;

  const TransferPlan plan = planner_.Plan(bytes, options_.chunk_bytes, profile_);
  return executor_.Submit(host, target, plan);
}

void TurboBusRuntime::Wait(const TransferHandle& handle) {
  executor_.Wait(handle);
}

}  // namespace turbobus

