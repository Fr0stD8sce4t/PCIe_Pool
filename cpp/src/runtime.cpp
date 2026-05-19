#include "turbobus/runtime.h"

#include <stdexcept>

namespace turbobus {

TurboBusRuntime::TurboBusRuntime(RuntimeOptions options) : options_(options) {}

TurboBusRuntime::~TurboBusRuntime() = default;

void TurboBusRuntime::Init(int target_device, const std::vector<int>& relay_devices) {
  target_device_ = target_device;
  requested_relays_ = relay_devices;
  profile_ = {};
  has_profile_ = false;

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

ProfileResult TurboBusRuntime::Profile(std::size_t bytes, bool force) {
  if (!initialized_) {
    throw std::runtime_error("runtime is not initialized");
  }
  if (force || !has_profile_ || !options_.profile_cache_enabled) {
    profile_ = profiler_.Profile(target_device_, enabled_relays_, bytes);
    has_profile_ = true;
  }
  return profile_;
}

void TurboBusRuntime::SetTransferMode(TransferMode mode) {
  options_.transfer_mode = mode;
}

TransferHandle TurboBusRuntime::FetchToGpu(void* host_ptr, void* target_gpu_ptr,
                                           std::size_t bytes) {
  if (!initialized_) {
    throw std::runtime_error("runtime is not initialized");
  }
  if (!has_profile_ && options_.profile_on_first_transfer) {
    Profile(options_.profile_bytes, false);
  }
  if (!has_profile_) {
    profile_.target_device = target_device_;
    profile_.direct_h2d_bw_gbps = 1.0;
    for (const int relay_device : enabled_relays_) {
      RelayProfile relay;
      relay.relay_device = relay_device;
      relay.target_device = target_device_;
      relay.h2d_bw_gbps = 1.0;
      relay.p2p_bw_gbps = 1.0;
      relay.effective_bw_gbps = 1.0;
      relay.p2p_enabled = true;
      profile_.relays.push_back(relay);
    }
    has_profile_ = true;
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

  const TransferPlan plan = planner_.Plan(bytes, options_.chunk_bytes, profile_,
                                         options_.transfer_mode,
                                         options_.min_chunks_for_relay);
  return executor_.Submit(host, target, plan);
}

void TurboBusRuntime::Wait(const TransferHandle& handle) {
  executor_.Wait(handle);
}

TransferStats TurboBusRuntime::GetStats(const TransferHandle& handle) const {
  return executor_.GetStats(handle);
}

const ProfileResult& TurboBusRuntime::CachedProfile() const {
  return profile_;
}

}  // namespace turbobus
