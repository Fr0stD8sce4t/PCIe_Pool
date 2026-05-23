#include "turbobus/runtime.h"

#include <algorithm>
#include <stdexcept>

namespace turbobus {

TurboBusRuntime::TurboBusRuntime(RuntimeOptions options) : options_(options) {}

TurboBusRuntime::~TurboBusRuntime() = default;

void TurboBusRuntime::Init(int target_device, const std::vector<int>& relay_devices) {
  target_device_ = target_device;
  requested_relays_ = relay_devices;
  profile_ = {};
  planner_profile_ = {};
  last_plan_ = {};
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
    planner_profile_ = profile_;
    has_profile_ = true;
  }
  return profile_;
}

void TurboBusRuntime::SetCachedProfile(const ProfileResult& profile) {
  if (!initialized_) {
    throw std::runtime_error("runtime is not initialized");
  }
  profile_ = profile;
  planner_profile_ = profile_;
  has_profile_ = true;
}

void TurboBusRuntime::SetTransferMode(TransferMode mode) {
  options_.transfer_mode = mode;
}

TransferHandle TurboBusRuntime::FetchToGpu(void* host_ptr, void* target_gpu_ptr,
                                           std::size_t bytes) {
  return SubmitTransfer(host_ptr, target_gpu_ptr, bytes, TransferDirection::H2D);
}

TransferHandle TurboBusRuntime::OffloadToCpu(void* target_gpu_ptr, void* host_ptr,
                                             std::size_t bytes) {
  return SubmitTransfer(target_gpu_ptr, host_ptr, bytes, TransferDirection::D2H);
}

TransferHandle TurboBusRuntime::FetchPlanToGpu(void* host_ptr, std::size_t host_bytes,
                                               void* target_gpu_ptr,
                                               std::size_t target_bytes,
                                               const TransferPlan& plan) {
  if (!initialized_) {
    throw std::runtime_error("runtime is not initialized");
  }
  BufferView source;
  source.ptr = host_ptr;
  source.bytes = host_bytes;
  source.device = kHostDevice;
  source.kind = MemoryKind::HostPinned;

  BufferView destination;
  destination.ptr = target_gpu_ptr;
  destination.bytes = target_bytes;
  destination.device = target_device_;
  destination.kind = MemoryKind::Device;

  last_plan_ = plan;
  return executor_.Submit(source, destination, last_plan_);
}

TransferHandle TurboBusRuntime::OffloadPlanToCpu(void* target_gpu_ptr,
                                                 std::size_t target_bytes,
                                                 void* host_ptr,
                                                 std::size_t host_bytes,
                                                 const TransferPlan& plan) {
  if (!initialized_) {
    throw std::runtime_error("runtime is not initialized");
  }
  BufferView source;
  source.ptr = target_gpu_ptr;
  source.bytes = target_bytes;
  source.device = target_device_;
  source.kind = MemoryKind::Device;

  BufferView destination;
  destination.ptr = host_ptr;
  destination.bytes = host_bytes;
  destination.device = kHostDevice;
  destination.kind = MemoryKind::HostPinned;

  last_plan_ = plan;
  return executor_.SubmitD2H(source, destination, last_plan_);
}

TransferHandle TurboBusRuntime::FetchRangesToGpu(
    void* host_ptr, std::size_t host_bytes, void* target_gpu_ptr,
    std::size_t target_bytes, const std::vector<TransferRange>& ranges) {
  return SubmitRanges(host_ptr, host_bytes, target_gpu_ptr, target_bytes, ranges,
                      TransferDirection::H2D);
}

TransferHandle TurboBusRuntime::OffloadRangesToCpu(
    void* target_gpu_ptr, std::size_t target_bytes, void* host_ptr,
    std::size_t host_bytes, const std::vector<TransferRange>& ranges) {
  return SubmitRanges(target_gpu_ptr, target_bytes, host_ptr, host_bytes, ranges,
                      TransferDirection::D2H);
}

DummyComputeStats TurboBusRuntime::RunDummyCompute(void* device_ptr, std::size_t elements,
                                                   int iterations) {
  if (!initialized_) {
    throw std::runtime_error("runtime is not initialized");
  }
  return turbobus::RunDummyCompute(target_device_, device_ptr, elements, iterations);
}

void TurboBusRuntime::EnsureProfile() {
  if (!initialized_) {
    throw std::runtime_error("runtime is not initialized");
  }
  if (!has_profile_ && options_.profile_on_first_transfer) {
    Profile(options_.profile_bytes, false);
  }
  if (!has_profile_) {
    profile_.target_device = target_device_;
    profile_.direct_h2d_bw_gbps = 1.0;
    profile_.direct_d2h_bw_gbps = 1.0;
    for (const int relay_device : enabled_relays_) {
      RelayProfile relay;
      relay.relay_device = relay_device;
      relay.target_device = target_device_;
      relay.h2d_bw_gbps = 1.0;
      relay.d2h_bw_gbps = 1.0;
      relay.p2p_bw_gbps = 1.0;
      relay.effective_bw_gbps = 1.0;
      relay.effective_d2h_bw_gbps = 1.0;
      relay.p2p_enabled = true;
      profile_.relays.push_back(relay);
    }
    planner_profile_ = profile_;
    has_profile_ = true;
  }
}

TransferHandle TurboBusRuntime::SubmitTransfer(void* source_ptr, void* destination_ptr,
                                               std::size_t bytes,
                                               TransferDirection direction) {
  EnsureProfile();

  BufferView source;
  source.ptr = source_ptr;
  source.bytes = bytes;
  source.device =
      direction == TransferDirection::H2D ? kHostDevice : target_device_;
  source.kind = direction == TransferDirection::H2D ? MemoryKind::HostPinned
                                                    : MemoryKind::Device;

  BufferView destination;
  destination.ptr = destination_ptr;
  destination.bytes = bytes;
  destination.device =
      direction == TransferDirection::H2D ? target_device_ : kHostDevice;
  destination.kind = direction == TransferDirection::H2D ? MemoryKind::Device
                                                         : MemoryKind::HostPinned;

  const auto& plan_profile =
      options_.enable_dynamic_weights ? planner_profile_ : profile_;
  last_plan_ = planner_.Plan(bytes, options_.chunk_bytes, plan_profile,
                             options_.transfer_mode,
                             options_.min_chunks_for_relay,
                             options_.relay_min_effective_bw_gbps,
                             options_.relay_min_direct_ratio,
                             direction);
  if (direction == TransferDirection::H2D) {
    return executor_.Submit(source, destination, last_plan_);
  }
  return executor_.SubmitD2H(source, destination, last_plan_);
}

TransferHandle TurboBusRuntime::SubmitRanges(
    void* source_ptr, std::size_t source_bytes, void* destination_ptr,
    std::size_t destination_bytes, const std::vector<TransferRange>& ranges,
    TransferDirection direction) {
  EnsureProfile();

  BufferView source;
  source.ptr = source_ptr;
  source.bytes = source_bytes;
  source.device =
      direction == TransferDirection::H2D ? kHostDevice : target_device_;
  source.kind = direction == TransferDirection::H2D ? MemoryKind::HostPinned
                                                    : MemoryKind::Device;

  BufferView destination;
  destination.ptr = destination_ptr;
  destination.bytes = destination_bytes;
  destination.device =
      direction == TransferDirection::H2D ? target_device_ : kHostDevice;
  destination.kind = direction == TransferDirection::H2D ? MemoryKind::Device
                                                         : MemoryKind::HostPinned;

  const auto& plan_profile =
      options_.enable_dynamic_weights ? planner_profile_ : profile_;
  last_plan_ = planner_.PlanRanges(
      ranges, options_.chunk_bytes, plan_profile, options_.transfer_mode,
      options_.min_chunks_for_relay, options_.relay_min_effective_bw_gbps,
      options_.relay_min_direct_ratio, direction);
  if (direction == TransferDirection::H2D) {
    return executor_.Submit(source, destination, last_plan_);
  }
  return executor_.SubmitD2H(source, destination, last_plan_);
}

void TurboBusRuntime::Wait(const TransferHandle& handle) {
  executor_.Wait(handle);
  if (options_.enable_dynamic_weights) {
    UpdateDynamicWeights(executor_.GetStats(handle));
  }
}

TransferStats TurboBusRuntime::GetStats(const TransferHandle& handle) const {
  return executor_.GetStats(handle);
}

const ProfileResult& TurboBusRuntime::CachedProfile() const {
  return profile_;
}

const ProfileResult& TurboBusRuntime::PlannerProfile() const {
  return planner_profile_;
}

const TransferPlan& TurboBusRuntime::LastPlan() const {
  return last_plan_;
}

void TurboBusRuntime::UpdateDynamicWeights(const TransferStats& stats) {
  if (!has_profile_) {
    return;
  }
  const double alpha = std::clamp(options_.dynamic_weight_alpha, 0.0, 1.0);
  for (const auto& path : stats.path_stats) {
    if (path.gib_per_second <= 0.0) {
      continue;
    }
    if (path.relay_device == kHostDevice) {
      if (path.direction == TransferDirection::H2D) {
        planner_profile_.direct_h2d_bw_gbps =
            alpha * path.gib_per_second +
            (1.0 - alpha) * planner_profile_.direct_h2d_bw_gbps;
      } else {
        planner_profile_.direct_d2h_bw_gbps =
            alpha * path.gib_per_second +
            (1.0 - alpha) * planner_profile_.direct_d2h_bw_gbps;
      }
      continue;
    }
    auto relay_it = std::find_if(
        planner_profile_.relays.begin(), planner_profile_.relays.end(),
        [&](const RelayProfile& relay) {
          return relay.relay_device == path.relay_device;
        });
    if (relay_it == planner_profile_.relays.end()) {
      continue;
    }
    if (path.direction == TransferDirection::H2D) {
      relay_it->h2d_bw_gbps =
          alpha * path.gib_per_second + (1.0 - alpha) * relay_it->h2d_bw_gbps;
      relay_it->effective_bw_gbps =
          alpha * path.gib_per_second +
          (1.0 - alpha) * relay_it->effective_bw_gbps;
    } else {
      relay_it->d2h_bw_gbps =
          alpha * path.gib_per_second + (1.0 - alpha) * relay_it->d2h_bw_gbps;
      relay_it->effective_d2h_bw_gbps =
          alpha * path.gib_per_second +
          (1.0 - alpha) * relay_it->effective_d2h_bw_gbps;
    }
  }
}

}  // namespace turbobus
