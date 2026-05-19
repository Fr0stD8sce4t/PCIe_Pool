#include "turbobus/executor.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace turbobus {

namespace {

void CheckCuda(cudaError_t result, const char* message) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " +
                             cudaGetErrorString(result));
  }
}

void IgnoreCuda(cudaError_t result) {
  (void)result;
}

double Gbps(std::size_t bytes, double milliseconds) {
  if (milliseconds <= 0.0) {
    return 0.0;
  }
  const double seconds = milliseconds / 1000.0;
  const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  return gib / seconds;
}

}  // namespace

struct CudaRelayExecutor::Impl {
  struct RelayState {
    int relay_device = kHostDevice;
    cudaStream_t h2d_stream = nullptr;
    cudaStream_t p2p_stream = nullptr;
    std::vector<void*> staging_slots;
    std::vector<cudaEvent_t> h2d_done_events;
    std::vector<cudaEvent_t> p2p_done_events;
    std::vector<bool> slot_has_pending_p2p;
  };

  int target_device = 0;
  RuntimeOptions options;
  cudaStream_t direct_stream = nullptr;
  std::unordered_map<int, RelayState> relays;
  std::atomic<std::uint64_t> next_id{1};
  std::unordered_map<std::uint64_t, std::vector<cudaEvent_t>> transfer_events;
  std::unordered_map<std::uint64_t, std::chrono::steady_clock::time_point> start_times;
  std::unordered_map<std::uint64_t, TransferStats> completed_stats;

  void Destroy() noexcept {
    for (auto& item : transfer_events) {
      for (auto event : item.second) {
        if (event != nullptr) {
          IgnoreCuda(cudaEventDestroy(event));
        }
      }
    }
    transfer_events.clear();
    start_times.clear();
    completed_stats.clear();

    if (direct_stream != nullptr) {
      IgnoreCuda(cudaStreamDestroy(direct_stream));
      direct_stream = nullptr;
    }

    for (auto& [_, relay] : relays) {
      for (auto event : relay.h2d_done_events) {
        if (event != nullptr) {
          IgnoreCuda(cudaEventDestroy(event));
        }
      }
      for (auto event : relay.p2p_done_events) {
        if (event != nullptr) {
          IgnoreCuda(cudaEventDestroy(event));
        }
      }
      if (relay.h2d_stream != nullptr) {
        IgnoreCuda(cudaStreamDestroy(relay.h2d_stream));
      }
      if (relay.p2p_stream != nullptr) {
        IgnoreCuda(cudaStreamDestroy(relay.p2p_stream));
      }
      IgnoreCuda(cudaSetDevice(relay.relay_device));
      for (void* slot : relay.staging_slots) {
        if (slot != nullptr) {
          IgnoreCuda(cudaFree(slot));
        }
      }
    }
    relays.clear();
  }
};

CudaRelayExecutor::CudaRelayExecutor() : impl_(new Impl()) {}

CudaRelayExecutor::~CudaRelayExecutor() {
  if (impl_ != nullptr) {
    impl_->Destroy();
    delete impl_;
    impl_ = nullptr;
  }
}

void CudaRelayExecutor::Init(int target_device, const std::vector<int>& relay_devices,
                             const RuntimeOptions& options) {
  impl_->Destroy();
  impl_->target_device = target_device;
  impl_->options = options;

  if (options.chunk_bytes == 0) {
    throw std::invalid_argument("chunk_bytes must be greater than zero");
  }
  if (options.staging_slots <= 0) {
    throw std::invalid_argument("staging_slots must be greater than zero");
  }

  try {
    CheckCuda(cudaSetDevice(target_device), "cudaSetDevice target failed");
    CheckCuda(cudaStreamCreateWithFlags(&impl_->direct_stream, cudaStreamNonBlocking),
              "cudaStreamCreate direct failed");

    for (const int relay_device : relay_devices) {
      if (relay_device == target_device) {
        continue;
      }

      Impl::RelayState relay;
      relay.relay_device = relay_device;
      relay.staging_slots.resize(static_cast<std::size_t>(options.staging_slots), nullptr);
      relay.h2d_done_events.resize(static_cast<std::size_t>(options.staging_slots), nullptr);
      relay.p2p_done_events.resize(static_cast<std::size_t>(options.staging_slots), nullptr);
      relay.slot_has_pending_p2p.resize(static_cast<std::size_t>(options.staging_slots), false);

      CheckCuda(cudaSetDevice(relay_device), "cudaSetDevice relay failed");
      CheckCuda(cudaStreamCreateWithFlags(&relay.h2d_stream, cudaStreamNonBlocking),
                "cudaStreamCreate relay h2d failed");
      CheckCuda(cudaStreamCreateWithFlags(&relay.p2p_stream, cudaStreamNonBlocking),
                "cudaStreamCreate relay p2p failed");

      for (int i = 0; i < options.staging_slots; ++i) {
        CheckCuda(cudaMalloc(&relay.staging_slots[static_cast<std::size_t>(i)],
                             options.chunk_bytes),
                  "cudaMalloc relay staging failed");
        CheckCuda(cudaEventCreateWithFlags(&relay.h2d_done_events[static_cast<std::size_t>(i)],
                                           cudaEventDisableTiming),
                  "cudaEventCreate relay h2d_done failed");
        CheckCuda(cudaEventCreateWithFlags(&relay.p2p_done_events[static_cast<std::size_t>(i)],
                                           cudaEventDisableTiming),
                  "cudaEventCreate relay p2p_done failed");
      }

      impl_->relays.emplace(relay_device, std::move(relay));
    }

    CheckCuda(cudaSetDevice(target_device), "cudaSetDevice restore target failed");
  } catch (...) {
    impl_->Destroy();
    throw;
  }
}

TransferHandle CudaRelayExecutor::Submit(const BufferView& host, const BufferView& target,
                                         const TransferPlan& plan) {
  if (host.ptr == nullptr || target.ptr == nullptr) {
    throw std::invalid_argument("host and target pointers must not be null");
  }
  if (host.kind != MemoryKind::HostPinned) {
    throw std::invalid_argument("host buffer must be pinned host memory");
  }
  if (target.kind != MemoryKind::Device || target.device != impl_->target_device) {
    throw std::invalid_argument("target buffer must be on target_device");
  }
  if (host.bytes < plan.total_bytes || target.bytes < plan.total_bytes) {
    throw std::invalid_argument("buffer is smaller than transfer plan");
  }

  const auto* host_bytes = static_cast<const char*>(host.ptr);
  auto* target_bytes = static_cast<char*>(target.ptr);
  const std::uint64_t id = impl_->next_id.fetch_add(1);
  std::vector<cudaEvent_t> completion_events;
  TransferStats stats;
  stats.bytes = plan.total_bytes;
  for (const auto& assignment : plan.assignments) {
    if (assignment.path.kind == PathKind::DirectH2D) {
      stats.direct_chunks += assignment.chunks.size();
    } else {
      stats.relay_chunks += assignment.chunks.size();
    }
  }

  try {
    for (const auto& assignment : plan.assignments) {
      if (assignment.path.kind == PathKind::DirectH2D) {
        CheckCuda(cudaSetDevice(impl_->target_device), "cudaSetDevice direct failed");
        for (const auto& chunk : assignment.chunks) {
          CheckCuda(cudaMemcpyAsync(target_bytes + chunk.dst_offset,
                                    host_bytes + chunk.src_offset, chunk.bytes,
                                    cudaMemcpyHostToDevice, impl_->direct_stream),
                    "direct cudaMemcpyAsync failed");
        }
        cudaEvent_t completion = nullptr;
        CheckCuda(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming),
                  "cudaEventCreate direct completion failed");
        CheckCuda(cudaEventRecord(completion, impl_->direct_stream),
                  "cudaEventRecord direct completion failed");
        completion_events.push_back(completion);
        continue;
      }

      auto relay_it = impl_->relays.find(assignment.path.relay_device);
      if (relay_it == impl_->relays.end()) {
        throw std::runtime_error("transfer plan references an uninitialized relay");
      }

      auto& relay = relay_it->second;
      CheckCuda(cudaSetDevice(relay.relay_device), "cudaSetDevice relay submit failed");
      std::size_t slot_index = 0;
      for (const auto& chunk : assignment.chunks) {
        if (chunk.bytes > impl_->options.chunk_bytes) {
          throw std::runtime_error("chunk is larger than relay staging slot");
        }
        slot_index %= relay.staging_slots.size();
        void* slot = relay.staging_slots[slot_index];
        cudaEvent_t h2d_done = relay.h2d_done_events[slot_index];
        cudaEvent_t p2p_done = relay.p2p_done_events[slot_index];

        if (relay.slot_has_pending_p2p[slot_index]) {
          CheckCuda(cudaStreamWaitEvent(relay.h2d_stream, p2p_done, 0),
                    "cudaStreamWaitEvent staging slot reuse failed");
        }

        CheckCuda(cudaMemcpyAsync(slot, host_bytes + chunk.src_offset, chunk.bytes,
                                  cudaMemcpyHostToDevice, relay.h2d_stream),
                  "relay h2d cudaMemcpyAsync failed");
        CheckCuda(cudaEventRecord(h2d_done, relay.h2d_stream),
                  "cudaEventRecord relay h2d_done failed");

        CheckCuda(cudaStreamWaitEvent(relay.p2p_stream, h2d_done, 0),
                  "cudaStreamWaitEvent relay p2p failed");
        CheckCuda(cudaMemcpyPeerAsync(target_bytes + chunk.dst_offset,
                                      impl_->target_device, slot, relay.relay_device,
                                      chunk.bytes, relay.p2p_stream),
                  "relay p2p cudaMemcpyPeerAsync failed");
        CheckCuda(cudaEventRecord(p2p_done, relay.p2p_stream),
                  "cudaEventRecord relay p2p_done failed");
        relay.slot_has_pending_p2p[slot_index] = true;
        ++slot_index;
      }
      cudaEvent_t completion = nullptr;
      CheckCuda(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming),
                "cudaEventCreate relay completion failed");
      CheckCuda(cudaEventRecord(completion, relay.p2p_stream),
                "cudaEventRecord relay completion failed");
      completion_events.push_back(completion);
    }

    impl_->transfer_events.emplace(id, std::move(completion_events));
    impl_->start_times.emplace(id, std::chrono::steady_clock::now());
    impl_->completed_stats.emplace(id, stats);
  } catch (...) {
    for (auto event : completion_events) {
      IgnoreCuda(cudaEventDestroy(event));
    }
    throw;
  }
  TransferHandle handle;
  handle.id = id;
  handle.status = TransferStatus::Submitted;
  handle.stats = stats;
  return handle;
}

void CudaRelayExecutor::Wait(const TransferHandle& handle) {
  const auto event_it = impl_->transfer_events.find(handle.id);
  if (event_it == impl_->transfer_events.end()) {
    throw std::invalid_argument("unknown transfer handle");
  }
  for (auto event : event_it->second) {
    CheckCuda(cudaEventSynchronize(event), "cudaEventSynchronize failed");
    CheckCuda(cudaEventDestroy(event), "cudaEventDestroy completion failed");
  }
  const auto end = std::chrono::steady_clock::now();
  const auto start_it = impl_->start_times.find(handle.id);
  const auto stats_it = impl_->completed_stats.find(handle.id);
  if (start_it != impl_->start_times.end() && stats_it != impl_->completed_stats.end()) {
    const auto microseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start_it->second)
            .count();
    stats_it->second.submit_to_complete_ms = static_cast<double>(microseconds) / 1000.0;
    stats_it->second.gib_per_second =
        Gbps(stats_it->second.bytes, stats_it->second.submit_to_complete_ms);
    impl_->start_times.erase(start_it);
  }
  impl_->transfer_events.erase(event_it);
}

TransferStats CudaRelayExecutor::GetStats(const TransferHandle& handle) const {
  const auto stats_it = impl_->completed_stats.find(handle.id);
  if (stats_it == impl_->completed_stats.end()) {
    throw std::invalid_argument("unknown transfer handle");
  }
  return stats_it->second;
}

}  // namespace turbobus
