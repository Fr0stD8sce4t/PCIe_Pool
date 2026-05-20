#pragma once

#include <vector>

#include "turbobus/executor.h"
#include "turbobus/dummy_compute.h"
#include "turbobus/planner.h"
#include "turbobus/profiler.h"
#include "turbobus/topology.h"
#include "turbobus/types.h"

namespace turbobus {

class TurboBusRuntime {
 public:
  explicit TurboBusRuntime(RuntimeOptions options = {});
  ~TurboBusRuntime();

  void Init(int target_device, const std::vector<int>& relay_devices);
  ProfileResult Profile(std::size_t bytes = 256ull * 1024ull * 1024ull,
                        bool force = false);
  void SetTransferMode(TransferMode mode);
  TransferHandle FetchToGpu(void* host_ptr, void* target_gpu_ptr, std::size_t bytes);
  TransferHandle OffloadToCpu(void* target_gpu_ptr, void* host_ptr, std::size_t bytes);
  TransferHandle FetchRangesToGpu(void* host_ptr, std::size_t host_bytes,
                                  void* target_gpu_ptr, std::size_t target_bytes,
                                  const std::vector<TransferRange>& ranges);
  TransferHandle OffloadRangesToCpu(void* target_gpu_ptr, std::size_t target_bytes,
                                    void* host_ptr, std::size_t host_bytes,
                                    const std::vector<TransferRange>& ranges);
  DummyComputeStats RunDummyCompute(void* device_ptr, std::size_t elements,
                                    int iterations);
  void Wait(const TransferHandle& handle);
  TransferStats GetStats(const TransferHandle& handle) const;
  const ProfileResult& CachedProfile() const;
  const TransferPlan& LastPlan() const;
  const ProfileResult& PlannerProfile() const;

 private:
  void EnsureProfile();
  TransferHandle SubmitTransfer(void* source_ptr, void* destination_ptr,
                                std::size_t bytes,
                                TransferDirection direction);
  TransferHandle SubmitRanges(void* source_ptr, std::size_t source_bytes,
                              void* destination_ptr,
                              std::size_t destination_bytes,
                              const std::vector<TransferRange>& ranges,
                              TransferDirection direction);
  void UpdateDynamicWeights(const TransferStats& stats);

  RuntimeOptions options_;
  int target_device_ = 0;
  std::vector<int> requested_relays_;
  std::vector<int> enabled_relays_;
  Topology topology_;
  ProfileResult profile_;
  ProfileResult planner_profile_;
  TransferPlan last_plan_;
  bool has_profile_ = false;
  bool initialized_ = false;

  TopologyManager topology_manager_;
  BandwidthProfiler profiler_;
  ChunkPlanner planner_;
  CudaRelayExecutor executor_;
};

}  // namespace turbobus
