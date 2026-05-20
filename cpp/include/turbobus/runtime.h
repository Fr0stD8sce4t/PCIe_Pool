#pragma once

#include <vector>

#include "turbobus/executor.h"
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
  void Wait(const TransferHandle& handle);
  TransferStats GetStats(const TransferHandle& handle) const;
  const ProfileResult& CachedProfile() const;
  const TransferPlan& LastPlan() const;

 private:
  RuntimeOptions options_;
  int target_device_ = 0;
  std::vector<int> requested_relays_;
  std::vector<int> enabled_relays_;
  Topology topology_;
  ProfileResult profile_;
  TransferPlan last_plan_;
  bool has_profile_ = false;
  bool initialized_ = false;

  TopologyManager topology_manager_;
  BandwidthProfiler profiler_;
  ChunkPlanner planner_;
  CudaRelayExecutor executor_;
};

}  // namespace turbobus
