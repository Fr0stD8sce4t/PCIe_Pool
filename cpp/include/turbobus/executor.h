#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "turbobus/types.h"

namespace turbobus {

class CudaRelayExecutor {
 public:
  CudaRelayExecutor();
  ~CudaRelayExecutor();

  CudaRelayExecutor(const CudaRelayExecutor&) = delete;
  CudaRelayExecutor& operator=(const CudaRelayExecutor&) = delete;

  void Init(int target_device, const std::vector<int>& relay_devices,
            const RuntimeOptions& options);

  TransferHandle Submit(const BufferView& host, const BufferView& target,
                        const TransferPlan& plan);

  void Wait(const TransferHandle& handle);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace turbobus
