#pragma once

#include <cstddef>
#include <vector>

#include "turbobus/types.h"

namespace turbobus {

class BandwidthProfiler {
 public:
  ProfileResult Profile(int target_device, const std::vector<int>& relay_devices,
                        std::size_t bytes = 256ull * 1024ull * 1024ull);

 private:
  double MeasureH2D(int device, std::size_t bytes);
  double MeasureD2H(int device, std::size_t bytes);
  double MeasureP2P(int src_device, int dst_device, std::size_t bytes);
};

}  // namespace turbobus
