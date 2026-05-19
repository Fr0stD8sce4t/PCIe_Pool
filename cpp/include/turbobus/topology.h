#pragma once

#include <vector>

namespace turbobus {

struct DeviceInfo {
  int device_id = 0;
  bool can_access_target = false;
  bool peer_access_enabled = false;
};

struct Topology {
  int target_device = 0;
  std::vector<DeviceInfo> devices;
  std::vector<DeviceInfo> relay_candidates;
};

class TopologyManager {
 public:
  Topology Discover(int target_device, const std::vector<int>& relay_devices,
                    bool enable_peer_access);
};

}  // namespace turbobus

