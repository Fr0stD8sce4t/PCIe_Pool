#include "turbobus/topology.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace turbobus {

namespace {

void CheckCuda(cudaError_t result, const char* message) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " +
                             cudaGetErrorString(result));
  }
}

}  // namespace

Topology TopologyManager::Discover(int target_device,
                                   const std::vector<int>& relay_devices,
                                   bool enable_peer_access) {
  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed");
  if (target_device < 0 || target_device >= device_count) {
    throw std::invalid_argument("target_device is out of range");
  }

  Topology topology;
  topology.target_device = target_device;

  for (int device = 0; device < device_count; ++device) {
    DeviceInfo info;
    info.device_id = device;
    if (device == target_device) {
      info.can_access_target = true;
      info.peer_access_enabled = true;
    } else {
      int can_access = 0;
      CheckCuda(cudaDeviceCanAccessPeer(&can_access, device, target_device),
                "cudaDeviceCanAccessPeer failed");
      info.can_access_target = can_access != 0;

      if (info.can_access_target && enable_peer_access) {
        int previous_device = 0;
        CheckCuda(cudaGetDevice(&previous_device), "cudaGetDevice failed");
        CheckCuda(cudaSetDevice(device), "cudaSetDevice relay failed");
        const cudaError_t enable_result = cudaDeviceEnablePeerAccess(target_device, 0);
        if (enable_result == cudaSuccess ||
            enable_result == cudaErrorPeerAccessAlreadyEnabled) {
          info.peer_access_enabled = true;
          if (enable_result == cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();
          }
        } else {
          CheckCuda(enable_result, "cudaDeviceEnablePeerAccess relay->target failed");
        }
        CheckCuda(cudaSetDevice(target_device), "cudaSetDevice target failed");
        const cudaError_t reverse_result = cudaDeviceEnablePeerAccess(device, 0);
        if (reverse_result == cudaSuccess ||
            reverse_result == cudaErrorPeerAccessAlreadyEnabled) {
          if (reverse_result == cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();
          }
        } else {
          CheckCuda(reverse_result, "cudaDeviceEnablePeerAccess target->relay failed");
        }
        CheckCuda(cudaSetDevice(previous_device), "cudaSetDevice restore failed");
      }
    }
    topology.devices.push_back(info);
  }

  for (const int relay_device : relay_devices) {
    if (relay_device < 0 || relay_device >= device_count) {
      throw std::invalid_argument("relay_device is out of range");
    }
    if (relay_device == target_device) {
      continue;
    }
    const auto& info = topology.devices[relay_device];
    if (info.can_access_target && (!enable_peer_access || info.peer_access_enabled)) {
      topology.relay_candidates.push_back(info);
    }
  }

  return topology;
}

}  // namespace turbobus
