#include "turbobus/profiler.h"

#include <cuda_runtime.h>

#include <algorithm>
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

double ElapsedGbps(float milliseconds, std::size_t bytes) {
  if (milliseconds <= 0.0f) {
    return 0.0;
  }
  const double seconds = static_cast<double>(milliseconds) / 1000.0;
  const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  return gib / seconds;
}

}  // namespace

double BandwidthProfiler::MeasureH2D(int device, std::size_t bytes) {
  void* host = nullptr;
  void* dst = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  CheckCuda(cudaSetDevice(device), "cudaSetDevice h2d profile failed");
  CheckCuda(cudaMallocHost(&host, bytes), "cudaMallocHost profile failed");
  CheckCuda(cudaMalloc(&dst, bytes), "cudaMalloc profile dst failed");
  CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "cudaStreamCreate h2d profile failed");
  CheckCuda(cudaEventCreate(&start), "cudaEventCreate start failed");
  CheckCuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");

  try {
    CheckCuda(cudaEventRecord(start, stream), "cudaEventRecord h2d start failed");
    CheckCuda(cudaMemcpyAsync(dst, host, bytes, cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync h2d profile failed");
    CheckCuda(cudaEventRecord(stop, stream), "cudaEventRecord h2d stop failed");
    CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize h2d failed");
  } catch (...) {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(dst);
    cudaFreeHost(host);
    throw;
  }

  float milliseconds = 0.0f;
  CheckCuda(cudaEventElapsedTime(&milliseconds, start, stop),
            "cudaEventElapsedTime h2d failed");

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaStreamDestroy(stream);
  cudaFree(dst);
  cudaFreeHost(host);
  return ElapsedGbps(milliseconds, bytes);
}

double BandwidthProfiler::MeasureP2P(int src_device, int dst_device, std::size_t bytes) {
  void* src = nullptr;
  void* dst = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  CheckCuda(cudaSetDevice(src_device), "cudaSetDevice p2p src failed");
  CheckCuda(cudaMalloc(&src, bytes), "cudaMalloc p2p src failed");
  CheckCuda(cudaSetDevice(dst_device), "cudaSetDevice p2p dst failed");
  CheckCuda(cudaMalloc(&dst, bytes), "cudaMalloc p2p dst failed");
  CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "cudaStreamCreate p2p profile failed");
  CheckCuda(cudaEventCreate(&start), "cudaEventCreate p2p start failed");
  CheckCuda(cudaEventCreate(&stop), "cudaEventCreate p2p stop failed");

  try {
    CheckCuda(cudaEventRecord(start, stream), "cudaEventRecord p2p start failed");
    CheckCuda(cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, stream),
              "cudaMemcpyPeerAsync p2p profile failed");
    CheckCuda(cudaEventRecord(stop, stream), "cudaEventRecord p2p stop failed");
    CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize p2p failed");
  } catch (...) {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaSetDevice(dst_device);
    cudaFree(dst);
    cudaSetDevice(src_device);
    cudaFree(src);
    throw;
  }

  float milliseconds = 0.0f;
  CheckCuda(cudaEventElapsedTime(&milliseconds, start, stop),
            "cudaEventElapsedTime p2p failed");

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaStreamDestroy(stream);
  cudaSetDevice(dst_device);
  cudaFree(dst);
  cudaSetDevice(src_device);
  cudaFree(src);
  return ElapsedGbps(milliseconds, bytes);
}

ProfileResult BandwidthProfiler::Profile(int target_device,
                                         const std::vector<int>& relay_devices,
                                         std::size_t bytes) {
  if (bytes == 0) {
    throw std::invalid_argument("profile bytes must be greater than zero");
  }

  ProfileResult result;
  result.target_device = target_device;
  result.direct_h2d_bw_gbps = MeasureH2D(target_device, bytes);

  for (const int relay_device : relay_devices) {
    if (relay_device == target_device) {
      continue;
    }
    int can_access = 0;
    CheckCuda(cudaDeviceCanAccessPeer(&can_access, relay_device, target_device),
              "cudaDeviceCanAccessPeer profile failed");

    RelayProfile relay;
    relay.relay_device = relay_device;
    relay.target_device = target_device;
    relay.p2p_enabled = can_access != 0;
    relay.h2d_bw_gbps = MeasureH2D(relay_device, bytes);
    if (relay.p2p_enabled) {
      relay.p2p_bw_gbps = MeasureP2P(relay_device, target_device, bytes);
      relay.effective_bw_gbps = std::min(relay.h2d_bw_gbps, relay.p2p_bw_gbps);
    }
    result.relays.push_back(relay);
  }
  return result;
}

}  // namespace turbobus
