#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

namespace {

void CheckCuda(cudaError_t result, const char* message) {
  if (result != cudaSuccess) {
    std::cerr << message << ": " << cudaGetErrorString(result) << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed");
  std::cout << "CUDA P2P access matrix, row=relay/source, col=target/destination\n";
  std::cout << "    ";
  for (int dst = 0; dst < device_count; ++dst) {
    std::cout << "GPU" << dst << " ";
  }
  std::cout << "\n";

  for (int src = 0; src < device_count; ++src) {
    std::cout << "GPU" << src << " ";
    for (int dst = 0; dst < device_count; ++dst) {
      if (src == dst) {
        std::cout << "  X  ";
        continue;
      }
      int can_access = 0;
      CheckCuda(cudaDeviceCanAccessPeer(&can_access, src, dst),
                "cudaDeviceCanAccessPeer failed");
      std::cout << "  " << can_access << "  ";
    }
    std::cout << "\n";
  }
  return 0;
}
