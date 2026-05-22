#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace turbobus {

constexpr int kHostDevice = -1;
constexpr std::size_t kDefaultChunkBytes = 16ull * 1024ull * 1024ull;
constexpr int kDefaultStagingSlots = 2;

enum class MemoryKind {
  HostPinned,
  Device,
};

enum class PathKind {
  DirectH2D,
  RelayH2DThenP2P,
  DirectD2H,
  RelayP2PThenD2H,
};

enum class TransferDirection {
  H2D,
  D2H,
};

enum class TransferStatus {
  Pending,
  Submitted,
  Complete,
  Failed,
};

enum class TransferMode {
  Pool,
  DirectOnly,
  RelayOnly,
};

struct BufferView {
  void* ptr = nullptr;
  std::size_t bytes = 0;
  MemoryKind kind = MemoryKind::HostPinned;
  int device = kHostDevice;
};

struct Chunk {
  std::size_t src_offset = 0;
  std::size_t dst_offset = 0;
  std::size_t bytes = 0;
};

struct TransferRange {
  std::size_t src_offset = 0;
  std::size_t dst_offset = 0;
  std::size_t bytes = 0;
};

struct Path {
  PathKind kind = PathKind::DirectH2D;
  TransferDirection direction = TransferDirection::H2D;
  int target_device = 0;
  int relay_device = kHostDevice;
  double h2d_bw_gbps = 0.0;
  double d2h_bw_gbps = 0.0;
  double p2p_bw_gbps = 0.0;
  double effective_bw_gbps = 0.0;
  bool enabled = true;
};

struct PathAssignment {
  Path path;
  std::vector<Chunk> chunks;
};

struct TransferPlan {
  std::size_t total_bytes = 0;
  std::size_t chunk_bytes = kDefaultChunkBytes;
  std::vector<PathAssignment> assignments;
};

struct RelayProfile {
  int relay_device = kHostDevice;
  int target_device = 0;
  double h2d_bw_gbps = 0.0;
  double d2h_bw_gbps = 0.0;
  double p2p_bw_gbps = 0.0;
  double effective_bw_gbps = 0.0;
  double effective_d2h_bw_gbps = 0.0;
  bool p2p_enabled = false;
};

struct ProfileResult {
  int target_device = 0;
  double direct_h2d_bw_gbps = 0.0;
  double direct_d2h_bw_gbps = 0.0;
  std::vector<RelayProfile> relays;
};

struct RuntimeOptions {
  std::size_t chunk_bytes = kDefaultChunkBytes;
  int staging_slots = kDefaultStagingSlots;
  bool enable_peer_access = true;
  std::size_t profile_bytes = 256ull * 1024ull * 1024ull;
  bool profile_on_first_transfer = true;
  bool profile_cache_enabled = true;
  TransferMode transfer_mode = TransferMode::Pool;
  std::size_t min_chunks_for_relay = 2;
  double relay_min_effective_bw_gbps = 0.0;
  double relay_min_direct_ratio = 0.0;
  bool enable_dynamic_weights = false;
  double dynamic_weight_alpha = 0.25;
};

struct PathStats {
  PathKind kind = PathKind::DirectH2D;
  TransferDirection direction = TransferDirection::H2D;
  int target_device = 0;
  int relay_device = kHostDevice;
  std::size_t bytes = 0;
  std::size_t chunks = 0;
  double cuda_elapsed_ms = 0.0;
  double gib_per_second = 0.0;
};

struct TransferStats {
  std::size_t bytes = 0;
  std::size_t direct_bytes = 0;
  std::size_t relay_bytes = 0;
  double submit_to_complete_ms = 0.0;
  double cuda_elapsed_ms = 0.0;
  double gib_per_second = 0.0;
  double submit_gib_per_second = 0.0;
  std::size_t direct_chunks = 0;
  std::size_t relay_chunks = 0;
  std::vector<int> relay_devices;
  std::vector<std::size_t> relay_device_bytes;
  std::vector<std::size_t> relay_device_chunks;
  std::vector<PathStats> path_stats;
};

struct TransferHandle {
  std::uint64_t id = 0;
  TransferStatus status = TransferStatus::Pending;
  std::string error;
  TransferStats stats;
};

struct DummyComputeStats {
  std::size_t elements = 0;
  int iterations = 0;
  double cuda_elapsed_ms = 0.0;
};

}  // namespace turbobus
