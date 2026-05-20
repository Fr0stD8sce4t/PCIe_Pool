#pragma once

#include <cstddef>
#include <vector>

#include "turbobus/types.h"

namespace turbobus {

class ChunkPlanner {
 public:
  TransferPlan Plan(std::size_t total_bytes, std::size_t chunk_bytes,
                    const ProfileResult& profile,
                    TransferMode mode = TransferMode::Pool,
                    std::size_t min_chunks_for_relay = 2,
                    double relay_min_effective_bw_gbps = 0.0,
                    double relay_min_direct_ratio = 0.0,
                    TransferDirection direction = TransferDirection::H2D) const;

 private:
  std::vector<Path> BuildPaths(const ProfileResult& profile, TransferMode mode,
                               double relay_min_effective_bw_gbps,
                               double relay_min_direct_ratio,
                               TransferDirection direction) const;
};

}  // namespace turbobus
