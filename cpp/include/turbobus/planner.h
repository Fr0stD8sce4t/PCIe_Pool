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
                    std::size_t min_chunks_for_relay = 2) const;

 private:
  std::vector<Path> BuildPaths(const ProfileResult& profile, TransferMode mode) const;
};

}  // namespace turbobus
