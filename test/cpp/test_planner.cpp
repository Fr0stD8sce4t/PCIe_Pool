#include <cassert>
#include <cstddef>
#include <iostream>

#include "turbobus/planner.h"

namespace {

std::size_t AssignedBytes(const turbobus::PathAssignment& assignment) {
  std::size_t bytes = 0;
  for (const auto& chunk : assignment.chunks) {
    bytes += chunk.bytes;
  }
  return bytes;
}

}  // namespace

int main() {
  turbobus::ProfileResult profile;
  profile.target_device = 0;
  profile.direct_h2d_bw_gbps = 20.0;

  turbobus::RelayProfile relay;
  relay.relay_device = 1;
  relay.target_device = 0;
  relay.h2d_bw_gbps = 40.0;
  relay.p2p_bw_gbps = 100.0;
  relay.effective_bw_gbps = 40.0;
  relay.p2p_enabled = true;
  profile.relays.push_back(relay);

  turbobus::ChunkPlanner planner;
  const std::size_t total_bytes = 128ull * 1024ull * 1024ull;
  const std::size_t chunk_bytes = 16ull * 1024ull * 1024ull;
  const auto plan = planner.Plan(total_bytes, chunk_bytes, profile);

  assert(plan.total_bytes == total_bytes);
  assert(plan.chunk_bytes == chunk_bytes);
  assert(plan.assignments.size() == 2);

  std::size_t direct_bytes = 0;
  std::size_t relay_bytes = 0;
  std::size_t all_bytes = 0;
  for (const auto& assignment : plan.assignments) {
    const auto bytes = AssignedBytes(assignment);
    all_bytes += bytes;
    if (assignment.path.kind == turbobus::PathKind::DirectH2D) {
      direct_bytes += bytes;
    } else {
      relay_bytes += bytes;
    }
  }

  assert(all_bytes == total_bytes);
  assert(direct_bytes > 0);
  assert(relay_bytes > 0);
  assert(relay_bytes >= direct_bytes);

  turbobus::TransferHandle handle;
  assert(handle.stats.bytes == 0);
  assert(handle.stats.direct_chunks == 0);
  assert(handle.stats.relay_chunks == 0);

  std::cout << "planner test passed\n";
  return 0;
}
