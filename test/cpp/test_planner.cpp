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
  profile.direct_d2h_bw_gbps = 10.0;

  turbobus::RelayProfile relay;
  relay.relay_device = 1;
  relay.target_device = 0;
  relay.h2d_bw_gbps = 40.0;
  relay.d2h_bw_gbps = 30.0;
  relay.p2p_bw_gbps = 100.0;
  relay.effective_bw_gbps = 40.0;
  relay.effective_d2h_bw_gbps = 30.0;
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
  assert(handle.stats.direct_bytes == 0);
  assert(handle.stats.relay_bytes == 0);
  assert(handle.stats.cuda_elapsed_ms == 0.0);
  assert(handle.stats.direct_chunks == 0);
  assert(handle.stats.relay_chunks == 0);
  assert(handle.stats.relay_devices.empty());
  assert(handle.stats.path_stats.empty());

  const auto small_plan = planner.Plan(4ull * 1024ull * 1024ull, chunk_bytes, profile);
  assert(small_plan.assignments.size() == 1);
  assert(small_plan.assignments.front().path.kind == turbobus::PathKind::DirectH2D);
  assert(small_plan.assignments.front().chunks.size() == 1);

  const auto direct_plan =
      planner.Plan(total_bytes, chunk_bytes, profile, turbobus::TransferMode::DirectOnly);
  assert(direct_plan.assignments.size() == 1);
  assert(direct_plan.assignments.front().path.kind == turbobus::PathKind::DirectH2D);

  const auto relay_plan =
      planner.Plan(total_bytes, chunk_bytes, profile, turbobus::TransferMode::RelayOnly);
  assert(relay_plan.assignments.size() == 1);
  assert(relay_plan.assignments.front().path.kind ==
         turbobus::PathKind::RelayH2DThenP2P);

  turbobus::ProfileResult slow_relay_profile = profile;
  slow_relay_profile.relays.front().h2d_bw_gbps = 5.0;
  slow_relay_profile.relays.front().effective_bw_gbps = 5.0;
  const auto filtered_plan =
      planner.Plan(total_bytes, chunk_bytes, slow_relay_profile,
                   turbobus::TransferMode::Pool, 2, 0.0, 0.9);
  assert(filtered_plan.assignments.size() == 1);
  assert(filtered_plan.assignments.front().path.kind == turbobus::PathKind::DirectH2D);

  const auto d2h_plan =
      planner.Plan(total_bytes, chunk_bytes, profile, turbobus::TransferMode::Pool,
                   2, 0.0, 0.0, turbobus::TransferDirection::D2H);
  assert(d2h_plan.assignments.size() == 2);
  bool has_direct_d2h = false;
  bool has_relay_d2h = false;
  for (const auto& assignment : d2h_plan.assignments) {
    assert(assignment.path.direction == turbobus::TransferDirection::D2H);
    if (assignment.path.kind == turbobus::PathKind::DirectD2H) {
      has_direct_d2h = true;
      assert(assignment.path.effective_bw_gbps == 10.0);
    }
    if (assignment.path.kind == turbobus::PathKind::RelayP2PThenD2H) {
      has_relay_d2h = true;
      assert(assignment.path.effective_bw_gbps == 30.0);
    }
  }
  assert(has_direct_d2h);
  assert(has_relay_d2h);

  std::vector<turbobus::TransferRange> ranges;
  ranges.push_back({0, 64, 10});
  ranges.push_back({1024, 4096, 25});
  const auto range_plan = planner.PlanRanges(ranges, 8, profile);
  assert(range_plan.total_bytes == 35);
  std::size_t range_bytes = 0;
  bool saw_non_identity_offset = false;
  for (const auto& assignment : range_plan.assignments) {
    for (const auto& chunk : assignment.chunks) {
      range_bytes += chunk.bytes;
      if (chunk.src_offset != chunk.dst_offset) {
        saw_non_identity_offset = true;
      }
      assert(chunk.bytes <= 8);
    }
  }
  assert(range_bytes == 35);
  assert(saw_non_identity_offset);

  std::cout << "planner test passed\n";
  return 0;
}
