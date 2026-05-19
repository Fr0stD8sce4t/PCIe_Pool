#include "turbobus/planner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <stdexcept>

namespace turbobus {

std::vector<Path> ChunkPlanner::BuildPaths(const ProfileResult& profile) const {
  std::vector<Path> paths;

  if (profile.direct_h2d_bw_gbps > 0.0) {
    Path direct;
    direct.kind = PathKind::DirectH2D;
    direct.target_device = profile.target_device;
    direct.relay_device = kHostDevice;
    direct.h2d_bw_gbps = profile.direct_h2d_bw_gbps;
    direct.p2p_bw_gbps = 0.0;
    direct.effective_bw_gbps = profile.direct_h2d_bw_gbps;
    direct.enabled = true;
    paths.push_back(direct);
  }

  for (const auto& relay : profile.relays) {
    if (!relay.p2p_enabled || relay.effective_bw_gbps <= 0.0) {
      continue;
    }
    Path path;
    path.kind = PathKind::RelayH2DThenP2P;
    path.target_device = relay.target_device;
    path.relay_device = relay.relay_device;
    path.h2d_bw_gbps = relay.h2d_bw_gbps;
    path.p2p_bw_gbps = relay.p2p_bw_gbps;
    path.effective_bw_gbps = relay.effective_bw_gbps;
    path.enabled = true;
    paths.push_back(path);
  }

  return paths;
}

TransferPlan ChunkPlanner::Plan(std::size_t total_bytes, std::size_t chunk_bytes,
                                const ProfileResult& profile) const {
  if (total_bytes == 0) {
    return {};
  }
  if (chunk_bytes == 0) {
    throw std::invalid_argument("chunk_bytes must be greater than zero");
  }

  auto paths = BuildPaths(profile);
  if (paths.empty()) {
    throw std::runtime_error("no enabled transfer path is available");
  }

  const double total_bw = std::accumulate(
      paths.begin(), paths.end(), 0.0,
      [](double sum, const Path& path) { return sum + path.effective_bw_gbps; });
  if (total_bw <= 0.0) {
    throw std::runtime_error("enabled paths have zero effective bandwidth");
  }

  TransferPlan plan;
  plan.total_bytes = total_bytes;
  plan.chunk_bytes = chunk_bytes;
  plan.assignments.reserve(paths.size());
  for (const auto& path : paths) {
    PathAssignment assignment;
    assignment.path = path;
    plan.assignments.push_back(std::move(assignment));
  }

  const std::size_t num_chunks = (total_bytes + chunk_bytes - 1) / chunk_bytes;
  std::vector<double> assigned_scores(paths.size(), 0.0);

  for (std::size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const std::size_t offset = chunk_idx * chunk_bytes;
    const std::size_t bytes = std::min(chunk_bytes, total_bytes - offset);

    std::size_t selected = 0;
    double best_score = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < paths.size(); ++i) {
      const double score = assigned_scores[i] / paths[i].effective_bw_gbps;
      if (score < best_score) {
        best_score = score;
        selected = i;
      }
    }

    Chunk chunk;
    chunk.src_offset = offset;
    chunk.dst_offset = offset;
    chunk.bytes = bytes;
    plan.assignments[selected].chunks.push_back(chunk);
    assigned_scores[selected] += static_cast<double>(bytes);
  }

  plan.assignments.erase(
      std::remove_if(plan.assignments.begin(), plan.assignments.end(),
                     [](const PathAssignment& assignment) {
                       return assignment.chunks.empty();
                     }),
      plan.assignments.end());
  return plan;
}

}  // namespace turbobus
