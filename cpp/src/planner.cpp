#include "turbobus/planner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace turbobus {

std::vector<Path> ChunkPlanner::BuildPaths(const ProfileResult& profile,
                                           TransferMode mode,
                                           double relay_min_effective_bw_gbps,
                                           double relay_min_direct_ratio,
                                           TransferDirection direction) const {
  std::vector<Path> paths;
  const double direct_bw =
      direction == TransferDirection::H2D
          ? profile.direct_h2d_bw_gbps
          : (profile.direct_d2h_bw_gbps > 0.0 ? profile.direct_d2h_bw_gbps
                                               : profile.direct_h2d_bw_gbps);

  if (mode != TransferMode::RelayOnly && direct_bw > 0.0) {
    Path direct;
    direct.kind = direction == TransferDirection::H2D ? PathKind::DirectH2D
                                                      : PathKind::DirectD2H;
    direct.direction = direction;
    direct.target_device = profile.target_device;
    direct.relay_device = kHostDevice;
    direct.h2d_bw_gbps = profile.direct_h2d_bw_gbps;
    direct.d2h_bw_gbps = profile.direct_d2h_bw_gbps > 0.0
                              ? profile.direct_d2h_bw_gbps
                              : profile.direct_h2d_bw_gbps;
    direct.p2p_bw_gbps = 0.0;
    direct.effective_bw_gbps = direct_bw;
    direct.enabled = true;
    paths.push_back(direct);
  }

  if (mode != TransferMode::DirectOnly) {
    for (const auto& relay : profile.relays) {
      const double relay_effective_bw =
          direction == TransferDirection::H2D
              ? relay.effective_bw_gbps
              : (relay.effective_d2h_bw_gbps > 0.0
                     ? relay.effective_d2h_bw_gbps
                     : relay.effective_bw_gbps);
      if (!relay.p2p_enabled || relay_effective_bw <= 0.0) {
        continue;
      }
      if (relay_effective_bw < relay_min_effective_bw_gbps) {
        continue;
      }
      if (direct_bw > 0.0 && relay_min_direct_ratio > 0.0 &&
          relay_effective_bw < direct_bw * relay_min_direct_ratio) {
        continue;
      }
      Path path;
      path.kind = direction == TransferDirection::H2D ? PathKind::RelayH2DThenP2P
                                                      : PathKind::RelayP2PThenD2H;
      path.direction = direction;
      path.target_device = relay.target_device;
      path.relay_device = relay.relay_device;
      path.h2d_bw_gbps = relay.h2d_bw_gbps;
      path.d2h_bw_gbps =
          relay.d2h_bw_gbps > 0.0 ? relay.d2h_bw_gbps : relay.h2d_bw_gbps;
      path.p2p_bw_gbps = relay.p2p_bw_gbps;
      path.effective_bw_gbps = relay_effective_bw;
      path.enabled = true;
      paths.push_back(path);
    }
  }

  return paths;
}

TransferPlan ChunkPlanner::Plan(std::size_t total_bytes, std::size_t chunk_bytes,
                                const ProfileResult& profile, TransferMode mode,
                                std::size_t min_chunks_for_relay,
                                double relay_min_effective_bw_gbps,
                                double relay_min_direct_ratio,
                                TransferDirection direction) const {
  if (total_bytes == 0) {
    return {};
  }
  if (chunk_bytes == 0) {
    throw std::invalid_argument("chunk_bytes must be greater than zero");
  }

  std::vector<Chunk> chunks;
  chunks.reserve((total_bytes + chunk_bytes - 1) / chunk_bytes);
  for (std::size_t offset = 0; offset < total_bytes; offset += chunk_bytes) {
    Chunk chunk;
    chunk.src_offset = offset;
    chunk.dst_offset = offset;
    chunk.bytes = std::min(chunk_bytes, total_bytes - offset);
    chunks.push_back(chunk);
  }

  const std::size_t num_chunks = chunks.size();
  if (mode == TransferMode::Pool && num_chunks < min_chunks_for_relay) {
    mode = TransferMode::DirectOnly;
  }

  auto paths = BuildPaths(profile, mode, relay_min_effective_bw_gbps,
                          relay_min_direct_ratio, direction);
  if (paths.empty()) {
    throw std::runtime_error("no enabled transfer path is available");
  }

  return PlanChunks(chunks, total_bytes, chunk_bytes, std::move(paths));
}

TransferPlan ChunkPlanner::PlanRanges(const std::vector<TransferRange>& ranges,
                                      std::size_t chunk_bytes,
                                      const ProfileResult& profile,
                                      TransferMode mode,
                                      std::size_t min_chunks_for_relay,
                                      double relay_min_effective_bw_gbps,
                                      double relay_min_direct_ratio,
                                      TransferDirection direction) const {
  if (ranges.empty()) {
    return {};
  }
  if (chunk_bytes == 0) {
    throw std::invalid_argument("chunk_bytes must be greater than zero");
  }

  std::vector<Chunk> chunks;
  std::size_t total_bytes = 0;
  for (const auto& range : ranges) {
    if (range.bytes == 0) {
      continue;
    }
    total_bytes += range.bytes;
    std::size_t consumed = 0;
    while (consumed < range.bytes) {
      const std::size_t bytes = std::min(chunk_bytes, range.bytes - consumed);
      Chunk chunk;
      chunk.src_offset = range.src_offset + consumed;
      chunk.dst_offset = range.dst_offset + consumed;
      chunk.bytes = bytes;
      chunks.push_back(chunk);
      consumed += bytes;
    }
  }

  if (chunks.empty()) {
    return {};
  }

  if (mode == TransferMode::Pool && chunks.size() < min_chunks_for_relay) {
    mode = TransferMode::DirectOnly;
  }

  auto paths = BuildPaths(profile, mode, relay_min_effective_bw_gbps,
                          relay_min_direct_ratio, direction);
  if (paths.empty()) {
    throw std::runtime_error("no enabled transfer path is available");
  }

  return PlanChunks(chunks, total_bytes, chunk_bytes, std::move(paths));
}

TransferPlan ChunkPlanner::PlanChunks(const std::vector<Chunk>& chunks,
                                      std::size_t total_bytes,
                                      std::size_t chunk_bytes,
                                      std::vector<Path> paths) const {
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

  std::vector<double> assigned_scores(paths.size(), 0.0);

  for (const auto& chunk : chunks) {
    std::size_t selected = 0;
    double best_score = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < paths.size(); ++i) {
      const double score = assigned_scores[i] / paths[i].effective_bw_gbps;
      if (score < best_score) {
        best_score = score;
        selected = i;
      }
    }

    plan.assignments[selected].chunks.push_back(chunk);
    assigned_scores[selected] += static_cast<double>(chunk.bytes);
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
