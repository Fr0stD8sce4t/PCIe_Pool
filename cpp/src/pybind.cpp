#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>

#include "turbobus/runtime.h"

namespace py = pybind11;

namespace {

std::string StatusToString(turbobus::TransferStatus status) {
  switch (status) {
    case turbobus::TransferStatus::Pending:
      return "pending";
    case turbobus::TransferStatus::Submitted:
      return "submitted";
    case turbobus::TransferStatus::Complete:
      return "complete";
    case turbobus::TransferStatus::Failed:
      return "failed";
  }
  return "unknown";
}

std::string PathKindToString(turbobus::PathKind kind) {
  switch (kind) {
    case turbobus::PathKind::DirectH2D:
      return "direct";
    case turbobus::PathKind::RelayH2DThenP2P:
      return "relay";
  }
  return "unknown";
}

}  // namespace

PYBIND11_MODULE(_turbobus, m) {
  py::enum_<turbobus::TransferMode>(m, "TransferMode")
      .value("Pool", turbobus::TransferMode::Pool)
      .value("DirectOnly", turbobus::TransferMode::DirectOnly)
      .value("RelayOnly", turbobus::TransferMode::RelayOnly);

  py::class_<turbobus::RuntimeOptions>(m, "RuntimeOptions")
      .def(py::init<>())
      .def_readwrite("chunk_bytes", &turbobus::RuntimeOptions::chunk_bytes)
      .def_readwrite("staging_slots", &turbobus::RuntimeOptions::staging_slots)
      .def_readwrite("enable_peer_access", &turbobus::RuntimeOptions::enable_peer_access)
      .def_readwrite("profile_bytes", &turbobus::RuntimeOptions::profile_bytes)
      .def_readwrite("profile_on_first_transfer",
                     &turbobus::RuntimeOptions::profile_on_first_transfer)
      .def_readwrite("profile_cache_enabled",
                     &turbobus::RuntimeOptions::profile_cache_enabled)
      .def_readwrite("transfer_mode", &turbobus::RuntimeOptions::transfer_mode)
      .def_readwrite("min_chunks_for_relay",
                     &turbobus::RuntimeOptions::min_chunks_for_relay)
      .def_readwrite("relay_min_effective_bw_gbps",
                     &turbobus::RuntimeOptions::relay_min_effective_bw_gbps)
      .def_readwrite("relay_min_direct_ratio",
                     &turbobus::RuntimeOptions::relay_min_direct_ratio);

  py::class_<turbobus::RelayProfile>(m, "RelayProfile")
      .def_readonly("relay_device", &turbobus::RelayProfile::relay_device)
      .def_readonly("target_device", &turbobus::RelayProfile::target_device)
      .def_readonly("h2d_bw_gbps", &turbobus::RelayProfile::h2d_bw_gbps)
      .def_readonly("p2p_bw_gbps", &turbobus::RelayProfile::p2p_bw_gbps)
      .def_readonly("effective_bw_gbps", &turbobus::RelayProfile::effective_bw_gbps)
      .def_readonly("p2p_enabled", &turbobus::RelayProfile::p2p_enabled);

  py::class_<turbobus::ProfileResult>(m, "ProfileResult")
      .def_readonly("target_device", &turbobus::ProfileResult::target_device)
      .def_readonly("direct_h2d_bw_gbps", &turbobus::ProfileResult::direct_h2d_bw_gbps)
      .def_readonly("relays", &turbobus::ProfileResult::relays);

  py::class_<turbobus::Path>(m, "Path")
      .def_property_readonly("kind",
                             [](const turbobus::Path& path) {
                               return PathKindToString(path.kind);
                             })
      .def_readonly("target_device", &turbobus::Path::target_device)
      .def_readonly("relay_device", &turbobus::Path::relay_device)
      .def_readonly("h2d_bw_gbps", &turbobus::Path::h2d_bw_gbps)
      .def_readonly("p2p_bw_gbps", &turbobus::Path::p2p_bw_gbps)
      .def_readonly("effective_bw_gbps", &turbobus::Path::effective_bw_gbps)
      .def_readonly("enabled", &turbobus::Path::enabled);

  py::class_<turbobus::Chunk>(m, "Chunk")
      .def_readonly("src_offset", &turbobus::Chunk::src_offset)
      .def_readonly("dst_offset", &turbobus::Chunk::dst_offset)
      .def_readonly("bytes", &turbobus::Chunk::bytes);

  py::class_<turbobus::PathAssignment>(m, "PathAssignment")
      .def_readonly("path", &turbobus::PathAssignment::path)
      .def_readonly("chunks", &turbobus::PathAssignment::chunks);

  py::class_<turbobus::TransferPlan>(m, "TransferPlan")
      .def_readonly("total_bytes", &turbobus::TransferPlan::total_bytes)
      .def_readonly("chunk_bytes", &turbobus::TransferPlan::chunk_bytes)
      .def_readonly("assignments", &turbobus::TransferPlan::assignments);

  py::class_<turbobus::TransferStats>(m, "TransferStats")
      .def_readonly("bytes", &turbobus::TransferStats::bytes)
      .def_readonly("direct_bytes", &turbobus::TransferStats::direct_bytes)
      .def_readonly("relay_bytes", &turbobus::TransferStats::relay_bytes)
      .def_readonly("submit_to_complete_ms",
                    &turbobus::TransferStats::submit_to_complete_ms)
      .def_readonly("cuda_elapsed_ms", &turbobus::TransferStats::cuda_elapsed_ms)
      .def_readonly("gib_per_second", &turbobus::TransferStats::gib_per_second)
      .def_readonly("submit_gib_per_second",
                    &turbobus::TransferStats::submit_gib_per_second)
      .def_readonly("direct_chunks", &turbobus::TransferStats::direct_chunks)
      .def_readonly("relay_chunks", &turbobus::TransferStats::relay_chunks)
      .def_readonly("relay_devices", &turbobus::TransferStats::relay_devices)
      .def_readonly("relay_device_bytes",
                    &turbobus::TransferStats::relay_device_bytes)
      .def_readonly("relay_device_chunks",
                    &turbobus::TransferStats::relay_device_chunks);

  py::class_<turbobus::TransferHandle>(m, "TransferHandle")
      .def_property_readonly("id", [](const turbobus::TransferHandle& h) { return h.id; })
      .def_property_readonly("status",
                             [](const turbobus::TransferHandle& h) {
                               return StatusToString(h.status);
                             })
      .def_readonly("error", &turbobus::TransferHandle::error);

  py::class_<turbobus::TurboBusRuntime>(m, "Runtime")
      .def(py::init<turbobus::RuntimeOptions>(), py::arg("options") = turbobus::RuntimeOptions{})
      .def("init", &turbobus::TurboBusRuntime::Init, py::arg("target_device"),
           py::arg("relay_devices"))
      .def("profile", &turbobus::TurboBusRuntime::Profile,
           py::arg("bytes") = 256ull * 1024ull * 1024ull,
           py::arg("force") = false)
      .def("set_transfer_mode", &turbobus::TurboBusRuntime::SetTransferMode,
           py::arg("mode"))
      .def("cached_profile", &turbobus::TurboBusRuntime::CachedProfile,
           py::return_value_policy::reference_internal)
      .def("last_plan", &turbobus::TurboBusRuntime::LastPlan,
           py::return_value_policy::reference_internal)
      .def("fetch_to_gpu",
           [](turbobus::TurboBusRuntime& runtime, std::uintptr_t host_ptr,
              std::uintptr_t target_ptr, std::size_t bytes) {
             return runtime.FetchToGpu(reinterpret_cast<void*>(host_ptr),
                                       reinterpret_cast<void*>(target_ptr), bytes);
           },
           py::arg("host_ptr"), py::arg("target_ptr"), py::arg("bytes"))
      .def("wait", &turbobus::TurboBusRuntime::Wait, py::arg("handle"))
      .def("stats", &turbobus::TurboBusRuntime::GetStats, py::arg("handle"));
}
