#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
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
    case turbobus::PathKind::DirectD2H:
      return "direct";
    case turbobus::PathKind::RelayP2PThenD2H:
      return "relay";
  }
  return "unknown";
}

std::string DirectionToString(turbobus::TransferDirection direction) {
  switch (direction) {
    case turbobus::TransferDirection::H2D:
      return "h2d";
    case turbobus::TransferDirection::D2H:
      return "d2h";
  }
  return "unknown";
}

void CheckCuda(cudaError_t result, const char* message) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " +
                             cudaGetErrorString(result));
  }
}

}  // namespace

PYBIND11_MODULE(_turbobus, m) {
  m.def("register_host_memory",
        [](std::uintptr_t host_ptr, std::size_t bytes) {
          if (host_ptr == 0) {
            throw std::invalid_argument("host_ptr must not be null");
          }
          if (bytes == 0) {
            throw std::invalid_argument("bytes must be positive");
          }
          CheckCuda(cudaHostRegister(reinterpret_cast<void*>(host_ptr), bytes,
                                     cudaHostRegisterPortable),
                    "cudaHostRegister failed");
        },
        py::arg("host_ptr"), py::arg("bytes"),
        py::call_guard<py::gil_scoped_release>());
  m.def("unregister_host_memory",
        [](std::uintptr_t host_ptr) {
          if (host_ptr == 0) {
            throw std::invalid_argument("host_ptr must not be null");
          }
          CheckCuda(cudaHostUnregister(reinterpret_cast<void*>(host_ptr)),
                    "cudaHostUnregister failed");
        },
        py::arg("host_ptr"), py::call_guard<py::gil_scoped_release>());
  m.def("export_device_ipc_handle",
        [](std::uintptr_t device_ptr) {
          if (device_ptr == 0) {
            throw std::invalid_argument("device_ptr must not be null");
          }
          cudaIpcMemHandle_t handle;
          CheckCuda(
              cudaIpcGetMemHandle(&handle, reinterpret_cast<void*>(device_ptr)),
              "cudaIpcGetMemHandle failed");
          return py::bytes(reinterpret_cast<const char*>(&handle), sizeof(handle));
        },
        py::arg("device_ptr"), py::call_guard<py::gil_scoped_release>());
  m.def("open_device_ipc_handle",
        [](py::bytes cuda_ipc_handle) {
          std::string raw = cuda_ipc_handle;
          if (raw.size() != sizeof(cudaIpcMemHandle_t)) {
            throw std::invalid_argument("cuda IPC handle has invalid size");
          }
          cudaIpcMemHandle_t handle;
          std::memcpy(&handle, raw.data(), sizeof(handle));
          void* device_ptr = nullptr;
          CheckCuda(cudaIpcOpenMemHandle(&device_ptr, handle,
                                         cudaIpcMemLazyEnablePeerAccess),
                    "cudaIpcOpenMemHandle failed");
          return reinterpret_cast<std::uintptr_t>(device_ptr);
        },
        py::arg("cuda_ipc_handle"), py::call_guard<py::gil_scoped_release>());
  m.def("close_device_ipc_handle",
        [](std::uintptr_t device_ptr) {
          if (device_ptr == 0) {
            throw std::invalid_argument("device_ptr must not be null");
          }
          CheckCuda(cudaIpcCloseMemHandle(reinterpret_cast<void*>(device_ptr)),
                    "cudaIpcCloseMemHandle failed");
        },
        py::arg("device_ptr"), py::call_guard<py::gil_scoped_release>());

  py::enum_<turbobus::TransferMode>(m, "TransferMode")
      .value("Pool", turbobus::TransferMode::Pool)
      .value("DirectOnly", turbobus::TransferMode::DirectOnly)
      .value("RelayOnly", turbobus::TransferMode::RelayOnly);

  py::enum_<turbobus::PathKind>(m, "PathKind")
      .value("DirectH2D", turbobus::PathKind::DirectH2D)
      .value("RelayH2DThenP2P", turbobus::PathKind::RelayH2DThenP2P)
      .value("DirectD2H", turbobus::PathKind::DirectD2H)
      .value("RelayP2PThenD2H", turbobus::PathKind::RelayP2PThenD2H);

  py::enum_<turbobus::TransferDirection>(m, "TransferDirection")
      .value("H2D", turbobus::TransferDirection::H2D)
      .value("D2H", turbobus::TransferDirection::D2H);

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
                     &turbobus::RuntimeOptions::relay_min_direct_ratio)
      .def_readwrite("enable_dynamic_weights",
                     &turbobus::RuntimeOptions::enable_dynamic_weights)
      .def_readwrite("dynamic_weight_alpha",
                     &turbobus::RuntimeOptions::dynamic_weight_alpha);

  py::class_<turbobus::RelayProfile>(m, "RelayProfile")
      .def(py::init<>())
      .def_readwrite("relay_device", &turbobus::RelayProfile::relay_device)
      .def_readwrite("target_device", &turbobus::RelayProfile::target_device)
      .def_readwrite("h2d_bw_gbps", &turbobus::RelayProfile::h2d_bw_gbps)
      .def_readwrite("d2h_bw_gbps", &turbobus::RelayProfile::d2h_bw_gbps)
      .def_readwrite("p2p_bw_gbps", &turbobus::RelayProfile::p2p_bw_gbps)
      .def_readwrite("effective_bw_gbps", &turbobus::RelayProfile::effective_bw_gbps)
      .def_readwrite("effective_d2h_bw_gbps",
                     &turbobus::RelayProfile::effective_d2h_bw_gbps)
      .def_readwrite("p2p_enabled", &turbobus::RelayProfile::p2p_enabled);

  py::class_<turbobus::ProfileResult>(m, "ProfileResult")
      .def(py::init<>())
      .def_readwrite("target_device", &turbobus::ProfileResult::target_device)
      .def_readwrite("direct_h2d_bw_gbps",
                     &turbobus::ProfileResult::direct_h2d_bw_gbps)
      .def_readwrite("direct_d2h_bw_gbps",
                     &turbobus::ProfileResult::direct_d2h_bw_gbps)
      .def_readwrite("relays", &turbobus::ProfileResult::relays);

  py::class_<turbobus::Path>(m, "Path")
      .def(py::init<>())
      .def_property_readonly("kind",
                             [](const turbobus::Path& path) {
                               return PathKindToString(path.kind);
                             })
      .def_readwrite("kind_value", &turbobus::Path::kind)
      .def_property_readonly("direction",
                             [](const turbobus::Path& path) {
                               return DirectionToString(path.direction);
                             })
      .def_readwrite("direction_value", &turbobus::Path::direction)
      .def_readwrite("target_device", &turbobus::Path::target_device)
      .def_readwrite("relay_device", &turbobus::Path::relay_device)
      .def_readwrite("h2d_bw_gbps", &turbobus::Path::h2d_bw_gbps)
      .def_readwrite("d2h_bw_gbps", &turbobus::Path::d2h_bw_gbps)
      .def_readwrite("p2p_bw_gbps", &turbobus::Path::p2p_bw_gbps)
      .def_readwrite("effective_bw_gbps", &turbobus::Path::effective_bw_gbps)
      .def_readwrite("enabled", &turbobus::Path::enabled);

  py::class_<turbobus::Chunk>(m, "Chunk")
      .def(py::init<>())
      .def_readwrite("src_offset", &turbobus::Chunk::src_offset)
      .def_readwrite("dst_offset", &turbobus::Chunk::dst_offset)
      .def_readwrite("bytes", &turbobus::Chunk::bytes);

  py::class_<turbobus::TransferRange>(m, "TransferRange")
      .def(py::init<>())
      .def_readwrite("src_offset", &turbobus::TransferRange::src_offset)
      .def_readwrite("dst_offset", &turbobus::TransferRange::dst_offset)
      .def_readwrite("bytes", &turbobus::TransferRange::bytes);

  py::class_<turbobus::PathAssignment>(m, "PathAssignment")
      .def(py::init<>())
      .def_readwrite("path", &turbobus::PathAssignment::path)
      .def_readwrite("chunks", &turbobus::PathAssignment::chunks);

  py::class_<turbobus::TransferPlan>(m, "TransferPlan")
      .def(py::init<>())
      .def_readwrite("total_bytes", &turbobus::TransferPlan::total_bytes)
      .def_readwrite("chunk_bytes", &turbobus::TransferPlan::chunk_bytes)
      .def_readwrite("assignments", &turbobus::TransferPlan::assignments);

  py::class_<turbobus::PathStats>(m, "PathStats")
      .def_property_readonly("kind",
                             [](const turbobus::PathStats& stats) {
                               return PathKindToString(stats.kind);
                             })
      .def_property_readonly("direction",
                             [](const turbobus::PathStats& stats) {
                               return DirectionToString(stats.direction);
                             })
      .def_readonly("target_device", &turbobus::PathStats::target_device)
      .def_readonly("relay_device", &turbobus::PathStats::relay_device)
      .def_readonly("bytes", &turbobus::PathStats::bytes)
      .def_readonly("chunks", &turbobus::PathStats::chunks)
      .def_readonly("cuda_elapsed_ms", &turbobus::PathStats::cuda_elapsed_ms)
      .def_readonly("gib_per_second", &turbobus::PathStats::gib_per_second);

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
                    &turbobus::TransferStats::relay_device_chunks)
      .def_readonly("path_stats", &turbobus::TransferStats::path_stats);

  py::class_<turbobus::DummyComputeStats>(m, "DummyComputeStats")
      .def_readonly("elements", &turbobus::DummyComputeStats::elements)
      .def_readonly("iterations", &turbobus::DummyComputeStats::iterations)
      .def_readonly("cuda_elapsed_ms",
                    &turbobus::DummyComputeStats::cuda_elapsed_ms);

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
      .def("set_cached_profile", &turbobus::TurboBusRuntime::SetCachedProfile,
           py::arg("profile"))
      .def("set_transfer_mode", &turbobus::TurboBusRuntime::SetTransferMode,
           py::arg("mode"))
      .def("cached_profile", &turbobus::TurboBusRuntime::CachedProfile,
           py::return_value_policy::reference_internal)
      .def("planner_profile", &turbobus::TurboBusRuntime::PlannerProfile,
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
      .def("offload_to_cpu",
           [](turbobus::TurboBusRuntime& runtime, std::uintptr_t target_ptr,
              std::uintptr_t host_ptr, std::size_t bytes) {
             return runtime.OffloadToCpu(reinterpret_cast<void*>(target_ptr),
                                         reinterpret_cast<void*>(host_ptr), bytes);
           },
           py::arg("target_ptr"), py::arg("host_ptr"), py::arg("bytes"))
      .def("fetch_plan_to_gpu",
           [](turbobus::TurboBusRuntime& runtime, std::uintptr_t host_ptr,
              std::size_t host_bytes, std::uintptr_t target_ptr,
              std::size_t target_bytes, const turbobus::TransferPlan& plan) {
             return runtime.FetchPlanToGpu(
                 reinterpret_cast<void*>(host_ptr), host_bytes,
                 reinterpret_cast<void*>(target_ptr), target_bytes, plan);
           },
           py::arg("host_ptr"), py::arg("host_bytes"), py::arg("target_ptr"),
           py::arg("target_bytes"), py::arg("plan"))
      .def("offload_plan_to_cpu",
           [](turbobus::TurboBusRuntime& runtime, std::uintptr_t target_ptr,
              std::size_t target_bytes, std::uintptr_t host_ptr,
              std::size_t host_bytes, const turbobus::TransferPlan& plan) {
             return runtime.OffloadPlanToCpu(
                 reinterpret_cast<void*>(target_ptr), target_bytes,
                 reinterpret_cast<void*>(host_ptr), host_bytes, plan);
           },
           py::arg("target_ptr"), py::arg("target_bytes"), py::arg("host_ptr"),
           py::arg("host_bytes"), py::arg("plan"))
      .def("fetch_ranges_to_gpu",
           [](turbobus::TurboBusRuntime& runtime, std::uintptr_t host_ptr,
              std::size_t host_bytes, std::uintptr_t target_ptr,
              std::size_t target_bytes,
              const std::vector<turbobus::TransferRange>& ranges) {
             return runtime.FetchRangesToGpu(
                 reinterpret_cast<void*>(host_ptr), host_bytes,
                 reinterpret_cast<void*>(target_ptr), target_bytes, ranges);
           },
           py::arg("host_ptr"), py::arg("host_bytes"), py::arg("target_ptr"),
           py::arg("target_bytes"), py::arg("ranges"))
      .def("offload_ranges_to_cpu",
           [](turbobus::TurboBusRuntime& runtime, std::uintptr_t target_ptr,
              std::size_t target_bytes, std::uintptr_t host_ptr,
              std::size_t host_bytes,
              const std::vector<turbobus::TransferRange>& ranges) {
             return runtime.OffloadRangesToCpu(
                 reinterpret_cast<void*>(target_ptr), target_bytes,
                 reinterpret_cast<void*>(host_ptr), host_bytes, ranges);
           },
           py::arg("target_ptr"), py::arg("target_bytes"), py::arg("host_ptr"),
           py::arg("host_bytes"), py::arg("ranges"))
      .def("run_dummy_compute",
           [](turbobus::TurboBusRuntime& runtime, std::uintptr_t device_ptr,
              std::size_t elements, int iterations) {
             return runtime.RunDummyCompute(reinterpret_cast<void*>(device_ptr),
                                            elements, iterations);
           },
           py::arg("device_ptr"), py::arg("elements"), py::arg("iterations"),
           py::call_guard<py::gil_scoped_release>())
      .def("wait", &turbobus::TurboBusRuntime::Wait, py::arg("handle"),
           py::call_guard<py::gil_scoped_release>())
      .def("stats", &turbobus::TurboBusRuntime::GetStats, py::arg("handle"));
}
