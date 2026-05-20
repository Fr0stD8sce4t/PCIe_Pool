#pragma once

#include <cstddef>

#include "turbobus/types.h"

namespace turbobus {

DummyComputeStats RunDummyCompute(int device, void* device_ptr, std::size_t elements,
                                  int iterations);

}  // namespace turbobus
