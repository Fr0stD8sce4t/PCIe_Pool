# TurboBus Test Plan

These tests are staged so the cheap checks run first and GPU-dependent checks
come later. They are added as source files only; they have not been executed in
this environment.

## 1. Python Runtime Handle Checks

File: `python/test_runtime_handle.py`

Purpose:

- verify `TransferHandle.wait()` updates status to `complete`
- verify failed wait updates status to `failed`
- verify repeated wait on a completed handle is harmless

Hardware:

- no GPU required

## 2. Daemon Session And Quota Checks

File: `python/test_daemon_state.py`

Purpose:

- register a session
- reject over-quota relay use
- close a session and release relay quota
- inspect daemon state through `PROFILE`

Hardware:

- no GPU required

## 3. Daemon Unix Socket Checks

File: `python/test_daemon_socket.py`

Purpose:

- start the daemon on a Unix socket
- send `REGISTER_SESSION`
- send `PROFILE`
- send `CLOSE_SESSION`

Hardware:

- no GPU required
- Unix domain socket support required

## 4. Planner Checks

File: `cpp/test_planner.cpp`

Purpose:

- verify chunks are generated for the full byte range
- verify direct and relay paths both receive chunks when both have bandwidth
- verify faster paths receive at least as many bytes as slower paths

Hardware:

- no GPU required
- C++ compiler required

## 5. Direct H2D CUDA Check

File: `cpp/test_direct_h2d.cu`

Purpose:

- allocate pinned host memory
- allocate target GPU memory
- call `TurboBusRuntime` with no relay GPU
- verify copied bytes on CPU after D2H readback

Hardware:

- one NVIDIA GPU
- CUDA toolkit

## 6. Relay CUDA Check

File: `cpp/test_relay_h2d_p2p.cu`

Purpose:

- require two P2P-capable GPUs
- force a relay-only profile by giving direct bandwidth zero in a manual plan
- verify `CPU -> relay GPU -> target GPU` data correctness

Hardware:

- at least two NVIDIA GPUs
- `cudaDeviceCanAccessPeer(relay, target) == true`
- CUDA toolkit

## 7. Profiler Check

File: `cpp/test_profiler.cu`

Purpose:

- run profiler on target GPU and optional relay GPU
- print direct H2D, relay H2D, P2P, effective bandwidth
- assert measured enabled paths have positive bandwidth

Hardware:

- one GPU for direct
- two P2P-capable GPUs for relay profile

## Suggested Order

```text
python runtime handle
daemon state
daemon socket
planner
direct H2D
relay H2D + P2P
profiler
```

## Example Commands

From the repository root:

```bash
python -m unittest discover -s test/python
```

For C++/CUDA tests:

```bash
cmake -S test/cpp -B build-test
cmake --build build-test --config Release
```

Then run the generated binaries in this order:

```bash
./build-test/test_planner
./build-test/test_direct_h2d
./build-test/test_relay_h2d_p2p
./build-test/test_profiler
```

On Windows, the executable paths and build configuration directory may differ,
for example `build-test\Release\test_planner.exe`.

## CUDA Test Environment Variables

The CUDA tests can be steered away from busy GPUs:

```bash
TURBOBUS_TARGET_GPU=6 TURBOBUS_TEST_BYTES=16777216 ./build-test/test_direct_h2d
TURBOBUS_TARGET_GPU=6 TURBOBUS_RELAY_GPU=5 TURBOBUS_TEST_BYTES=33554432 ./build-test/test_relay_h2d_p2p
TURBOBUS_TARGET_GPU=6 TURBOBUS_RELAY_GPU=5 TURBOBUS_PROFILE_BYTES=16777216 ./build-test/test_profiler
```

Useful variables:

- `TURBOBUS_TARGET_GPU`: target GPU id
- `TURBOBUS_RELAY_GPU`: relay GPU id
- `TURBOBUS_TEST_BYTES`: correctness-test transfer size in bytes
- `TURBOBUS_PROFILE_BYTES`: profiler transfer size in bytes
- `TURBOBUS_CHUNK_BYTES`: chunk size in bytes

If `test_relay_h2d_p2p` is run without `TURBOBUS_TARGET_GPU` and
`TURBOBUS_RELAY_GPU`, it scans for the first CUDA P2P-capable pair.
