# TurboBus Progress

Update this file after every coding turn that changes the project.

## Status As Of 2026-05-22

The active goal is to turn TurboBus from a working prototype into a paper
reproduction system for PCIe bandwidth pooling via relay GPUs.

## Recent Mainline Commits

- Split Runtime plan trace helper from Runtime
  - Added `turbobus/plan_trace.py`.
  - Moved `transfer_plan_to_dict()` out of `turbobus/runtime.py`.
  - Preserved `from turbobus.runtime import transfer_plan_to_dict`.
  - Did not create `profile_cache.py`; the current Runtime profile helpers do
    not justify a new layer yet.

- `cbfdcc2 Document targeted verification policy`
  - Added the verification policy: future code updates should use targeted
    checks based on the files changed instead of running full test, CUDA, and
    vLLM suites by default.

- `Split transfer selector from Runtime`
  - Added `turbobus/transfer_selector.py`.
  - Moved `TransferMode`, `AutoTransferDecision`, and
    `AutoTransferSelector` out of `turbobus/runtime.py`.
  - Preserved public imports from `turbobus.runtime` and `turbobus`.

- `830d137 Document TurboBus roadmap workflow`
  - Added repository roadmap files and AGENTS rules for keeping future coding
    turns on the TurboBus paper reproduction main line.

- `c9cb837 Persist vLLM save intent until block allocation`
  - The vLLM connector now keeps save parameters by request id until enough
    allocated block ids arrive.
  - This makes save metadata less dependent on one ideal scheduler-output
    shape.

- `c9ebc1f Add direction-aware transfer profiling`
  - Native profiling now measures D2H bandwidth.
  - Planner, Runtime, pybind, Python auto selection, benchmarks, and tests now
    carry direction-specific D2H profile fields.

- `86119ce Write structured vLLM sweep case outputs`
  - vLLM connector sweep output is structured for later comparison.

- `2b1597b Enforce daemon session chunk quota`
  - Daemon session quota checks reject excess transfer chunks.

- `ba8c1d0 Share vLLM example CUDA mapping`
  - vLLM examples share CUDA device mapping helpers.

- `7507ca6 Consolidate vLLM connector configuration`
  - Connector configuration was consolidated around shared keys.

## Last Verified Checks

On the local Windows development environment:

```text
python -m unittest discover -s test\python -p "test_*.py" -v
```

Result: 96 tests passed, 3 skipped.

```text
python -m compileall turbobus benchmarks examples test\python -q
git diff --check
```

Result: passed.

Additional import check:

```text
from turbobus.runtime import AutoTransferSelector, TransferMode
from turbobus.transfer_selector import AutoTransferSelector, TransferMode
```

Result: passed.

For the plan trace split:

```text
python -m unittest discover -s test\python -p "test_runtime_handle.py" -v
```

Result: 21 tests passed, 2 skipped because PyTorch is not installed locally.

```text
python -m compileall turbobus test\python\test_runtime_handle.py -q
git diff --check
```

Result: passed.

Local C++/CUDA checks were not run because `cmake` is not installed in this
environment.

## Known Server Follow-Up

After C++/CUDA/pybind edits, reinstall before server tests:

```bash
pip uninstall -y turbobus
rm -rf build build-test build-temp *.egg-info turbobus/_turbobus*.so
pip install -e .
```

Then run native and vLLM checks on target GPU 6 with relay GPU 5.

## Next Task

Start with the task under `## Current` in `docs/NEXT_STEPS.md`: add
multi-relay planner coverage for two relay GPUs and direction-specific
bandwidth fields.
