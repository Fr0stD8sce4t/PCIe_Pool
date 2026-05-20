from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_relay_gpus(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def default_log_path() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("benchmarks") / "results" / f"vllm_turbobus_connector_{stamp}.log")


def configure_cuda_devices(args) -> None:
    physical_relays = parse_relay_gpus(args.relay_gpus)
    visible = [args.target_gpu, *physical_relays]
    if args.map_physical_gpus and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in visible)
        args.runtime_target_gpu = 0
        args.runtime_relay_gpus = list(range(1, len(visible)))
        return
    args.runtime_target_gpu = args.target_gpu
    args.runtime_relay_gpus = physical_relays


def prompt_text(prompt: str, repeat: int) -> str:
    if repeat <= 1:
        return prompt
    return " ".join([prompt] * repeat)


def make_runtime(args, mode: str):
    import turbobus

    options = turbobus.RuntimeOptions(
        chunk_bytes=args.chunk_bytes,
        profile_bytes=args.profile_bytes,
        transfer_mode=mode,
        enable_dynamic_weights=args.dynamic_weights,
    )
    return turbobus.Runtime(
        target_gpu=args.runtime_target_gpu,
        relay_gpus=args.runtime_relay_gpus,
        options=options,
    )


def first_request_id(integration) -> str:
    if not integration.state.allocations:
        raise RuntimeError("vLLM did not allocate any KV blocks")
    return next(iter(integration.state.allocations))


def run(args) -> None:
    configure_cuda_devices(args)
    if args.disable_multiproc_executor:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    from vllm import LLM, SamplingParams

    import turbobus
    from turbobus.vllm_connector import VllmTurboBusConnector
    from turbobus.vllm_integration import VllmTurboBusIntegration

    torch.cuda.set_device(args.runtime_target_gpu)
    runtime = make_runtime(args, args.mode)
    integration = VllmTurboBusIntegration(runtime)
    integration.install()
    connector = VllmTurboBusConnector(integration)
    connector.install()

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )

    first_prompt = prompt_text(args.prompt, args.prompt_repeat)
    second_prompt = prompt_text(args.prompt + args.second_prompt_suffix, args.prompt_repeat)
    start = time.perf_counter()
    first_outputs = llm.generate([first_prompt], sampling)
    first_ms = (time.perf_counter() - start) * 1000.0
    source_request_id = first_request_id(integration)
    source_blocks = len(integration.block_ids_for_request(source_request_id))
    if source_blocks < args.restore_blocks:
        raise RuntimeError(
            f"source request has {source_blocks} blocks, need {args.restore_blocks}; "
            "increase --prompt-repeat or lower --restore-blocks"
        )

    connector.allocate_cpu_backings_for_blocks(args.restore_blocks)
    save_event = connector.save_request(source_request_id, args.restore_blocks)
    integration.state.allocations.clear()
    connector.restore_next_allocation(args.restore_blocks)

    start = time.perf_counter()
    second_outputs = llm.generate([second_prompt], sampling)
    second_ms = (time.perf_counter() - start) * 1000.0
    allocation_events = [event for event in connector.events if event.operation == "allocation"]
    restore_events = [event for event in connector.events if event.operation == "restore"]
    if not restore_events:
        raise RuntimeError(
            "TurboBus restore did not run inside vLLM allocate_slots(); "
            f"allocation_events={len(allocation_events)}"
        )
    restore_event = restore_events[-1]

    first_text = first_outputs[0].outputs[0].text if first_outputs[0].outputs else ""
    second_text = second_outputs[0].outputs[0].text if second_outputs[0].outputs else ""
    text_match = first_text == second_text

    print("COPY_SUMMARY_BEGIN")
    print(
        "vllm_connector_config",
        f"target={args.target_gpu}",
        f"relays={parse_relay_gpus(args.relay_gpus)}",
        f"runtime_target={args.runtime_target_gpu}",
        f"runtime_relays={args.runtime_relay_gpus}",
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        f"model={args.model}",
        f"prompt_repeat={args.prompt_repeat}",
        f"restore_blocks={args.restore_blocks}",
        f"second_prompt_suffix={args.second_prompt_suffix!r}",
        f"chunk_bytes={args.chunk_bytes}",
        f"mode={args.mode}",
        f"dynamic_weights={args.dynamic_weights}",
    )
    print(
        "vllm_connector_scenario",
        "type=real_vllm_allocate_slots_restore",
        "first_request=save_kv_after_generate",
        "second_request=restore_inside_allocate_slots",
        "boundary=KVCacheManager.allocate_slots",
        "note=real_inference_path_not_post_generate_restore_benchmark",
    )
    print(
        "vllm_connector_result",
        f"source_request={source_request_id}",
        f"source_blocks={source_blocks}",
        f"first_generate_ms={first_ms:.3f}",
        f"second_generate_ms={second_ms:.3f}",
        f"text_match={text_match}",
        f"allocation_events={len(allocation_events)}",
    )
    for index, event in enumerate(allocation_events):
        print(
            "vllm_connector_allocation",
            f"index={index}",
            f"request={event.request_id}",
            f"armed_blocks={event.armed_blocks}",
            f"available_blocks={event.available_blocks}",
        )
    print(
        "vllm_connector_save",
        f"blocks={save_event.block_count}",
        f"bytes={save_event.bytes}",
        f"elapsed_ms={save_event.elapsed_ms:.3f}",
        f"direct_chunks={save_event.direct_chunks}",
        f"relay_chunks={save_event.relay_chunks}",
    )
    print(
        "vllm_connector_restore",
        f"request={restore_event.request_id}",
        f"blocks={restore_event.block_count}",
        f"bytes={restore_event.bytes}",
        f"elapsed_ms={restore_event.elapsed_ms:.3f}",
        f"direct_chunks={restore_event.direct_chunks}",
        f"relay_chunks={restore_event.relay_chunks}",
    )
    print("COPY_SUMMARY_END")


def parse_args():
    parser = argparse.ArgumentParser(description="Run TurboBus from the real vLLM allocate_slots path")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--second-prompt-suffix", default=" Italy")
    parser.add_argument("--prompt-repeat", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--target-gpu", type=int, required=True)
    parser.add_argument("--relay-gpus", default="")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-multiproc-executor",
        dest="disable_multiproc_executor",
        action="store_false",
        help="Allow vLLM to use a separate engine process. The TurboBus hook needs in-process vLLM.",
    )
    parser.set_defaults(disable_multiproc_executor=True)
    parser.add_argument("--restore-blocks", type=int, default=8)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--mode", choices=["direct", "relay", "pool"], default="pool")
    parser.add_argument("--dynamic-weights", action="store_true")
    parser.add_argument("--log-output", default=None)
    parser.add_argument(
        "--no-map-physical-gpus",
        dest="map_physical_gpus",
        action="store_false",
    )
    parser.set_defaults(map_physical_gpus=True)
    args = parser.parse_args()
    if args.log_output is None:
        args.log_output = default_log_path()
    return args


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_output)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ok = False
    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            try:
                run(args)
            except Exception:
                print("VLLM_CONNECTOR_ERROR_BEGIN")
                traceback.print_exc()
                print("VLLM_CONNECTOR_ERROR_END")
            else:
                ok = True
    print("vllm_turbobus_connector log", log_path)
    if ok:
        print("vllm_turbobus_connector status ok")
    else:
        print("vllm_turbobus_connector status failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
