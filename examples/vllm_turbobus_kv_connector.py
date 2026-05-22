from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbobus.example_config import configure_cuda_runtime_mapping, parse_gpu_list


def default_log_path() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("benchmarks") / "results" / f"vllm_turbobus_kv_connector_{stamp}.log")


def configure_cuda_devices(args) -> None:
    mapping = configure_cuda_runtime_mapping(
        args.target_gpu,
        args.relay_gpus,
        map_physical_gpus=args.map_physical_gpus,
    )
    args.runtime_target_gpu = mapping.runtime_target_gpu
    args.runtime_relay_gpus = list(mapping.runtime_relay_gpus)


def prompt_text(prompt: str, repeat: int) -> str:
    if repeat <= 1:
        return prompt
    return " ".join([prompt] * repeat)


def second_prompt_text(args, first_prompt: str) -> str:
    return first_prompt + args.second_prompt_suffix


def last_event(events: list[dict], event: str, **matches):
    for item in reversed(events):
        if item.get("event") != event:
            continue
        if all(item.get(key) == value for key, value in matches.items()):
            return item
    return None


def event_value(event: dict | None, key: str, default=0):
    if event is None:
        return default
    return event.get(key, default)


def format_ms(event: dict | None, key: str) -> str:
    value = event_value(event, key, 0.0)
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def run(args) -> None:
    configure_cuda_devices(args)
    if args.disable_multiproc_executor:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    from turbobus.vllm_kv_connector import (
        clear_connector_events,
        clear_saved_prefixes,
        get_connector_events,
    )

    torch.cuda.set_device(args.runtime_target_gpu)
    clear_connector_events()
    clear_saved_prefixes(args.session_id)

    relay_gpus = ",".join(str(gpu) for gpu in args.runtime_relay_gpus)
    ktc = KVTransferConfig(
        kv_connector="TurboBusConnector",
        kv_role="kv_both",
        kv_connector_module_path="turbobus.vllm_kv_connector",
        kv_connector_extra_config={
            "turbobus.target_gpu": args.runtime_target_gpu,
            "turbobus.relay_gpus": relay_gpus,
            "turbobus.chunk_bytes": args.chunk_bytes,
            "turbobus.profile_bytes": args.profile_bytes,
            "turbobus.mode": args.mode,
            "turbobus.min_pool_bytes": args.min_pool_bytes,
            "turbobus.restore_block_limit": args.restore_blocks,
            "turbobus.restore_enabled": args.restore_enabled,
            "turbobus.max_saved_prefixes": args.max_saved_prefixes,
            "turbobus.session_id": args.session_id,
        },
    )
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": False,
        "kv_transfer_config": ktc,
    }
    llm = LLM(**llm_kwargs)

    first_prompt = prompt_text(args.prompt, args.prompt_repeat)
    second_prompt = second_prompt_text(args, first_prompt)
    base_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        extra_args={
            "kv_transfer_params": {
                "turbobus.do_save": args.save_enabled,
                "turbobus.prefix_key": args.prefix_key,
                "turbobus.save_blocks": args.restore_blocks,
                "turbobus.matched_tokens": args.matched_tokens,
            }
        },
    )
    first_outputs = llm.generate([first_prompt], base_sampling)

    events = get_connector_events()
    first_save_event = last_event(events, "save", prefix_key=args.prefix_key)
    restore_prefix_key = args.prefix_key
    restore_base_prompt = first_prompt
    save_event = first_save_event
    second_save_event = None
    if args.second_save_prefix_key:
        second_save_prompt = prompt_text(args.second_save_prompt, args.prompt_repeat)
        second_save_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            extra_args={
                "kv_transfer_params": {
                    "turbobus.do_save": args.save_enabled,
                    "turbobus.prefix_key": args.second_save_prefix_key,
                    "turbobus.save_blocks": args.restore_blocks,
                    "turbobus.matched_tokens": args.matched_tokens,
                }
            },
        )
        llm.generate([second_save_prompt], second_save_sampling)
        events = get_connector_events()
        second_save_event = last_event(events, "save", prefix_key=args.second_save_prefix_key)
        restore_prefix_key = args.second_save_prefix_key
        restore_base_prompt = second_save_prompt
        save_event = second_save_event

    source_request_id = str(event_value(save_event, "request_id", "unknown"))
    source_blocks = int(event_value(save_event, "block_count", 0) or 0)
    if args.restore_enabled:
        if save_event is None:
            raise RuntimeError(
                f"connector did not save prefix {restore_prefix_key!r}; "
                "check turbobus.do_save and vLLM wait_for_save"
            )
        if source_blocks < args.restore_blocks:
            raise RuntimeError(
                f"source request has {source_blocks} blocks, need {args.restore_blocks}; "
                "increase --prompt-repeat or lower --restore-blocks"
            )

    matched_tokens = max(0, args.matched_tokens) if args.restore_enabled else 0
    second_prompt = second_prompt_text(args, restore_base_prompt)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        extra_args={
            "kv_transfer_params": {
                "turbobus.do_restore": True,
                "turbobus.prefix_key": restore_prefix_key,
                "turbobus.matched_tokens": matched_tokens,
            }
        },
    )
    outputs = llm.generate([second_prompt], sampling)
    events = get_connector_events()
    restore_event = last_event(events, "restore", prefix_key=restore_prefix_key)
    generated_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""

    print("COPY_SUMMARY_BEGIN")
    print(
        "vllm_kv_connector_config",
        f"target={args.target_gpu}",
        f"relays={parse_gpu_list(args.relay_gpus)}",
        f"runtime_target={args.runtime_target_gpu}",
        f"runtime_relays={args.runtime_relay_gpus}",
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        f"model={args.model}",
        f"prompt_repeat={args.prompt_repeat}",
        f"prefix_key={args.prefix_key}",
        f"second_save_prefix_key={args.second_save_prefix_key}",
        f"session_id={args.session_id}",
        f"restore_prefix_key={restore_prefix_key}",
        f"second_prompt_suffix={args.second_prompt_suffix!r}",
        f"requested_matched_tokens={args.matched_tokens}",
        f"matched_tokens={matched_tokens}",
        f"restore_blocks={args.restore_blocks}",
        f"save_enabled={args.save_enabled}",
        f"restore_enabled={args.restore_enabled}",
        f"max_saved_prefixes={args.max_saved_prefixes}",
        f"chunk_bytes={args.chunk_bytes}",
        f"min_pool_bytes={args.min_pool_bytes}",
        f"mode={args.mode}",
    )
    print(
        "vllm_kv_connector_scenario",
        "type=real_vllm_kv_transfer_connector",
        "boundary=KVConnectorBase_V1",
        "entry=start_load_kv",
        "note=official_vllm_external_kv_path",
    )
    if save_event is not None:
        print(
            "vllm_kv_connector_save",
            f"source_request={source_request_id}",
            f"source_blocks={source_blocks}",
            f"blocks={event_value(save_event, 'block_count')}",
            f"bytes={event_value(save_event, 'bytes')}",
            f"elapsed_ms={format_ms(save_event, 'elapsed_ms')}",
            f"runtime_init_ms={format_ms(save_event, 'runtime_init_ms')}",
            f"prepare_ms={format_ms(save_event, 'prepare_ms')}",
            f"cpu_alloc_ms={format_ms(save_event, 'cpu_alloc_ms')}",
            f"group_ms={format_ms(save_event, 'group_ms')}",
            f"adapter_ms={format_ms(save_event, 'adapter_ms')}",
            f"refs_ms={format_ms(save_event, 'refs_ms')}",
            f"transfer_ms={format_ms(save_event, 'transfer_ms')}",
            f"register_ms={format_ms(save_event, 'register_ms')}",
            f"total_ms={format_ms(save_event, 'total_ms')}",
            f"direct_chunks={event_value(save_event, 'direct_chunks')}",
            f"relay_chunks={event_value(save_event, 'relay_chunks')}",
            f"save_layer_count={event_value(save_event, 'layers')}",
            f"save_layer_ranges={event_value(save_event, 'ranges')}",
        )
    if first_save_event is not None and first_save_event is not save_event:
        print(
            "vllm_kv_connector_first_save",
            f"source_request={event_value(first_save_event, 'request_id', 'unknown')}",
            f"blocks={event_value(first_save_event, 'block_count')}",
            f"bytes={event_value(first_save_event, 'bytes')}",
            f"elapsed_ms={format_ms(first_save_event, 'elapsed_ms')}",
        )
    if second_save_event is not None and second_save_event is not save_event:
        print(
            "vllm_kv_connector_second_save",
            f"source_request={event_value(second_save_event, 'request_id', 'unknown')}",
            f"blocks={event_value(second_save_event, 'block_count')}",
            f"bytes={event_value(second_save_event, 'bytes')}",
            f"elapsed_ms={format_ms(second_save_event, 'elapsed_ms')}",
        )
    if restore_event is not None:
        print(
            "vllm_kv_connector_restore",
            f"request_id={event_value(restore_event, 'request_id', 'unknown')}",
            f"prefix_key={event_value(restore_event, 'prefix_key', restore_prefix_key)}",
            f"bytes={event_value(restore_event, 'bytes')}",
            f"elapsed_ms={format_ms(restore_event, 'elapsed_ms')}",
            f"prepare_ms={format_ms(restore_event, 'prepare_ms')}",
            f"transfer_ms={format_ms(restore_event, 'transfer_ms')}",
            f"total_ms={format_ms(restore_event, 'total_ms')}",
            f"direct_chunks={event_value(restore_event, 'direct_chunks')}",
            f"relay_chunks={event_value(restore_event, 'relay_chunks')}",
            f"layers={event_value(restore_event, 'layers')}",
            f"ranges={event_value(restore_event, 'ranges')}",
            f"auto_resolved_mode={event_value(restore_event, 'auto_resolved_mode', 'NA')}",
            f"auto_reason={event_value(restore_event, 'auto_reason', 'NA')}",
        )
    print(
        "vllm_kv_connector_result",
        f"source_request={source_request_id}",
        f"source_blocks={source_blocks}",
        f"shared_prefix={second_prompt.startswith(restore_base_prompt)}",
        f"prompt_tokens={len(getattr(outputs[0], 'prompt_token_ids', []) or []) if outputs else 0}",
        f"generated_text={generated_text!r}",
    )
    print("COPY_SUMMARY_END")


def parse_args():
    parser = argparse.ArgumentParser(description="Run TurboBus through the vLLM KV connector entry")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prompt-repeat", type=int, default=64)
    parser.add_argument("--second-prompt-suffix", default=" Italy")
    parser.add_argument("--prefix-key", default="qwen3-prefix")
    parser.add_argument("--second-save-prefix-key", default=None)
    parser.add_argument("--second-save-prompt", default="The capital of Germany is")
    parser.add_argument("--session-id", default="vllm-kv-connector-example")
    parser.add_argument("--matched-tokens", type=int, default=128)
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
    )
    parser.set_defaults(disable_multiproc_executor=True)
    parser.add_argument("--restore-blocks", type=int, default=8)
    parser.add_argument("--max-saved-prefixes", type=int, default=0)
    parser.add_argument(
        "--save-enabled",
        dest="save_enabled",
        action="store_true",
        help="Ask the first request to save its prefix through the vLLM connector lifecycle.",
    )
    parser.add_argument(
        "--no-save",
        dest="save_enabled",
        action="store_false",
        help="Do not ask the first request to save a prefix.",
    )
    parser.set_defaults(save_enabled=True)
    parser.add_argument(
        "--restore-enabled",
        action="store_true",
        help="Actually write saved TurboBus CPU backing into later vLLM KV slots.",
    )
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--min-pool-bytes", type=int, default=12 * 1024 * 1024)
    parser.add_argument("--mode", choices=["auto", "direct", "relay", "pool"], default="pool")
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
                print("VLLM_KV_CONNECTOR_ERROR_BEGIN")
                traceback.print_exc()
                print("VLLM_KV_CONNECTOR_ERROR_END")
            else:
                ok = True
    print("vllm_turbobus_kv_connector log", log_path)
    if ok:
        print("vllm_turbobus_kv_connector status ok")
    else:
        print("vllm_turbobus_kv_connector status failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
