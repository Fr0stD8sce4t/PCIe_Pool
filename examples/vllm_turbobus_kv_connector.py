from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_relay_gpus(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def default_log_path() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("benchmarks") / "results" / f"vllm_turbobus_kv_connector_{stamp}.log")


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


def second_prompt_text(args, first_prompt: str) -> str:
    return first_prompt + args.second_prompt_suffix


def run(args) -> None:
    configure_cuda_devices(args)
    if args.disable_multiproc_executor:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    from turbobus.vllm_kv_connector import clear_saved_prefixes, get_saved_prefix

    torch.cuda.set_device(args.runtime_target_gpu)
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
                "turbobus.do_save": args.restore_enabled,
                "turbobus.prefix_key": args.prefix_key,
                "turbobus.save_blocks": args.restore_blocks,
                "turbobus.matched_tokens": args.matched_tokens,
            }
        },
    )
    first_outputs = llm.generate([first_prompt], base_sampling)

    first_saved = get_saved_prefix(args.prefix_key, args.session_id)
    restore_prefix_key = args.prefix_key
    restore_base_prompt = first_prompt
    saved = first_saved
    second_saved = None
    if args.second_save_prefix_key:
        second_save_prompt = prompt_text(args.second_save_prompt, args.prompt_repeat)
        second_save_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            extra_args={
                "kv_transfer_params": {
                    "turbobus.do_save": args.restore_enabled,
                    "turbobus.prefix_key": args.second_save_prefix_key,
                    "turbobus.save_blocks": args.restore_blocks,
                    "turbobus.matched_tokens": args.matched_tokens,
                }
            },
        )
        llm.generate([second_save_prompt], second_save_sampling)
        second_saved = get_saved_prefix(args.second_save_prefix_key, args.session_id)
        restore_prefix_key = args.second_save_prefix_key
        restore_base_prompt = second_save_prompt
        saved = second_saved

    source_request_id = saved.source_request_id if saved is not None else "unknown"
    source_blocks = saved.block_count if saved is not None else 0
    if args.restore_enabled:
        if saved is None:
            raise RuntimeError(
                f"connector did not save prefix {restore_prefix_key!r}; "
                "check turbobus.do_save and vLLM wait_for_save"
            )
        if saved.block_count < args.restore_blocks:
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
    generated_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""

    print("COPY_SUMMARY_BEGIN")
    print(
        "vllm_kv_connector_config",
        f"target={args.target_gpu}",
        f"relays={parse_relay_gpus(args.relay_gpus)}",
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
        f"restore_enabled={args.restore_enabled}",
        f"max_saved_prefixes={args.max_saved_prefixes}",
        f"chunk_bytes={args.chunk_bytes}",
        f"mode={args.mode}",
    )
    print(
        "vllm_kv_connector_scenario",
        "type=real_vllm_kv_transfer_connector",
        "boundary=KVConnectorBase_V1",
        "entry=start_load_kv",
        "note=official_vllm_external_kv_path",
    )
    if saved is not None:
        print(
            "vllm_kv_connector_save",
            f"source_request={source_request_id}",
            f"source_blocks={source_blocks}",
            f"blocks={saved.block_count}",
            f"bytes={saved.bytes}",
            f"elapsed_ms={saved.elapsed_ms:.3f}",
            f"runtime_init_ms={saved.runtime_init_ms:.3f}",
            f"prepare_ms={saved.prepare_ms:.3f}",
            f"cpu_alloc_ms={saved.cpu_alloc_ms:.3f}",
            f"group_ms={saved.group_ms:.3f}",
            f"adapter_ms={saved.adapter_ms:.3f}",
            f"refs_ms={saved.refs_ms:.3f}",
            f"transfer_ms={saved.transfer_ms:.3f}",
            f"register_ms={saved.register_ms:.3f}",
            f"total_ms={saved.total_ms:.3f}",
            f"direct_chunks={saved.direct_chunks}",
            f"relay_chunks={saved.relay_chunks}",
            f"save_layer_count={saved.save_layer_count}",
            f"save_layer_ranges={saved.save_layer_ranges}",
        )
    if first_saved is not None and first_saved is not saved:
        print(
            "vllm_kv_connector_first_save",
            f"source_request={first_saved.source_request_id}",
            f"blocks={first_saved.block_count}",
            f"bytes={first_saved.bytes}",
            f"elapsed_ms={first_saved.elapsed_ms:.3f}",
        )
    if second_saved is not None and second_saved is not saved:
        print(
            "vllm_kv_connector_second_save",
            f"source_request={second_saved.source_request_id}",
            f"blocks={second_saved.block_count}",
            f"bytes={second_saved.bytes}",
            f"elapsed_ms={second_saved.elapsed_ms:.3f}",
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
        "--restore-enabled",
        action="store_true",
        help="Actually write TurboBus CPU backing into vLLM KV slots. Keep off until saved backing data is available.",
    )
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
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
