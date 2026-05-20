from __future__ import annotations

import importlib
import inspect


MODULES = [
    "vllm",
    "vllm.v1.core.kv_cache_manager",
    "vllm.v1.core.kv_cache_utils",
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.gpu_worker",
    "vllm.worker.cache_engine",
]


CLASS_NAMES = [
    "KVCacheManager",
    "KVCacheBlocks",
    "KVCacheConfig",
    "KVCacheGroupSpec",
    "GPUModelRunner",
    "Worker",
    "CacheEngine",
]


KEYWORDS = [
    "kv_cache",
    "kv_caches",
    "gpu_cache",
    "block",
    "block_table",
    "prefix",
    "computed",
    "allocate",
    "cache_config",
]


def main() -> None:
    print("VLLM_INTROSPECT_BEGIN")
    vllm = importlib.import_module("vllm")
    print("vllm_version", getattr(vllm, "__version__", "unknown"))
    for module_name in MODULES:
        describe_module(module_name)
    print("VLLM_INTROSPECT_END")


def describe_module(module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print("module", module_name, "import_error", type(exc).__name__, str(exc))
        return

    print("module", module_name, "path", getattr(module, "__file__", "unknown"))
    for name in CLASS_NAMES:
        obj = getattr(module, name, None)
        if obj is not None:
            describe_class(module_name, name, obj)

    keyword_members = []
    for name, _obj in inspect.getmembers(module):
        lowered = name.lower()
        if any(keyword in lowered for keyword in KEYWORDS):
            keyword_members.append(name)
    if keyword_members:
        print(
            "module_members",
            module_name,
            "names",
            ",".join(keyword_members[:80]),
        )


def describe_class(module_name: str, name: str, cls) -> None:
    try:
        signature = str(inspect.signature(cls))
    except Exception:
        signature = "unknown"
    print("class", module_name, name, "signature", signature)

    interesting = []
    for member_name, member in inspect.getmembers(cls):
        lowered = member_name.lower()
        if any(keyword in lowered for keyword in KEYWORDS):
            try:
                member_signature = str(inspect.signature(member))
            except Exception:
                member_signature = "unknown"
            interesting.append(f"{member_name}{member_signature}")
    if interesting:
        print("class_members", module_name, name, "names", "|".join(interesting[:80]))

    source_file = inspect.getsourcefile(cls)
    if source_file:
        print("class_source", module_name, name, source_file)


if __name__ == "__main__":
    main()
