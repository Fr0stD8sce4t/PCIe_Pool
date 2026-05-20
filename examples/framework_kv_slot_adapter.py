from turbobus.inference import (
    FrameworkKVSlot,
    FrameworkKVSlotAdapter,
    InferenceKVSlot,
    InferenceKVSlotAdapter,
    make_contiguous_kv_slots,
)


def make_contiguous_slots(prefix: str, count: int, block_bytes: int) -> list[FrameworkKVSlot]:
    return make_contiguous_kv_slots(prefix, count, block_bytes)


def main() -> None:
    raise SystemExit(
        "Import FrameworkKVSlotAdapter from turbobus.inference for real "
        "framework integration."
    )


if __name__ == "__main__":
    main()
