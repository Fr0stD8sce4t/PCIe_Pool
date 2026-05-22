from __future__ import annotations

from typing import Iterable

from .offload_store import (
    BlockState,
    OffloadBlock,
    OffloadBlockInfo,
    OffloadStore,
    TransferStats,
)


class ModelWeightLoader:
    """Model-weight bucket loading API backed by Runtime H2D transfers."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self.store = OffloadStore(runtime)

    def add_bucket(
        self,
        name: str,
        cpu_tensor,
        gpu_tensor=None,
        *,
        bucket_id=None,
        cpu_offset: int = 0,
        gpu_offset: int = 0,
        byte_count: int | None = None,
    ) -> OffloadBlock:
        return self.store.add(
            name,
            cpu_tensor,
            gpu_tensor,
            block_id=name if bucket_id is None else bucket_id,
            cpu_slot=bucket_id,
            gpu_slot=bucket_id,
            cpu_offset=cpu_offset,
            gpu_offset=gpu_offset,
            byte_count=byte_count,
        )

    def add_packed_buckets(
        self,
        prefix: str,
        cpu_tensor,
        gpu_tensor,
        *,
        bucket_bytes: int,
        bucket_count: int,
        start_offset: int = 0,
    ) -> list[OffloadBlock]:
        if bucket_bytes <= 0:
            raise ValueError("bucket_bytes must be positive")
        if bucket_count <= 0:
            raise ValueError("bucket_count must be positive")
        if start_offset < 0:
            raise ValueError("start_offset must be non-negative")

        blocks = []
        for index in range(bucket_count):
            offset = start_offset + index * bucket_bytes
            blocks.append(
                self.add_bucket(
                    f"{prefix}{index}",
                    cpu_tensor,
                    gpu_tensor,
                    bucket_id=index,
                    cpu_offset=offset,
                    gpu_offset=offset,
                    byte_count=bucket_bytes,
                )
            )
        return blocks

    def names(self) -> list[str]:
        return self.store.names()

    def bucket(self, name: str) -> OffloadBlock:
        return self.store.block(name)

    def bucket_info(self, name: str) -> OffloadBlockInfo:
        return self.store.block_info(name)

    def bucket_infos(self, names: Iterable[str] | None = None) -> list[OffloadBlockInfo]:
        return self.store.block_infos(names)

    def load_bucket(self, name: str):
        return self.store.prefetch(name)

    def load_buckets(self, names: Iterable[str]) -> list:
        return self.store.prefetch_many(names)

    def load_all(self) -> list:
        return self.load_buckets(self.names())

    def wait(self, name: str) -> None:
        self.store.wait(name)

    def wait_many(self, names: Iterable[str]) -> None:
        self.store.wait_many(names)

    def wait_all(self) -> None:
        self.wait_many(self.names())

    def transfer_stats(self, name: str) -> TransferStats | None:
        return self.store.transfer_stats(name)

    def transfer_stats_many(self, names: Iterable[str]) -> TransferStats:
        return self.store.transfer_stats_many(names)

    def mark_unloaded(self, names: Iterable[str] | None = None) -> None:
        selected = self.names() if names is None else list(names)
        for name in selected:
            block = self.bucket(name)
            block.state = BlockState.CPU
            block.last_operation = None
            block.last_handle = None
            block.last_prefetch = None


ModelLoader = ModelWeightLoader
