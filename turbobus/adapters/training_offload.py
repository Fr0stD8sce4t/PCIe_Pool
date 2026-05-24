from __future__ import annotations

from typing import Iterable

from ..offload_store import (
    AdapterTransferContext,
    BlockState,
    OffloadBatch,
    OffloadBlock,
    OffloadBlockInfo,
    OffloadStore,
)


class TrainingOffloadManager(OffloadStore):
    """Parameter or optimizer bucket movement API backed by daemon transfer intent."""

    def __init__(
        self,
        client,
        transfer_context: AdapterTransferContext,
    ) -> None:
        super().__init__(client, transfer_context)

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
        return self.add(
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

    def bucket(self, name: str) -> OffloadBlock:
        return self.get_block(name)

    def bucket_info(self, name: str) -> OffloadBlockInfo:
        return self.block_info(name)

    def bucket_infos(self, names: Iterable[str] | None = None) -> list[OffloadBlockInfo]:
        return self.block_infos(names)

    def prefetch_bucket(self, name: str):
        return self.prefetch(name)

    def prefetch_buckets(self, names: Iterable[str]) -> list:
        return self.prefetch_many(names)

    def submit_prefetch_buckets(self, names: Iterable[str]) -> OffloadBatch:
        return self.submit_prefetch_many(names)

    def prefetch_batch(self, names: Iterable[str]) -> OffloadBatch:
        return self.submit_prefetch_buckets(names)

    def prefetch_all(self) -> list:
        return self.prefetch_buckets(self.names())

    def offload_bucket(self, name: str):
        return self.evict(name)

    def offload_buckets(self, names: Iterable[str]) -> list:
        return self.evict_many(names)

    def submit_offload_buckets(self, names: Iterable[str]) -> OffloadBatch:
        return self.submit_evict_many(names)

    def offload_batch(self, names: Iterable[str]) -> OffloadBatch:
        return self.submit_offload_buckets(names)

    def offload_all(self) -> list:
        return self.evict_many(self.names())

    def wait_all(self) -> None:
        self.wait_many(self.names())

    def mark_on_cpu(self, names: Iterable[str] | None = None) -> None:
        selected = self.names() if names is None else list(names)
        for name in selected:
            self.set_block_state(name, BlockState.CPU, clear_transfer_state=True)

    def mark_on_gpu(self, names: Iterable[str] | None = None) -> None:
        selected = self.names() if names is None else list(names)
        for name in selected:
            self.set_block_state(name, BlockState.GPU, clear_transfer_state=True)


TrainingOffloadStore = TrainingOffloadManager
