from __future__ import annotations

from typing import Iterable

from ..offload_store import (
    BlockState,
    OffloadBatch,
    OffloadBlock,
    OffloadBlockInfo,
    OffloadStore,
)


class ModelWeightLoader(OffloadStore):
    """Model-weight bucket loading API backed by Runtime H2D transfers."""

    def __init__(self, runtime) -> None:
        super().__init__(runtime)

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

    def load_bucket(self, name: str):
        return self.prefetch(name)

    def load_buckets(self, names: Iterable[str]) -> list:
        return self.prefetch_many(names)

    def submit_load_buckets(self, names: Iterable[str]) -> OffloadBatch:
        return self.submit_prefetch_many(names)

    def load_batch(self, names: Iterable[str]) -> OffloadBatch:
        return self.submit_load_buckets(names)

    def load_all(self) -> list:
        return self.prefetch_many(self.names())

    def wait_all(self) -> None:
        self.wait_many(self.names())

    def mark_unloaded(self, names: Iterable[str] | None = None) -> None:
        selected = self.names() if names is None else list(names)
        for name in selected:
            self.set_block_state(name, BlockState.CPU, clear_transfer_state=True)


ModelLoader = ModelWeightLoader
