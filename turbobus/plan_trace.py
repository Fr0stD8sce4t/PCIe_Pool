from __future__ import annotations


def transfer_plan_to_dict(plan) -> dict:
    assignments = []
    for assignment in plan.assignments:
        path = assignment.path
        chunks = [
            {
                "src_offset": chunk.src_offset,
                "dst_offset": chunk.dst_offset,
                "bytes": chunk.bytes,
            }
            for chunk in assignment.chunks
        ]
        assignments.append(
            {
                "path": {
                    "kind": path.kind,
                    "direction": path.direction,
                    "target_device": path.target_device,
                    "relay_device": path.relay_device,
                    "h2d_bw_gbps": path.h2d_bw_gbps,
                    "d2h_bw_gbps": path.d2h_bw_gbps,
                    "p2p_bw_gbps": path.p2p_bw_gbps,
                    "effective_bw_gbps": path.effective_bw_gbps,
                    "enabled": path.enabled,
                },
                "chunks": chunks,
                "bytes": sum(chunk["bytes"] for chunk in chunks),
                "chunk_count": len(chunks),
            }
        )
    return {
        "total_bytes": plan.total_bytes,
        "chunk_bytes": plan.chunk_bytes,
        "assignments": assignments,
    }
