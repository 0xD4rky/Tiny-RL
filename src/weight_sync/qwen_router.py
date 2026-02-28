from __future__ import annotations

import re
from typing import Any


def _flatten_rollout_meta(rollout_meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for worker in rollout_meta.get("workers", []):
        for p in worker.get("params", []):
            name = str(p.get("name", ""))
            if name and name not in out:
                out[name] = p
    return out


def _flatten_mrs(rollout_mrs: list[dict[str, Any]]) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for r in rollout_mrs:
        ptr = int(r.get("ptr", 0))
        size = int(r.get("size", 0))
        rkey = int(r.get("rkey", 0))
        if ptr > 0 and size > 0 and rkey > 0:
            out.append({"ptr": ptr, "size": size, "rkey": rkey})
    return out


def _find_mr(mrs: list[dict[str, int]], ptr: int, nbytes: int) -> dict[str, int] | None:
    if ptr <= 0 or nbytes <= 0:
        return None
    end = ptr + nbytes
    for mr in mrs:
        lo = mr["ptr"]
        hi = lo + mr["size"]
        if lo <= ptr and end <= hi:
            return mr
    return None


def _route_exact(
    src_by_name: dict[str, dict[str, Any]],
    dst_by_name: dict[str, dict[str, Any]],
    mrs: list[dict[str, int]],
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for name, src in src_by_name.items():
        dst = dst_by_name.get(name)
        if dst is None:
            continue
        nbytes = min(int(src.get("nbytes", 0)), int(dst.get("nbytes", 0)))
        dst_ptr = int(dst.get("ptr", 0))
        mr = _find_mr(mrs, dst_ptr, nbytes)
        if mr is None:
            continue
        routes.append(
            {
                "src_param": name,
                "dst_param": name,
                "src_off": 0,
                "dst_off": 0,
                "nbytes": nbytes,
                "pack": False,
                "dst_ptr": dst_ptr,
                "dst_rkey": int(mr["rkey"]),
            }
        )
    return routes


def _collect_by_suffix(params: dict[str, dict[str, Any]], suffix: str) -> dict[str, dict[str, Any]]:
    return {k: v for k, v in params.items() if k.endswith(suffix)}


def _prefix(name: str, suffix: str) -> str:
    return name[: -len(suffix)]


def _rows(shape: list[int]) -> int:
    return int(shape[0]) if shape else 0


def _cols(shape: list[int]) -> int:
    return int(shape[1]) if len(shape) > 1 else 1


def _elem_size(src_nbytes: int, shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= max(int(d), 1)
    return max(1, int(src_nbytes) // max(n, 1))


def _add_fused_route(
    routes: list[dict[str, Any]],
    src_name: str,
    dst_name: str,
    src: dict[str, Any],
    dst_row_offset: int,
    dst_meta: dict[str, Any],
    mrs: list[dict[str, int]],
):
    shape = [int(x) for x in src.get("shape", [])]
    if len(shape) != 2:
        return
    nrows, ncols = _rows(shape), _cols(shape)
    elem = _elem_size(int(src.get("nbytes", 0)), shape)
    row_bytes = ncols * elem
    dst_base = int(dst_meta.get("ptr", 0))
    dst_ptr = dst_base + dst_row_offset * row_bytes
    nbytes = nrows * row_bytes
    mr = _find_mr(mrs, dst_ptr, nbytes)
    if mr is None:
        return
    routes.append(
        {
            "src_param": src_name,
            "dst_param": dst_name,
            "src_off": 0,
            "dst_off": dst_row_offset * row_bytes,
            "nbytes": nbytes,
            "pack": False,
            "dst_ptr": dst_ptr,
            "dst_rkey": int(mr["rkey"]),
        }
    )


def _route_qkv_fusion(
    src_by_name: dict[str, dict[str, Any]],
    dst_by_name: dict[str, dict[str, Any]],
    mrs: list[dict[str, int]],
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    q_suffix = "q_proj.weight"
    for q_name, q_src in _collect_by_suffix(src_by_name, q_suffix).items():
        root = _prefix(q_name, q_suffix)
        k_name = root + "k_proj.weight"
        v_name = root + "v_proj.weight"
        dst_name = root + "qkv_proj.weight"
        if k_name not in src_by_name or v_name not in src_by_name:
            continue
        if dst_name not in dst_by_name:
            continue
        dst_meta = dst_by_name[dst_name]

        q_rows = _rows([int(x) for x in q_src.get("shape", [])])
        k_rows = _rows([int(x) for x in src_by_name[k_name].get("shape", [])])
        _add_fused_route(routes, q_name, dst_name, q_src, dst_row_offset=0, dst_meta=dst_meta, mrs=mrs)
        _add_fused_route(
            routes,
            k_name,
            dst_name,
            src_by_name[k_name],
            dst_row_offset=q_rows,
            dst_meta=dst_meta,
            mrs=mrs,
        )
        _add_fused_route(
            routes,
            v_name,
            dst_name,
            src_by_name[v_name],
            dst_row_offset=q_rows + k_rows,
            dst_meta=dst_meta,
            mrs=mrs,
        )
    return routes


def _route_gate_up_fusion(
    src_by_name: dict[str, dict[str, Any]],
    dst_by_name: dict[str, dict[str, Any]],
    mrs: list[dict[str, int]],
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    gate_suffix = "gate_proj.weight"
    for gate_name, gate_src in _collect_by_suffix(src_by_name, gate_suffix).items():
        root = _prefix(gate_name, gate_suffix)
        up_name = root + "up_proj.weight"
        dst_name = root + "gate_up_proj.weight"
        if up_name not in src_by_name:
            continue
        if dst_name not in dst_by_name:
            continue
        dst_meta = dst_by_name[dst_name]
        gate_rows = _rows([int(x) for x in gate_src.get("shape", [])])
        _add_fused_route(
            routes,
            gate_name,
            dst_name,
            gate_src,
            dst_row_offset=0,
            dst_meta=dst_meta,
            mrs=mrs,
        )
        _add_fused_route(
            routes,
            up_name,
            dst_name,
            src_by_name[up_name],
            dst_row_offset=gate_rows,
            dst_meta=dst_meta,
            mrs=mrs,
        )
    return routes


def _route_qwen_moe_gate_up_fusion(
    src_by_name: dict[str, dict[str, Any]],
    dst_by_name: dict[str, dict[str, Any]],
    mrs: list[dict[str, int]],
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    pat = re.compile(r"(.*experts\.\d+\.)gate_proj\.weight$")
    for src_name, src in src_by_name.items():
        m = pat.match(src_name)
        if not m:
            continue
        root = m.group(1)
        up_name = root + "up_proj.weight"
        dst_name = root + "gate_up_proj.weight"
        if up_name not in src_by_name or dst_name not in dst_by_name:
            continue
        dst_meta = dst_by_name[dst_name]
        gate_rows = _rows([int(x) for x in src.get("shape", [])])
        _add_fused_route(
            routes,
            src_name,
            dst_name,
            src,
            dst_row_offset=0,
            dst_meta=dst_meta,
            mrs=mrs,
        )
        _add_fused_route(
            routes,
            up_name,
            dst_name,
            src_by_name[up_name],
            dst_row_offset=gate_rows,
            dst_meta=dst_meta,
            mrs=mrs,
        )
    return routes


def build_qwen_routing_table(
    trainer_routes: list[dict[str, Any]],
    rollout_meta: dict[str, Any],
    rollout_mrs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build route table for Qwen dense+MoE models.

    Current strategy:
    - exact-name matches
    - q/k/v -> qkv fusion
    - gate/up -> gate_up fusion
    - moe experts gate/up -> gate_up fusion
    """
    src_by_name = {str(r.get("src_param", "")): r for r in trainer_routes if r.get("src_param")}
    dst_by_name = _flatten_rollout_meta(rollout_meta)
    mrs = _flatten_mrs(rollout_mrs)

    routes = _route_exact(src_by_name, dst_by_name, mrs)

    routed_pairs = {(r["src_param"], r["dst_param"]) for r in routes}

    for extra in (
        _route_qkv_fusion(src_by_name, dst_by_name, mrs)
        + _route_gate_up_fusion(src_by_name, dst_by_name, mrs)
        + _route_qwen_moe_gate_up_fusion(src_by_name, dst_by_name, mrs)
    ):
        key = (extra["src_param"], extra["dst_param"])
        if key in routed_pairs:
            continue
        routes.append(extra)
        routed_pairs.add(key)

    return routes
