from __future__ import annotations

import ctypes
import ctypes.util
import json
import socket
import time
from typing import Any

from ..transport import TransferOp, WeightTransferTransport


# libfabric constants (subset)
FI_VERSION = (1 << 16) | 18  # 1.18
FI_REMOTE_READ = 1 << 10
FI_REMOTE_WRITE = 1 << 11
FI_READ = 1 << 8
FI_WRITE = 1 << 9
FI_SEND = 1 << 1
FI_RECV = 1 << 2
FI_AV_MAP = 1


class _FiFabricAttr(ctypes.Structure):
    _fields_ = [
        ("fabric", ctypes.c_void_p),
        ("name", ctypes.c_char_p),
        ("prov_name", ctypes.c_char_p),
        ("prov_version", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
    ]


class _FiInfo(ctypes.Structure):
    pass


_FiInfo._fields_ = [
    ("next", ctypes.POINTER(_FiInfo)),
    ("caps", ctypes.c_uint64),
    ("mode", ctypes.c_uint64),
    ("addr_format", ctypes.c_uint32),
    ("src_addrlen", ctypes.c_size_t),
    ("dest_addrlen", ctypes.c_size_t),
    ("src_addr", ctypes.c_void_p),
    ("dest_addr", ctypes.c_void_p),
    ("fabric_attr", ctypes.POINTER(_FiFabricAttr)),
    ("domain_attr", ctypes.c_void_p),
    ("ep_attr", ctypes.c_void_p),
    ("rx_attr", ctypes.c_void_p),
    ("tx_attr", ctypes.c_void_p),
]


class _FiCqAttr(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("flags", ctypes.c_uint64),
        ("format", ctypes.c_int),
        ("wait_obj", ctypes.c_int),
        ("signaling_vector", ctypes.c_uint64),
        ("wait_cond", ctypes.c_int),
        ("wait_set", ctypes.c_void_p),
        ("wait", ctypes.c_void_p),
    ]


class _FiAvAttr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("rx_ctx_bits", ctypes.c_int),
        ("count", ctypes.c_size_t),
        ("ep_per_node", ctypes.c_size_t),
        ("name", ctypes.c_char_p),
        ("map_addr", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
    ]


def _json_send(sock: socket.socket, payload: dict[str, Any]):
    data = json.dumps(payload).encode() + b"\n"
    sock.sendall(data)


def _json_recv(sock: socket.socket) -> dict[str, Any]:
    f = sock.makefile("rb")
    line = f.readline()
    if not line:
        return {}
    return json.loads(line.decode())


class LibfabricTransport(WeightTransferTransport):
    """libfabric transport with endpoint/CQ/AV setup and fi_write data path."""

    def __init__(self):
        self._lib = None
        self._info = ctypes.POINTER(_FiInfo)()
        self._fabric = ctypes.c_void_p()
        self._domain = ctypes.c_void_p()
        self._cq = ctypes.c_void_p()
        self._av = ctypes.c_void_p()
        self._ep = ctypes.c_void_p()
        self._peer_addr = ctypes.c_uint64(0)
        self._provider = ""
        self._mrs: list[dict[str, Any]] = []
        self._connected = False

    def _load(self):
        if self._lib is not None:
            return
        path = ctypes.util.find_library("fabric")
        if not path:
            raise RuntimeError("libfabric not found")
        self._lib = ctypes.CDLL(path)

        self._lib.fi_getinfo.argtypes = [
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.POINTER(_FiInfo),
            ctypes.POINTER(ctypes.POINTER(_FiInfo)),
        ]
        self._lib.fi_getinfo.restype = ctypes.c_int
        self._lib.fi_freeinfo.argtypes = [ctypes.POINTER(_FiInfo)]
        self._lib.fi_freeinfo.restype = None
        self._lib.fi_fabric.argtypes = [ctypes.POINTER(_FiFabricAttr), ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self._lib.fi_fabric.restype = ctypes.c_int
        self._lib.fi_domain.argtypes = [ctypes.c_void_p, ctypes.POINTER(_FiInfo), ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self._lib.fi_domain.restype = ctypes.c_int
        self._lib.fi_cq_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(_FiCqAttr), ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self._lib.fi_cq_open.restype = ctypes.c_int
        self._lib.fi_av_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(_FiAvAttr), ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self._lib.fi_av_open.restype = ctypes.c_int
        self._lib.fi_endpoint.argtypes = [ctypes.c_void_p, ctypes.POINTER(_FiInfo), ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self._lib.fi_endpoint.restype = ctypes.c_int
        self._lib.fi_ep_bind.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        self._lib.fi_ep_bind.restype = ctypes.c_int
        self._lib.fi_enable.argtypes = [ctypes.c_void_p]
        self._lib.fi_enable.restype = ctypes.c_int
        self._lib.fi_getname.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self._lib.fi_getname.restype = ctypes.c_int
        self._lib.fi_av_insert.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64, ctypes.c_void_p]
        self._lib.fi_av_insert.restype = ctypes.c_long
        self._lib.fi_write.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p]
        self._lib.fi_write.restype = ctypes.c_int
        self._lib.fi_cq_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self._lib.fi_cq_read.restype = ctypes.c_long
        self._lib.fi_mr_reg.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self._lib.fi_mr_reg.restype = ctypes.c_int
        self._lib.fi_mr_key.argtypes = [ctypes.c_void_p]
        self._lib.fi_mr_key.restype = ctypes.c_uint64
        self._lib.fi_mr_desc.argtypes = [ctypes.c_void_p]
        self._lib.fi_mr_desc.restype = ctypes.c_void_p
        self._lib.fi_close.argtypes = [ctypes.c_void_p]
        self._lib.fi_close.restype = ctypes.c_int
        self._lib.fi_strerror.argtypes = [ctypes.c_int]
        self._lib.fi_strerror.restype = ctypes.c_char_p

    def _err(self, rc: int) -> str:
        if self._lib is None:
            return f"rc={rc}"
        try:
            msg = self._lib.fi_strerror(int(rc))
            if msg:
                return msg.decode()
        except Exception:
            pass
        return f"rc={rc}"

    def _pick_info(self, provider: str) -> ctypes.POINTER(_FiInfo) | None:
        p = self._info
        while p:
            try:
                prov = p.contents.fabric_attr.contents.prov_name
                name = prov.decode() if prov else ""
            except Exception:
                name = ""
            if not provider or name == provider:
                self._provider = name
                return p
            p = p.contents.next
        return None

    def _tcp_exchange(self, role: str, payload: dict[str, Any], host: str, port: int, timeout_s: float) -> dict[str, Any]:
        if role == "server":
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((host, port))
                srv.listen(1)
                srv.settimeout(timeout_s)
                conn, _ = srv.accept()
                with conn:
                    peer = _json_recv(conn)
                    _json_send(conn, payload)
                    return peer
        with socket.create_connection((host, port), timeout=timeout_s) as conn:
            _json_send(conn, payload)
            return _json_recv(conn)

    def init_endpoint(self, **kwargs) -> dict[str, Any]:
        try:
            self._load()
        except Exception as exc:
            return {"status": "error", "transport": "libfabric", "reason": str(exc)}

        role = str(kwargs.get("role", "client")).lower()
        if role not in {"server", "client"}:
            role = "client"
        provider = str(kwargs.get("provider", "")).strip()
        node = kwargs.get("node")
        service = kwargs.get("service")
        node_b = str(node).encode() if node else None
        service_b = str(service).encode() if service else None
        timeout_s = float(kwargs.get("tcp_timeout_s", 30.0))
        listen_host = str(kwargs.get("listen_host", "0.0.0.0"))
        listen_port = int(kwargs.get("listen_port", kwargs.get("master_port", 6000)))
        peer_host = str(kwargs.get("peer_host", kwargs.get("master_address", "127.0.0.1")))
        peer_port = int(kwargs.get("peer_port", kwargs.get("master_port", 6000)))

        info = ctypes.POINTER(_FiInfo)()
        rc = self._lib.fi_getinfo(FI_VERSION, node_b, service_b, 0, None, ctypes.byref(info))
        if rc != 0:
            return {"status": "error", "transport": "libfabric", "reason": f"fi_getinfo_failed: {self._err(rc)}"}
        self._info = info

        pick = self._pick_info(provider)
        if pick is None:
            return {"status": "error", "transport": "libfabric", "reason": f"provider_not_found: {provider}"}

        rc = self._lib.fi_fabric(pick.contents.fabric_attr, ctypes.byref(self._fabric), None)
        if rc != 0:
            return {"status": "error", "transport": "libfabric", "reason": f"fi_fabric_failed: {self._err(rc)}"}

        rc = self._lib.fi_domain(self._fabric, pick, ctypes.byref(self._domain), None)
        if rc != 0:
            return {"status": "error", "transport": "libfabric", "reason": f"fi_domain_failed: {self._err(rc)}"}

        cq_attr = _FiCqAttr()
        cq_attr.size = 4096
        cq_attr.flags = 0
        cq_attr.format = 0
        cq_attr.wait_obj = 0
        rc = self._lib.fi_cq_open(self._domain, ctypes.byref(cq_attr), ctypes.byref(self._cq), None)
        if rc != 0:
            return {"status": "error", "transport": "libfabric", "reason": f"fi_cq_open_failed: {self._err(rc)}"}

        av_attr = _FiAvAttr()
        av_attr.type = FI_AV_MAP
        av_attr.count = 16
        rc = self._lib.fi_av_open(self._domain, ctypes.byref(av_attr), ctypes.byref(self._av), None)
        if rc != 0:
            return {"status": "error", "transport": "libfabric", "reason": f"fi_av_open_failed: {self._err(rc)}"}

        rc = self._lib.fi_endpoint(self._domain, pick, ctypes.byref(self._ep), None)
        if rc != 0:
            return {"status": "error", "transport": "libfabric", "reason": f"fi_endpoint_failed: {self._err(rc)}"}

        rc = self._lib.fi_ep_bind(self._ep, self._cq, FI_SEND | FI_RECV)
        if rc != 0:
            return {"status": "error", "transport": "libfabric", "reason": f"fi_ep_bind_cq_failed: {self._err(rc)}"}
        rc = self._lib.fi_ep_bind(self._ep, self._av, 0)
        if rc != 0:
            return {"status": "error", "transport": "libfabric", "reason": f"fi_ep_bind_av_failed: {self._err(rc)}"}
        rc = self._lib.fi_enable(self._ep)
        if rc != 0:
            return {"status": "error", "transport": "libfabric", "reason": f"fi_enable_failed: {self._err(rc)}"}

        addrlen = ctypes.c_size_t(0)
        rc = self._lib.fi_getname(self._ep, None, ctypes.byref(addrlen))
        if addrlen.value == 0:
            return {"status": "error", "transport": "libfabric", "reason": f"fi_getname_size_failed: {self._err(rc)}"}
        addr_buf = ctypes.create_string_buffer(addrlen.value)
        rc = self._lib.fi_getname(self._ep, addr_buf, ctypes.byref(addrlen))
        if rc != 0:
            return {"status": "error", "transport": "libfabric", "reason": f"fi_getname_failed: {self._err(rc)}"}

        local = {"addr_hex": bytes(addr_buf.raw[: addrlen.value]).hex()}
        if role == "server":
            remote = self._tcp_exchange("server", local, listen_host, listen_port, timeout_s)
        else:
            remote = self._tcp_exchange("client", local, peer_host, peer_port, timeout_s)
        if not remote:
            return {"status": "error", "transport": "libfabric", "reason": "tcp_exchange_failed"}

        remote_raw = bytes.fromhex(str(remote.get("addr_hex", "")))
        if not remote_raw:
            return {"status": "error", "transport": "libfabric", "reason": "remote_addr_missing"}
        remote_buf = ctypes.create_string_buffer(remote_raw, len(remote_raw))
        fi_addr = ctypes.c_uint64(0)
        inserted = self._lib.fi_av_insert(self._av, remote_buf, 1, ctypes.byref(fi_addr), 0, None)
        if inserted < 1:
            return {
                "status": "error",
                "transport": "libfabric",
                "reason": f"fi_av_insert_failed: {self._err(int(inserted))}",
            }

        self._peer_addr = fi_addr
        self._connected = True
        return {
            "status": "ok",
            "transport": "libfabric",
            "provider": self._provider,
            "role": role,
        }

    def _ensure_local_mr(self, ptr: int, size: int) -> dict[str, Any] | None:
        end = ptr + size
        for m in self._mrs:
            lo = int(m["ptr"])
            hi = lo + int(m["size"])
            if lo <= ptr and end <= hi:
                return m

        mr = ctypes.c_void_p()
        rc = self._lib.fi_mr_reg(
            self._domain,
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(size),
            ctypes.c_uint64(FI_READ | FI_WRITE),
            ctypes.c_uint64(0),
            ctypes.c_uint64(0),
            ctypes.c_uint64(0),
            ctypes.byref(mr),
            None,
        )
        if rc != 0:
            return None
        desc = self._lib.fi_mr_desc(mr)
        entry = {"ptr": ptr, "size": size, "mr": mr, "desc": desc}
        self._mrs.append(entry)
        return entry

    def register_memory_regions(self, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._domain:
            return []

        access = FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE
        out: list[dict[str, Any]] = []
        for idx, r in enumerate(regions):
            ptr = int(r.get("ptr", 0))
            size = int(r.get("size", 0))
            if ptr <= 0 or size <= 0:
                continue

            mr = ctypes.c_void_p()
            requested_key = int(r.get("requested_key", idx + 1))
            rc = self._lib.fi_mr_reg(
                self._domain,
                ctypes.c_void_p(ptr),
                ctypes.c_size_t(size),
                ctypes.c_uint64(access),
                ctypes.c_uint64(0),
                ctypes.c_uint64(requested_key),
                ctypes.c_uint64(0),
                ctypes.byref(mr),
                None,
            )
            if rc != 0:
                continue

            try:
                rkey = int(self._lib.fi_mr_key(mr))
            except Exception:
                rkey = requested_key
            desc = self._lib.fi_mr_desc(mr)
            self._mrs.append({"ptr": ptr, "size": size, "mr": mr, "desc": desc})
            out.append(
                {
                    "ptr": ptr,
                    "size": size,
                    "rkey": rkey,
                    "mr_handle": int(mr.value or 0),
                    "transport": "libfabric",
                }
            )
        return out

    def transfer(self, ops: list[TransferOp]) -> dict[str, Any]:
        if not self._connected or not self._ep or not self._cq:
            return {
                "status": "error",
                "transport": "libfabric",
                "reason": "transport_not_connected",
                "num_ops": len(ops),
            }

        total_bytes = 0
        sent = 0
        cq_entry = (ctypes.c_uint64 * 8)()

        for i, op in enumerate(ops):
            nbytes = int(op.nbytes)
            if nbytes <= 0:
                continue
            src_ptr = int(op.src_ptr)
            dst_ptr = int(op.dst_ptr)
            dst_rkey = int(op.dst_rkey)
            if src_ptr <= 0 or dst_ptr <= 0 or dst_rkey <= 0:
                return {
                    "status": "error",
                    "transport": "libfabric",
                    "reason": f"invalid_op_fields_at_{i}",
                }

            local = self._ensure_local_mr(src_ptr, nbytes)
            if local is None:
                return {
                    "status": "error",
                    "transport": "libfabric",
                    "reason": f"local_mr_reg_failed_at_{i}",
                }

            rc = self._lib.fi_write(
                self._ep,
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(nbytes),
                local["desc"],
                ctypes.c_uint64(self._peer_addr.value),
                ctypes.c_uint64(dst_ptr),
                ctypes.c_uint64(dst_rkey),
                None,
            )
            if rc != 0:
                return {
                    "status": "error",
                    "transport": "libfabric",
                    "reason": f"fi_write_failed_at_{i}: {self._err(rc)}",
                }

            t0 = time.time()
            while True:
                n = int(self._lib.fi_cq_read(self._cq, ctypes.byref(cq_entry), 1))
                if n > 0:
                    break
                if n < 0 and n not in (-11,):
                    return {
                        "status": "error",
                        "transport": "libfabric",
                        "reason": f"fi_cq_read_failed_at_{i}: {self._err(n)}",
                    }
                if time.time() - t0 > 30.0:
                    return {
                        "status": "error",
                        "transport": "libfabric",
                        "reason": f"cq_timeout_at_{i}",
                    }

            sent += 1
            total_bytes += nbytes

        return {
            "status": "ok",
            "transport": "libfabric",
            "num_ops": sent,
            "total_bytes": total_bytes,
        }

    def close(self):
        if self._lib is None:
            return
        while self._mrs:
            m = self._mrs.pop()
            try:
                self._lib.fi_close(m["mr"])
            except Exception:
                pass
        for h in [self._ep, self._av, self._cq, self._domain, self._fabric]:
            if h:
                try:
                    self._lib.fi_close(h)
                except Exception:
                    pass
        self._ep = ctypes.c_void_p()
        self._av = ctypes.c_void_p()
        self._cq = ctypes.c_void_p()
        self._domain = ctypes.c_void_p()
        self._fabric = ctypes.c_void_p()
        if self._info:
            try:
                self._lib.fi_freeinfo(self._info)
            except Exception:
                pass
            self._info = ctypes.POINTER(_FiInfo)()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
