from __future__ import annotations

import ctypes
import ctypes.util
import json
import random
import socket
import time
from typing import Any

from ..transport import TransferOp, WeightTransferTransport


# Access flags from ibv_reg_mr(3)
IBV_ACCESS_LOCAL_WRITE = 1 << 0
IBV_ACCESS_REMOTE_WRITE = 1 << 1
IBV_ACCESS_REMOTE_READ = 1 << 2

# QP type / states / opcodes
IBV_QPT_RC = 2
IBV_QPS_INIT = 1
IBV_QPS_RTR = 2
IBV_QPS_RTS = 3
IBV_WR_RDMA_WRITE = 0
IBV_SEND_SIGNALED = 1

# MTU enums
IBV_MTU_1024 = 3

# QP attr masks
IBV_QP_STATE = 1 << 0
IBV_QP_ACCESS_FLAGS = 1 << 3
IBV_QP_PKEY_INDEX = 1 << 4
IBV_QP_PORT = 1 << 5
IBV_QP_AV = 1 << 7
IBV_QP_PATH_MTU = 1 << 8
IBV_QP_TIMEOUT = 1 << 9
IBV_QP_RETRY_CNT = 1 << 10
IBV_QP_RNR_RETRY = 1 << 11
IBV_QP_RQ_PSN = 1 << 12
IBV_QP_MAX_QP_RD_ATOMIC = 1 << 13
IBV_QP_MIN_RNR_TIMER = 1 << 15
IBV_QP_SQ_PSN = 1 << 16
IBV_QP_MAX_DEST_RD_ATOMIC = 1 << 17
IBV_QP_DEST_QPN = 1 << 20


class _IbvGid(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_uint8 * 16)]


class _IbvGlobalRoute(ctypes.Structure):
    _fields_ = [
        ("dgid", _IbvGid),
        ("flow_label", ctypes.c_uint32),
        ("sgid_index", ctypes.c_uint8),
        ("hop_limit", ctypes.c_uint8),
        ("traffic_class", ctypes.c_uint8),
    ]


class _IbvAhAttr(ctypes.Structure):
    _fields_ = [
        ("grh", _IbvGlobalRoute),
        ("dlid", ctypes.c_uint16),
        ("sl", ctypes.c_uint8),
        ("src_path_bits", ctypes.c_uint8),
        ("static_rate", ctypes.c_uint8),
        ("is_global", ctypes.c_uint8),
        ("port_num", ctypes.c_uint8),
    ]


class _IbvQpCap(ctypes.Structure):
    _fields_ = [
        ("max_send_wr", ctypes.c_uint32),
        ("max_recv_wr", ctypes.c_uint32),
        ("max_send_sge", ctypes.c_uint32),
        ("max_recv_sge", ctypes.c_uint32),
        ("max_inline_data", ctypes.c_uint32),
    ]


class _IbvQpInitAttr(ctypes.Structure):
    _fields_ = [
        ("qp_context", ctypes.c_void_p),
        ("send_cq", ctypes.c_void_p),
        ("recv_cq", ctypes.c_void_p),
        ("srq", ctypes.c_void_p),
        ("cap", _IbvQpCap),
        ("qp_type", ctypes.c_int),
        ("sq_sig_all", ctypes.c_int),
    ]


class _IbvQpAttr(ctypes.Structure):
    _fields_ = [
        ("qp_state", ctypes.c_int),
        ("cur_qp_state", ctypes.c_int),
        ("path_mtu", ctypes.c_int),
        ("path_mig_state", ctypes.c_int),
        ("qkey", ctypes.c_uint32),
        ("rq_psn", ctypes.c_uint32),
        ("sq_psn", ctypes.c_uint32),
        ("dest_qp_num", ctypes.c_uint32),
        ("qp_access_flags", ctypes.c_int),
        ("cap", _IbvQpCap),
        ("ah_attr", _IbvAhAttr),
        ("alt_ah_attr", _IbvAhAttr),
        ("pkey_index", ctypes.c_uint16),
        ("alt_pkey_index", ctypes.c_uint16),
        ("en_sqd_async_notify", ctypes.c_uint8),
        ("sq_draining", ctypes.c_uint8),
        ("max_rd_atomic", ctypes.c_uint8),
        ("max_dest_rd_atomic", ctypes.c_uint8),
        ("min_rnr_timer", ctypes.c_uint8),
        ("port_num", ctypes.c_uint8),
        ("timeout", ctypes.c_uint8),
        ("retry_cnt", ctypes.c_uint8),
        ("rnr_retry", ctypes.c_uint8),
        ("alt_port_num", ctypes.c_uint8),
        ("alt_timeout", ctypes.c_uint8),
    ]


class _IbvSge(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint64),
        ("length", ctypes.c_uint32),
        ("lkey", ctypes.c_uint32),
    ]


class _IbvWrRdma(ctypes.Structure):
    _fields_ = [("remote_addr", ctypes.c_uint64), ("rkey", ctypes.c_uint32)]


class _IbvSendWrWr(ctypes.Union):
    _fields_ = [("rdma", _IbvWrRdma)]


class _IbvSendWr(ctypes.Structure):
    pass


_IbvSendWr._fields_ = [
    ("wr_id", ctypes.c_uint64),
    ("next", ctypes.POINTER(_IbvSendWr)),
    ("sg_list", ctypes.POINTER(_IbvSge)),
    ("num_sge", ctypes.c_int),
    ("opcode", ctypes.c_int),
    ("send_flags", ctypes.c_int),
    ("imm_data_invalidated_rkey_union", ctypes.c_uint32),
    ("wr", _IbvSendWrWr),
]


class _IbvWc(ctypes.Structure):
    _fields_ = [
        ("wr_id", ctypes.c_uint64),
        ("status", ctypes.c_uint32),
        ("opcode", ctypes.c_uint32),
        ("vendor_err", ctypes.c_uint32),
        ("byte_len", ctypes.c_uint32),
        ("imm_data_invalidated_rkey_union", ctypes.c_uint32),
        ("qp_num", ctypes.c_uint32),
        ("src_qp", ctypes.c_uint32),
        ("wc_flags", ctypes.c_uint32),
        ("pkey_index", ctypes.c_uint16),
        ("slid", ctypes.c_uint16),
        ("sl", ctypes.c_uint8),
        ("dlid_path_bits", ctypes.c_uint8),
    ]


class _IbvPortAttr(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_uint8),
        ("max_mtu", ctypes.c_uint8),
        ("active_mtu", ctypes.c_uint8),
        ("gid_tbl_len", ctypes.c_int),
        ("port_cap_flags", ctypes.c_uint32),
        ("max_msg_sz", ctypes.c_uint32),
        ("bad_pkey_cntr", ctypes.c_uint32),
        ("qkey_viol_cntr", ctypes.c_uint32),
        ("pkey_tbl_len", ctypes.c_uint16),
        ("lid", ctypes.c_uint16),
        ("sm_lid", ctypes.c_uint16),
        ("lmc", ctypes.c_uint8),
        ("max_vl_num", ctypes.c_uint8),
        ("sm_sl", ctypes.c_uint8),
        ("subnet_timeout", ctypes.c_uint8),
        ("init_type_reply", ctypes.c_uint8),
        ("active_width", ctypes.c_uint8),
        ("active_speed", ctypes.c_uint8),
        ("phys_state", ctypes.c_uint8),
        ("link_layer", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("port_cap_flags2", ctypes.c_uint16),
    ]


class _IbvMr(ctypes.Structure):
    _fields_ = [
        ("context", ctypes.c_void_p),
        ("pd", ctypes.c_void_p),
        ("addr", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
        ("handle", ctypes.c_uint32),
        ("lkey", ctypes.c_uint32),
        ("rkey", ctypes.c_uint32),
    ]


class _IbvQp(ctypes.Structure):
    _fields_ = [
        ("context", ctypes.c_void_p),
        ("qp_context", ctypes.c_void_p),
        ("pd", ctypes.c_void_p),
        ("send_cq", ctypes.c_void_p),
        ("recv_cq", ctypes.c_void_p),
        ("srq", ctypes.c_void_p),
        ("handle", ctypes.c_uint32),
        ("qp_num", ctypes.c_uint32),
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


class IbverbsTransport(WeightTransferTransport):
    """ibverbs transport with RC QP setup and RDMA write data path."""

    def __init__(self):
        self._lib = None
        self._ctx = ctypes.c_void_p()
        self._pd = ctypes.c_void_p()
        self._cq = ctypes.c_void_p()
        self._qp: ctypes.POINTER(_IbvQp) | None = None
        self._device_name = ""
        self._port_num = 1
        self._gid_index = 0
        self._connected = False
        self._local_mrs: list[dict[str, Any]] = []

    def _load(self):
        if self._lib is not None:
            return
        lib_path = ctypes.util.find_library("ibverbs")
        if not lib_path:
            raise RuntimeError("libibverbs not found")
        self._lib = ctypes.CDLL(lib_path)

        self._lib.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
        self._lib.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.ibv_free_device_list.restype = None
        self._lib.ibv_get_device_name.argtypes = [ctypes.c_void_p]
        self._lib.ibv_get_device_name.restype = ctypes.c_char_p
        self._lib.ibv_open_device.argtypes = [ctypes.c_void_p]
        self._lib.ibv_open_device.restype = ctypes.c_void_p
        self._lib.ibv_close_device.argtypes = [ctypes.c_void_p]
        self._lib.ibv_close_device.restype = ctypes.c_int
        self._lib.ibv_alloc_pd.argtypes = [ctypes.c_void_p]
        self._lib.ibv_alloc_pd.restype = ctypes.c_void_p
        self._lib.ibv_dealloc_pd.argtypes = [ctypes.c_void_p]
        self._lib.ibv_dealloc_pd.restype = ctypes.c_int
        self._lib.ibv_create_cq.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self._lib.ibv_create_cq.restype = ctypes.c_void_p
        self._lib.ibv_destroy_cq.argtypes = [ctypes.c_void_p]
        self._lib.ibv_destroy_cq.restype = ctypes.c_int
        self._lib.ibv_create_qp.argtypes = [ctypes.c_void_p, ctypes.POINTER(_IbvQpInitAttr)]
        self._lib.ibv_create_qp.restype = ctypes.POINTER(_IbvQp)
        self._lib.ibv_destroy_qp.argtypes = [ctypes.POINTER(_IbvQp)]
        self._lib.ibv_destroy_qp.restype = ctypes.c_int
        self._lib.ibv_modify_qp.argtypes = [ctypes.POINTER(_IbvQp), ctypes.POINTER(_IbvQpAttr), ctypes.c_int]
        self._lib.ibv_modify_qp.restype = ctypes.c_int
        self._lib.ibv_query_port.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.POINTER(_IbvPortAttr)]
        self._lib.ibv_query_port.restype = ctypes.c_int
        self._lib.ibv_query_gid.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_int, ctypes.POINTER(_IbvGid)]
        self._lib.ibv_query_gid.restype = ctypes.c_int
        self._lib.ibv_reg_mr.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self._lib.ibv_reg_mr.restype = ctypes.POINTER(_IbvMr)
        self._lib.ibv_dereg_mr.argtypes = [ctypes.POINTER(_IbvMr)]
        self._lib.ibv_dereg_mr.restype = ctypes.c_int
        self._lib.ibv_post_send.argtypes = [ctypes.POINTER(_IbvQp), ctypes.POINTER(_IbvSendWr), ctypes.POINTER(ctypes.POINTER(_IbvSendWr))]
        self._lib.ibv_post_send.restype = ctypes.c_int
        self._lib.ibv_poll_cq.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_IbvWc)]
        self._lib.ibv_poll_cq.restype = ctypes.c_int

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

    def _setup_qp(self):
        self._cq = self._lib.ibv_create_cq(self._ctx, 4096, None, None, 0)
        if not self._cq:
            raise RuntimeError("ibv_create_cq failed")

        init_attr = _IbvQpInitAttr()
        init_attr.send_cq = self._cq
        init_attr.recv_cq = self._cq
        init_attr.cap.max_send_wr = 4096
        init_attr.cap.max_recv_wr = 1
        init_attr.cap.max_send_sge = 1
        init_attr.cap.max_recv_sge = 1
        init_attr.cap.max_inline_data = 0
        init_attr.qp_type = IBV_QPT_RC
        init_attr.sq_sig_all = 0

        self._qp = self._lib.ibv_create_qp(self._pd, ctypes.byref(init_attr))
        if not self._qp:
            raise RuntimeError("ibv_create_qp failed")

    def _qp_to_init(self):
        attr = _IbvQpAttr()
        attr.qp_state = IBV_QPS_INIT
        attr.pkey_index = 0
        attr.port_num = self._port_num
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE
        mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS
        rc = self._lib.ibv_modify_qp(self._qp, ctypes.byref(attr), mask)
        if rc != 0:
            raise RuntimeError(f"ibv_modify_qp INIT failed rc={rc}")

    def _qp_to_rtr(self, remote: dict[str, Any]):
        attr = _IbvQpAttr()
        attr.qp_state = IBV_QPS_RTR
        attr.path_mtu = IBV_MTU_1024
        attr.dest_qp_num = int(remote["qp_num"])
        attr.rq_psn = int(remote["psn"])
        attr.max_dest_rd_atomic = 1
        attr.min_rnr_timer = 12
        attr.ah_attr.dlid = int(remote.get("lid", 0))
        attr.ah_attr.sl = 0
        attr.ah_attr.src_path_bits = 0
        attr.ah_attr.port_num = self._port_num

        gid_hex = str(remote.get("gid", ""))
        if gid_hex and gid_hex != "0" * 32:
            attr.ah_attr.is_global = 1
            gid_bytes = bytes.fromhex(gid_hex)
            for i in range(min(16, len(gid_bytes))):
                attr.ah_attr.grh.dgid.raw[i] = gid_bytes[i]
            attr.ah_attr.grh.flow_label = 0
            attr.ah_attr.grh.hop_limit = 1
            attr.ah_attr.grh.traffic_class = 0
            attr.ah_attr.grh.sgid_index = self._gid_index

        mask = (
            IBV_QP_STATE
            | IBV_QP_AV
            | IBV_QP_PATH_MTU
            | IBV_QP_DEST_QPN
            | IBV_QP_RQ_PSN
            | IBV_QP_MAX_DEST_RD_ATOMIC
            | IBV_QP_MIN_RNR_TIMER
        )
        rc = self._lib.ibv_modify_qp(self._qp, ctypes.byref(attr), mask)
        if rc != 0:
            raise RuntimeError(f"ibv_modify_qp RTR failed rc={rc}")

    def _qp_to_rts(self, local_psn: int):
        attr = _IbvQpAttr()
        attr.qp_state = IBV_QPS_RTS
        attr.timeout = 14
        attr.retry_cnt = 7
        attr.rnr_retry = 7
        attr.sq_psn = int(local_psn)
        attr.max_rd_atomic = 1

        mask = (
            IBV_QP_STATE
            | IBV_QP_TIMEOUT
            | IBV_QP_RETRY_CNT
            | IBV_QP_RNR_RETRY
            | IBV_QP_SQ_PSN
            | IBV_QP_MAX_QP_RD_ATOMIC
        )
        rc = self._lib.ibv_modify_qp(self._qp, ctypes.byref(attr), mask)
        if rc != 0:
            raise RuntimeError(f"ibv_modify_qp RTS failed rc={rc}")

    def init_endpoint(self, **kwargs) -> dict[str, Any]:
        try:
            self._load()
        except Exception as exc:
            return {"status": "error", "transport": "ibverbs", "reason": str(exc)}

        role = str(kwargs.get("role", "client")).lower()
        if role not in {"server", "client"}:
            role = "client"

        self._port_num = int(kwargs.get("ib_port", 1))
        self._gid_index = int(kwargs.get("gid_index", 0))
        timeout_s = float(kwargs.get("tcp_timeout_s", 30.0))

        listen_host = str(kwargs.get("listen_host", "0.0.0.0"))
        listen_port = int(kwargs.get("listen_port", kwargs.get("master_port", 6000)))
        peer_host = str(kwargs.get("peer_host", kwargs.get("master_address", "127.0.0.1")))
        peer_port = int(kwargs.get("peer_port", kwargs.get("master_port", 6000)))

        device_name = str(kwargs.get("device_name", "")).strip()
        device_index = int(kwargs.get("device_index", 0))

        num = ctypes.c_int(0)
        dev_list = self._lib.ibv_get_device_list(ctypes.byref(num))
        if not dev_list or num.value <= 0:
            return {"status": "error", "transport": "ibverbs", "reason": "no_ibverbs_devices"}

        chosen = ctypes.c_void_p()
        chosen_name = ""
        try:
            if device_name:
                for i in range(num.value):
                    dev = dev_list[i]
                    name = self._lib.ibv_get_device_name(dev)
                    if name and name.decode() == device_name:
                        chosen = dev
                        chosen_name = device_name
                        break
            else:
                i = max(0, min(device_index, num.value - 1))
                chosen = dev_list[i]
                name = self._lib.ibv_get_device_name(chosen)
                chosen_name = name.decode() if name else f"index-{i}"

            if not chosen:
                return {"status": "error", "transport": "ibverbs", "reason": f"device_not_found: {device_name}"}

            ctx = self._lib.ibv_open_device(chosen)
            if not ctx:
                return {"status": "error", "transport": "ibverbs", "reason": f"ibv_open_device_failed: {chosen_name}"}
            self._ctx = ctypes.c_void_p(ctx)

            pd = self._lib.ibv_alloc_pd(self._ctx)
            if not pd:
                self._lib.ibv_close_device(self._ctx)
                self._ctx = ctypes.c_void_p()
                return {"status": "error", "transport": "ibverbs", "reason": f"ibv_alloc_pd_failed: {chosen_name}"}
            self._pd = ctypes.c_void_p(pd)
            self._device_name = chosen_name

            self._setup_qp()
            self._qp_to_init()

            port_attr = _IbvPortAttr()
            rc = self._lib.ibv_query_port(self._ctx, ctypes.c_uint8(self._port_num), ctypes.byref(port_attr))
            if rc != 0:
                return {"status": "error", "transport": "ibverbs", "reason": f"ibv_query_port_failed: rc={rc}"}
            lid = int(port_attr.lid)

            gid = _IbvGid()
            rc = self._lib.ibv_query_gid(
                self._ctx,
                ctypes.c_uint8(self._port_num),
                ctypes.c_int(self._gid_index),
                ctypes.byref(gid),
            )
            if rc != 0:
                return {"status": "error", "transport": "ibverbs", "reason": f"ibv_query_gid_failed: rc={rc}"}

            local_psn = random.randint(0, (1 << 24) - 1)
            local = {
                "qp_num": int(self._qp.contents.qp_num),
                "psn": int(local_psn),
                "lid": int(lid),
                "gid": bytes(gid.raw).hex(),
                "port": int(self._port_num),
                "gid_index": int(self._gid_index),
            }

            if role == "server":
                remote = self._tcp_exchange("server", local, listen_host, listen_port, timeout_s)
            else:
                remote = self._tcp_exchange("client", local, peer_host, peer_port, timeout_s)
            if not remote:
                return {"status": "error", "transport": "ibverbs", "reason": "tcp_exchange_failed"}

            self._qp_to_rtr(remote)
            self._qp_to_rts(local_psn)
            self._connected = True
            return {
                "status": "ok",
                "transport": "ibverbs",
                "device": chosen_name,
                "role": role,
                "local": local,
                "remote": remote,
            }
        except Exception as exc:
            return {"status": "error", "transport": "ibverbs", "reason": str(exc)}
        finally:
            self._lib.ibv_free_device_list(dev_list)

    def _ensure_local_mr(self, ptr: int, size: int) -> ctypes.POINTER(_IbvMr) | None:
        end = ptr + size
        for x in self._local_mrs:
            lo = int(x["ptr"])
            hi = lo + int(x["size"])
            if lo <= ptr and end <= hi:
                return x["mr"]

        access = IBV_ACCESS_LOCAL_WRITE
        mr = self._lib.ibv_reg_mr(
            self._pd,
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(size),
            access,
        )
        if not mr:
            return None
        self._local_mrs.append({"ptr": ptr, "size": size, "mr": mr})
        return mr

    def register_memory_regions(self, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._pd:
            return []

        access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
        out: list[dict[str, Any]] = []
        for r in regions:
            size = int(r.get("size", 0))
            ptr = int(r.get("ptr", 0))
            if size <= 0 or ptr <= 0:
                continue
            mr = self._lib.ibv_reg_mr(self._pd, ctypes.c_void_p(ptr), ctypes.c_size_t(size), access)
            if not mr:
                continue
            self._local_mrs.append({"ptr": ptr, "size": size, "mr": mr})
            out.append(
                {
                    "ptr": ptr,
                    "size": size,
                    "lkey": int(mr.contents.lkey),
                    "rkey": int(mr.contents.rkey),
                    "mr_handle": int(ctypes.cast(mr, ctypes.c_void_p).value or 0),
                    "transport": "ibverbs",
                }
            )
        return out

    def transfer(self, ops: list[TransferOp]) -> dict[str, Any]:
        if not self._connected or not self._qp or not self._cq:
            return {
                "status": "error",
                "transport": "ibverbs",
                "reason": "transport_not_connected",
                "num_ops": len(ops),
            }

        total_bytes = 0
        sent = 0
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
                    "transport": "ibverbs",
                    "reason": f"invalid_op_fields_at_{i}",
                }

            mr = self._ensure_local_mr(src_ptr, nbytes)
            if mr is None:
                return {
                    "status": "error",
                    "transport": "ibverbs",
                    "reason": f"local_mr_reg_failed_at_{i}",
                }

            sge = _IbvSge()
            sge.addr = ctypes.c_uint64(src_ptr)
            sge.length = ctypes.c_uint32(nbytes)
            sge.lkey = ctypes.c_uint32(int(mr.contents.lkey))

            wr = _IbvSendWr()
            wr.wr_id = ctypes.c_uint64(i + 1)
            wr.next = ctypes.POINTER(_IbvSendWr)()
            wr.sg_list = ctypes.pointer(sge)
            wr.num_sge = 1
            wr.opcode = IBV_WR_RDMA_WRITE
            wr.send_flags = IBV_SEND_SIGNALED
            wr.wr.rdma.remote_addr = ctypes.c_uint64(dst_ptr)
            wr.wr.rdma.rkey = ctypes.c_uint32(dst_rkey)

            bad_wr = ctypes.POINTER(_IbvSendWr)()
            rc = self._lib.ibv_post_send(self._qp, ctypes.byref(wr), ctypes.byref(bad_wr))
            if rc != 0:
                return {
                    "status": "error",
                    "transport": "ibverbs",
                    "reason": f"ibv_post_send_failed_at_{i}: rc={rc}",
                }

            wc = _IbvWc()
            t0 = time.time()
            while True:
                n = self._lib.ibv_poll_cq(self._cq, 1, ctypes.byref(wc))
                if n < 0:
                    return {
                        "status": "error",
                        "transport": "ibverbs",
                        "reason": f"ibv_poll_cq_failed_at_{i}",
                    }
                if n == 0:
                    if time.time() - t0 > 30.0:
                        return {
                            "status": "error",
                            "transport": "ibverbs",
                            "reason": f"cq_timeout_at_{i}",
                        }
                    continue
                if int(wc.status) != 0:
                    return {
                        "status": "error",
                        "transport": "ibverbs",
                        "reason": f"wc_error_at_{i}: status={int(wc.status)} vendor={int(wc.vendor_err)}",
                    }
                break

            sent += 1
            total_bytes += nbytes

        return {
            "status": "ok",
            "transport": "ibverbs",
            "num_ops": sent,
            "total_bytes": total_bytes,
        }

    def close(self):
        if self._lib is None:
            return
        while self._local_mrs:
            mr = self._local_mrs.pop()["mr"]
            try:
                self._lib.ibv_dereg_mr(mr)
            except Exception:
                pass
        if self._qp:
            try:
                self._lib.ibv_destroy_qp(self._qp)
            except Exception:
                pass
            self._qp = None
        if self._cq:
            try:
                self._lib.ibv_destroy_cq(self._cq)
            except Exception:
                pass
            self._cq = ctypes.c_void_p()
        if self._pd:
            try:
                self._lib.ibv_dealloc_pd(self._pd)
            except Exception:
                pass
            self._pd = ctypes.c_void_p()
        if self._ctx:
            try:
                self._lib.ibv_close_device(self._ctx)
            except Exception:
                pass
            self._ctx = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
