import ctypes, json, os, socket, sys, time
sys.path.append("src")
from weight_sync.transport import TransferOp, build_transport

def is_master(master: str) -> bool:
    names = {socket.gethostname(), socket.gethostname().split(".")[0], socket.getfqdn()}
    if master in names:
        return True
    try:
        ips = set(socket.gethostbyname_ex(socket.gethostname())[2] + socket.gethostbyname_ex(socket.getfqdn())[2])
        return master in ips or socket.gethostbyname(master) in ips
    except Exception:
        return False

master = os.environ["MASTER_ADDR"]
role = "server" if is_master(master) else "client"
port, ctrl = int(os.getenv("RDMA_PORT", "6000")), int(os.getenv("RDMA_CTRL_PORT", "6001"))
t = build_transport(os.getenv("RDMA_BACKEND", "rdma"))
r = t.init_endpoint(
    role=role,
    master_address=master,
    master_port=port,
    listen_host="0.0.0.0",
    listen_port=port,
    peer_host=master,
    peer_port=port,
    provider=os.getenv("LIBFABRIC_PROVIDER", ""),
)
assert r.get("status") == "ok", f"init failed: {r}"
print(f"RDMA init ok: role={role} transport={r.get('transport')} provider={r.get('provider', '')}")
if role == "server":
    b = ctypes.create_string_buffer(256); ptr = ctypes.addressof(b); mr = t.register_memory_regions([{"ptr": ptr, "size": 256}])[0]
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("0.0.0.0", ctrl)); s.listen(1)
    c, _ = s.accept(); c.sendall((json.dumps({"ptr": ptr, "rkey": int(mr["rkey"]), "nbytes": 256}) + "\n").encode()); c.recv(16); c.close(); s.close()
    print("PASS multinode rdma backend smoke test (server)")
else:
    time.sleep(1); c = socket.create_connection((master, ctrl), timeout=60); meta = json.loads(c.makefile("r").readline())
    src = ctypes.create_string_buffer(b"x" * int(meta["nbytes"])); sp = ctypes.addressof(src)
    tr = t.transfer([TransferOp("w", "w", 0, 0, int(meta["nbytes"]), False, sp, int(meta["ptr"]), int(meta["rkey"]))])
    assert tr.get("status") == "ok", tr
    c.sendall(b"ok"); c.close()
    print("PASS multinode rdma backend smoke test (client)")
