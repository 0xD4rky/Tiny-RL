import sys
sys.path.append("src")
from weight_sync.transport import TransferOp, build_transport

t = build_transport("mock")
t.init_endpoint(role="client")
mr = t.register_memory_regions([{"ptr": 4096, "size": 1024}])[0]
op = TransferOp(src_param="w", dst_param="w", src_off=0, dst_off=0, nbytes=512, src_ptr=4096, dst_ptr=8192, dst_rkey=mr["rkey"])
r = t.transfer([op])
assert r["status"] == "ok" and r["total_bytes"] == 512
print("PASS single-node weight-transfer backend smoke test")
