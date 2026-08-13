#!/usr/bin/env python
import argparse
import importlib.util
import os
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
EXT = Path(os.environ.get("HEAD64_DECODE_EXT", ROOT / "build/head64_decode_test_ext.so"))
spec = importlib.util.spec_from_file_location("head64_decode_test_ext", EXT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"missing {EXT}; run DECODE_HEAD64_BARRIER_TIMING=1 ./compile_decode_head64.sh")
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--s-q", type=int, default=1)
parser.add_argument("--topk", type=int, default=512)
parser.add_argument("--page-size", type=int, default=64)
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--iters", type=int, default=5)
args = parser.parse_args()
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    raise RuntimeError("requires an SM100-family GPU")
if args.topk < 64 or args.topk % 64:
    raise ValueError("topk must be a positive multiple of 64")

device = torch.device("cuda")
q = torch.randn((args.batch, args.s_q, 64, 512), device=device, dtype=torch.bfloat16)
num_blocks = max(args.topk, 64) // args.page_size
row_bytes = args.page_size * 584
block_bytes = ((row_bytes + 575) // 576) * 576
kv_storage = torch.randint(0, 256, (num_blocks * block_bytes,), device=device, dtype=torch.uint8)
kv = torch.as_strided(kv_storage, (num_blocks, args.page_size, 1, 584),
                      (block_bytes, 584, 584, 1))
indices = torch.randint(kv.shape[0] * args.page_size, (args.batch, args.s_q, args.topk), device=device, dtype=torch.int32)

def run():
    return ext.head64_decode(q, kv, indices, 512 ** -0.5)

run(); torch.cuda.synchronize()
for _ in range(args.warmup): run()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(args.iters): run()
end.record(); end.synchronize()
print(f"head64 Model1 decode: batch={args.batch} s_q={args.s_q} topk={args.topk} mean_us={(start.elapsed_time(end)*1000/args.iters):.3f}")
