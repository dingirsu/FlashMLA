# MXFP8 DSA Prefill Kernel — Bug-Fix Plan (D=512)

> Target file: `csrc/sm100/prefill/sparse/mxfp8_fwd/head64/phase1.cuh`
> Companion file: `csrc/sm100/prefill/sparse/mxfp8_fwd/head64/config.h`
> API: `csrc/sm100/prefill/sparse/mxfp8_fwd/head64/phase1.h`
> Instantiation: `csrc/sm100/prefill/sparse/mxfp8_fwd/head64/instantiations/phase1_k512.cu`
>
> Goal: get the `D=512` MXFP8 (e4m3 NoPE + UE8M0 scales, no RoPE) sparse
> prefill kernel to compile, link, launch, and produce a numerically correct
> `O = softmax(QK^T) @ V` for `B_H=64, B_TOPK=128`. Only correctness is fixed;
> vectorisation / pipeline tuning is deferred.
>
> References:
> - KernelWiki `wiki/hardware/tmem.md` — TMEM = 128 rows × 512 cols × 4 B = 256 KB
> - KernelWiki `wiki/hardware/tcgen05-mma.md` — MMA atom semantics
> - KernelWiki `wiki/techniques/fine-grained-quantization.md` — MXFP8 / UE8M0
> - Non-mxfp8 reference: `csrc/sm100/prefill/sparse/fwd/head64/`
> - SM100 UTCCP / MMF8F6F4 cutlass traits:
>   `csrc/cutlass/include/cute/atom/mma_traits_sm100.hpp:2700-2784`

## Hardware Constraints (from KernelWiki)

| Item | Value | Notes |
|---|---|---|
| TMEM total | 128 rows × 512 cols × 4 B = 256 KB | Per SM (`hw-tmem.md`) |
| TMEM row mapping | Row `r` ↔ thread (warp `r/32`, lane `r%32`) | Each thread owns one TMEM row |
| 128×256 FP32 MMA tile | 256 TMEM cols | `hw-tmem.md` "128x256 MMA accumulator tile occupies 128 rows × 256 cols" |
| MMA atom `M × N` | 128 rows × N cols of TMEM | CLayout stride `[0, _1, Int<M>=128]` ⇒ addr = m + 128·n |
| `tcgen05.ld.32dp32b.Nx` | N broadcasts × 32 lanes | `data[N]` = N unique fp32 per thread, each loaded from a distinct TMEM col |
| UTCCP variants | `SM100_UTCCP_4x32dp128bit_1cta` | 32-lane UTCCP copy smem → tmem |

For our `TiledMMA_O = SM100_MMA_MXF8F6F4_SS_NOELECT<e4m3, e4m3, float,
e8m0, B_TOPK=128, B_H=64, UMMA::Major::MN, UMMA::Major::K>{}`:

- Atom shape: `[M=128, N=64, K=32]`
- CLayout stride: `[0, _1, Int<128>]` ⇒ `addr = m + 128·n`
- One atom occupies 64 TMEM cols × 128 rows = 8192 fp32 cells
- Each TMEM col holds 128 row-elements (the atom's M-dim)
- 4 atoms at base cols `O, O+64, O+128, O+192` ⇒ 256 TMEM cols total

For `TiledMMA_P = SM100_MMA_MXF8F6F4_SS_NOELECT<e4m3, e4m3, float,
e8m0, B_TOPK=128, B_H=64, UMMA::Major::K, UMMA::Major::K>{}`:

- Atom shape: `[128, 64, 32]`, same TMEM footprint as `O`
- `partition_fragment_C(tiled_mma_P, Shape<B_TOPK, B_H>)` returns a TMEM
  fragment of `[128 × 64]` = 8192 fp32 = 64 cols × 128 rows

## TMEM Column Budget

Per KernelWiki, the 4 warps of WG0 + 4 warps of WG1 + 4 warps of WG2 = 12
warps total.  TMEM is **per SM**, shared across all CTAs on the SM.  Each
CTA independently allocates from the 512-col pool.

Current allocation (`config.h:55-63`):

```cpp
namespace tmem_cols {
    constexpr int O        = 0;     // 256 cols (4 atoms × 64 cols)
    constexpr int Q_Scale  = 320;   // 18 cols
    constexpr int K_Scale  = 338;   // 18 cols
    constexpr int S_Scale  = 356;   // 44 cols
    constexpr int P        = 400;   // 64 cols (atom)
}
```

Budget used: 320 + 64 = 384 cols (O + P disjoint lifetimes ⇒ can share
0..319 in time, but they must be at distinct start cols).  With current
layout O at 0..319 and P at 400..463, **384 cols are reserved** out of
512.  Fits, but tight.  Add a `static_assert`:

```cpp
static_assert(tmem_cols::S_Scale + 32 <= 512, "TMEM budget overflow");
```

## Status of Earlier Fixes

| Item | Status |
| --- | --- |
| `run_mxfp8_fwd_phase1_kernel` rename + launch | done (phase1.cuh:573-574) |
| `TmaParams` struct | done (config.h:17-26, only 4 fields: shape_O/tma_O, shape_Q_scale/tma_Q_scale, tensor_map_q, tensor_map_kv) |
| `SmemLayoutOBuf_TMA` | done (config.h:94) |
| `plan.s_q_rope.q_tail.q_scale(...)` → `plan.s_q_scale.{q_scale,s}(...)` | done |
| Barrier init renames (no more `_rope`/`_nope` suffix) | done (config.h:162-166) |
| `B_H_TMEM`, `FP8_MAX`, `Q_SCALE_BYTES`, `K_SCALE_BYTES`, `K_QUANT_GROUP_SIZE` defined | done (config.h:36-46) |
| `SmemLayoutO = 8` tiles | done (config.h:93) |
| `SmemLayoutK_TiledMMA = SmemLayoutK` | done (config.h:112) |
| `tQ_scale` / `tK_scale` / `tV_scale` / `tS_scale` declarations hoisted | done (phase1.cuh:72-75) |
| `sQ_rope` / `sQ_nope` `...` placeholders removed | done |
| O epilogue rewritten: `dv = tile*B_TOPK + idx_in_warpgroup` | done (phase1.cuh:247-260) — correct parallelisation |
| `bar_prologue_utccp_q_scale.wait(0)` (no `arrive_and_expect_tx`) | done (phase1.cuh:372) |
| `bar_kv_scale_ready.wait(...)` added in MMA warp | done (phase1.cuh:405) |
| `bar_p_free.arrive()` in scale/exp warp | done (phase1.cuh:147) |
| `bar_so_ready.init(1)` + `elect_one_sync()` arrive | done (phase1.cuh:113, 216-218) |
| `kk` re-declared in inner scale/exp loop | done (phase1.cuh:165, 194) |
| Valid-mask unsigned compare | done (phase1.cuh:166) |
| `s_scale` smem staging buffer added | done (config.h:158) |
| `sS_scale(h, g*MXFP8_SCALE_VEC_SIZE, _0{}) = scale_g;` writes the S scales | done (phase1.cuh:192) |
| S_Scale UTCCP in MMA warp before SV | done (phase1.cuh:446-454) |
| All RoPE removed (no `TiledMMA_P_RoPE`, no `bar_kv_rope_ready`, no `q_rope` in s_q_scale) | done |
| `D = 512` (no more `D_NOPE`, `D_NOPE_PAD`, `Q_ROPE_OFFSET`) | done |

## Remaining Bugs

### P0 — `k_row` race in P-load → softmax uses wrong P

`phase1.cuh:148`:

```cpp
int k_row = lane_idx + (warp_idx&3);
for (int h = 0; h < B_H; ++h) {
    plan.p_t[h + B_H * k_row] = p[h];
}
```

- `warp_idx` (cutlass canonical warp idx, range 0..11) has `&3 = 0..3` for
  warps in WG0.  So `k_row = lane_idx + warp_idx ∈ 0..34`.
- **Race**: thread (warp 0, lane 1) and thread (warp 1, lane 0) both write
  `k_row = 1` with **different values** (different TMEM rows and different
  TMEM col blocks).
- The full `B_TOPK=128` rows of P are not all loaded — only k_row values
  `0..34` are written, leaving `p_t[*, 35..127]` uninitialised.
- The softmax loop at lines 161-186 reads `plan.p_t[kk*B_H + h]` for
  `kk = 0..127`.  For `kk ≥ 35` it reads garbage.

**Fix:** each (warp, lane) must own a unique `k_row` in `[0, B_TOPK=128)`.
The natural formula for 4 warps × 32 lanes = 128 threads is:

```cpp
int k_row = warp_idx * 32 + lane_idx;     // 0..127, no overlap
```

Then `plan.p_t[h + B_H * k_row] = p[h]` writes the correct lane's P
strip to the correct smem position.

The corresponding TMEM col offset per warp must also be `warp_idx * 16`
to read the atom's 64-col layout in 4 chunks of 16 cols each:

```cpp
ku::tmem_ld_32dp32bNx<B_H_TMEM>(tmem_cols::P + warp_idx * 16, p);
```

(Verify: 4 warps × 16 cols = 64 cols total — matches the P atom's 64
cols.)

### P0 — `plan.p_t` should be `p_t[B_H][B_TOPK]` (row-major), not flat

`config.h:161`:

```cpp
float p_t[B_TOPK*B_H];      // 128*64 = 8192 floats
```

Access pattern: `p_t[h + B_H * k_row]` = `p_t[h + 64 * k_row]`.  This is
the flat layout where rows are k_row (topk index) and cols are h (head).

The softmax loop reads `plan.p_t[kk*B_H + h]` = `plan.p_t[kk*64 + h]`
where `kk` is the topk index and `h` is the head.  Same layout — flat
row-major with k_row as outer index.  ✓

The flat layout is fine; no change needed.

### P0 — `bar_prologue_utccp_k_scale` has no consumer

`phase1.cuh:419`:

```cpp
ku::umma_arrive_noelect(plan.bar_prologue_utccp_k_scale);
```

No `wait(...)` on this barrier anywhere in the kernel.  The K-scale
UTCCP completes synchronously inside `cute::copy(...)` at line 418, so
the data is in TMEM before the next instruction.

**Fix:** Drop `umma_arrive_noelect(plan.bar_prologue_utccp_k_scale)`
(line 419).  Optionally remove the field from `SharedMemoryPlan`.

### P1 — `bar_kv_ready.arrive_and_expect_tx` byte estimate

`phase1.cuh:421`:

```cpp
plan.bar_kv_ready[cur_buf].arrive_and_expect_tx(B_TOPK*D_K*sizeof(e4m3));
```

With `B_TOPK=128, D_K=512`: 65536 bytes.  Each `ku::tma_gather4` call
(64 calls × 4 producer warps = 256 calls per buffer) reads
`4 rows × 64 int64 = 2048 bytes` per call.  Total: 256 × 2048 = 524288
bytes (8× the estimate).

The TMA hardware may accept the under-estimate (the barrier just
signals that data has arrived; the byte count is a hint for prefetch).
This may not actually cause a correctness bug — but it could lead to
hardware stalls because the expected-byte counter is too low.

**Fix:** Either:

(a) Reduce the per-call gather4 box to fit the actual data (use a
smaller box and call gather4 with fewer rows), or
(b) Increase `arrive_and_expect_tx` to the actual byte count
(`B_TOPK * D_K * sizeof(e4m3) * (4 warps)` or a per-warp estimate).

For v1 correctness, the under-estimate is harmless — verify by NCU
profile later.

### P1 — TMEM load broadcast waste in epilogue

`phase1.cuh:247-260`: all 128 WG0 threads issue the same `tmem_ld` with
the same TMEM col start.  Each lane broadcasts 64 fp32 (one per col) to
all 32 lanes in its warp.  4 warps × 32 lanes each get their own 64
fp32 from rows `(w*32, w*32+1, …, w*32+31)` of cols `[tile*64,
tile*64+1, …, tile*64+63]`.

The mapping to `sO(h, dv)` is correct (verified in design):

```
thread (w, l) writes
  sO(h, tile*128 + w*32 + l) = o_head[h]
where o_head[h] = TMEM[w*32 + l, tile*64 + h]
```

This maps exactly to `O[h, dv] = TMEM[dv % 128, atom_idx * 64 + h]`
where `atom_idx = dv / 128 = tile` and `dv % 128 = w*32 + l`.  ✓

**No change needed** for the epilogue itself.  But the design wastes
3/4 of the load bandwidth because only 64 rows of each 128-row TMEM
col are useful.  This is a perf-only concern; not a correctness issue.

### P1 — Dead top-level `tO` partition fragment

`phase1.cuh:71, 78`:

```cpp
Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<B_TOPK>, Int<B_H>>{});
...
tO.data().get() = tmem_cols::O;
```

`TiledMMA_O` is used in the SV MMA loop (line 461), but `tO` (the
top-level fragment) is never referenced — the loop creates fresh
`tO_block` fragments per iteration.  The top-level declaration +
assignment are dead code.

**Fix:** Either remove the top-level `tO` (clean up), or refactor the
SV MMA loop to issue MMAs against a single `tO` with proper slicing.

### P1 — `bar_k_valid_free` arrive count was a non-bug

`phase1.cuh:122`: `init(128)`, but only 128 WG0 threads arrive once per
block.  The wait side (`phase1.cuh:498`) is in warp 9 with only 16
lanes — the wait just decrements the count, and 128 arrives vs 16
waits is fine because they are different thread sets.

**Confirmed OK**; remove from bug list.

### P2 — `kv_scale_ready` arrives before `kv_ready`

The producer warp at line 354 arrives on `bar_kv_scale_ready` AFTER
loading the scale data into SMEM.  The MMA warp waits on
`bar_kv_scale_ready` (line 405) before doing the K-scale UTCCP.  ✓

This is correct, but note that the order is: TMA gather4 (KV data) →
SMEM scale write → `bar_kv_scale_ready.arrive()` → MMA waits → K UTCCP.

The KV data TMA itself arrives on `bar_kv_ready` (line 421's
`arrive_and_expect_tx`).  The MMA waits on `bar_kv_ready` at line 422
AFTER the K-scale UTCCP, which is fine because the UTCCP doesn't
need the KV data — it only needs the scale data which arrived earlier
via `bar_kv_scale_ready`.

### P2 — Perf TODOs (deferred)

These `//TODO: change here to vectorized…` comments at
`phase1.cuh:191, 195, 197` and `// TODO: change here to tcgen05.ld
instead of tcgen05.cp` at line 451 are perf-only.  Leave them.

## Verification

After applying the P0 fixes:

1. `nvcc -arch=sm_100` should compile.
2. The `static_assert` on TMEM budget should pass.
3. Add a small test driver:
   - Allocate Q/K/V in MXFP8 layout with known `topk_length`,
     `attn_sink`, and a deterministic set of `indices`.
   - Compute the reference attention output with PyTorch in FP32.
   - Compare the kernel output to the reference to within
     `atol=2e-2, rtol=2e-2`.

## Summary of Open Tasks

1. (P0) **Fix `k_row` race** at `phase1.cuh:148`:
   ```cpp
   int k_row = warp_idx * 32 + lane_idx;     // 0..127 unique
   ```
   And the TMEM col stride at line 144:
   ```cpp
   ku::tmem_ld_32dp32bNx<B_H_TMEM>(tmem_cols::P + warp_idx * 16, p);
   ```
2. (P0) **Drop dead `umma_arrive_noelect(plan.bar_prologue_utccp_k_scale)`**
   at `phase1.cuh:419`.
3. (P1) **Remove top-level `tO` partition fragment** at `phase1.cuh:71, 78`
   (or refactor SV MMA loop to use it).
4. (P1) **Add `static_assert`** on TMEM column budget in `config.h`.
5. (P2) Defer perf TODOs.