# Sparse Head-Small Transposed Backward Plan

This plan targets a first SM100 backward kernel for
`csrc/sm100/prefill/sparse/fwd/head_small/phase1.cuh` style sparse prefill.
The forward kernel uses a transposed attention dataflow:

```text
P_t = K @ Q^T          [B_TOPK, B_H]
S_t = softmax(P_t)     logically [B_TOPK, B_H]
O_t = V^T @ S_t        [D_V, B_H]
```

Backward should keep the same principle: avoid register-fragment transpose and
cross-warp shuffle paths by using TMEM as the accumulator/exchange layer and
SMEM as the explicit transpose/staging surface.

KernelWiki references used:

- `sources/prs/cutlass/PR-2466.md` (`pr-cutlass-2466`): Blackwell MLA backward
  uses transposed `S^T = QK` and `dP^T = dOV`, stores score/probability data in
  SMEM, and dedicates TMEM regions to `dK`, `dV`, and reused `dQ/dP` workspaces.
- `wiki/techniques/warp-specialization.md` (`technique-warp-specialization`):
  on SM100, one warp can issue `tcgen05.mma` while other warps specialize on
  load/compute/epilogue because accumulators live in TMEM.
- `wiki/kernels/flash-attention-4.md` (`kernel-flash-attention-4`): overlap
  softmax/rescale work with MMA work and avoid unnecessary correction work.

## Fixed Decisions

- Inputs/outputs: `dO`, final `dQ`, final `dK`, and final `dV` are BF16.
- Internal accumulation: FP32 for `dQ` TMEM, `dK/dV` global accumulation buffers,
  softmax derivative scalars, and `d_attn_sink`.
- Sparse indices are per query token and shared by all heads:
  `indices[s_q_idx, kk]`, not per-head indices.
- Sparse indices have no duplicates inside one query. A CTA therefore does not
  need local duplicate-row combining for its own topk list.
- Cross-query accumulation is still required for `dK/dV` because different query
  CTAs can reference the same KV row.
- `attn_sink` is `nn.Parameter(torch.empty(n_q_heads, dtype=torch.float32))`.
  Its gradient is a global FP32 vector `[n_q_heads]` reduced across all query
  rows.
- Backward always recomputes `P_t = K @ Q^T`. Forward does not need to save P or
  probabilities.
- Backward uses forward `lse` only. `max_logits` is not a backward input.
- Grid is one CTA per query row: `grid = [s_q, 1, 1]`.

Non-goals for the first kernel:

- No direct BF16 global stores to final `dK/dV` from the main backward kernel.
- No FP8/MXFP8 path until the BF16 backward is correct and profiled.
- No 2-SM cooperative mode initially.

## Math

Forward recomputation uses natural-log logits conceptually:

```text
raw[kk,h] = dot(K[kk,:], Q[h,:])
p[kk,h]   = raw[kk,h] * sm_scale
prob[kk,h] = exp(p[kk,h] - lse[h])
sink_prob[h] = exp(attn_sink[h] - lse[h])
O[h,dv] = sum_kk prob[kk,h] * V[kk,dv] + sink_prob[h] * 0
```

Forward code may compute in base-2 for speed:

```text
p_log2 = raw * sm_scale * log2(e)
prob = exp2(p_log2 - lse * log2(e))
sink_prob = exp2(attn_sink * log2(e) - lse * log2(e))
```

Useful per-head scalar:

```text
D[h] = sum_dv dO[h,dv] * O[h,dv]
```

For each valid sparse KV row `kk` and head `h`:

```text
dProb[kk,h] = dot(V[kk,:], dO[h,:])
dP_scaled[kk,h] = prob[kk,h] * (dProb[kk,h] - D[h])
dRaw[kk,h] = dP_scaled[kk,h] * sm_scale

dV[kk,dv] += prob[kk,h] * dO[h,dv]
dQ[h,dq]  += dRaw[kk,h] * K[kk,dq]
dK[kk,dq] += dRaw[kk,h] * Q[h,dq]
```

Sink term:

```text
d_attn_sink[h] += sink_prob[h] * (0 - D[h])
```

There is no direct sink contribution to `dV`, `dQ`, or `dK`. The sink only
changes the denominator through `lse`; that effect is already included in
`prob` and `D` for real KV logits. Do not multiply `d_attn_sink` by `sm_scale`:
`attn_sink` is already a natural-logit parameter in the forward path.

## Transposed Backward Dataflow

Use these logical matrices inside the CTA:

```text
P_t      = K @ Q^T                        [B_TOPK, B_H]
Prob_t   = softmax(P_t, lse)              [B_TOPK, B_H]
dP_t     = V @ dO^T                       [B_TOPK, B_H]
dRaw_t   = Prob_t * (dP_t - D) * sm_scale [B_TOPK, B_H]
dV       = Prob_t @ dO                    [B_TOPK, D_V]
dK       = dRaw_t @ Q                     [B_TOPK, D_QK]
dQ_t     = K^T @ dRaw_t                   [D_QK, B_H]
```

The accumulator layout for `dV` is `[B_TOPK, D_V]`, matching the sparse KV row
storage order and making scatter/atomic stores straightforward.

## Tile Sizes and Launch

```text
B_TOPK      = 64
B_H         = H_Q                    // 8, 16, 24, or 32
B_H_TMEM    = H_Q == 24 ? 32 : H_Q
D_V         = 128                    // two 64-wide tiles
D_QK        = 128 or 192             // NoPE 128 + optional RoPE 64
NUM_BUFS    = 2                      // KV, valid, and indices buffers
NUM_THREADS = 384                    // three 128-thread warpgroups
```

Launch shape:

```cpp
__launch_bounds__(384, 1, 1)
kernel<<<params.s_q, 384, smem_size, params.stream>>>(params, tma_params);
```

No cluster is used in v1; this matches the existing head-small forward kernel's
1-SM CTA model.

## TMEM Allocation

Allocate 512 TMEM columns and use explicit column starts. Column widths are
based on MMA N-dimension columns, not logical row counts.

Worst-case `D_QK=192`, `B_H=32` layout:

| Region | Start | Active width | Reserved end | Contents |
| --- | ---: | ---: | ---: | --- |
| `DV` | 0 | 128 | 127 | `dV [B_TOPK, D_V]`, two 64-wide tiles |
| `DK` | 128 | 128 | 255 | `dK_nope [B_TOPK, 128]`, two 64-wide tiles |
| `Score` | 256 | `B_H_TMEM` <= 32 | 319 | reused for `P_t` / `dP_t [B_TOPK, B_H]` |
| `DQ` | 320 | `2 * B_H_TMEM` <= 64 | 383 | `dQ_nope_t [128, B_H]`, two 64-wide tiles |
| `DK_RoPE` | 384 | 64 | 447 | `dK_rope [B_TOPK, 64]` when `D_QK=192` |
| `DQ_RoPE` | 448 | `B_H_TMEM` <= 32 | 479 | `dQ_rope_t [64, B_H]` when `D_QK=192` |
| `Spare` | 480 | 32 | 511 | future score double buffer or scratch |

Recommended constants:

```cpp
struct tmem_cols {
    static constexpr int DV = 0;
    static constexpr int DV_WIDTH = 128;
    static constexpr int DK = 128;
    static constexpr int DK_WIDTH = 128;
    static constexpr int SCORE = 256;
    static constexpr int SCORE_WIDTH = B_H_TMEM;
    static constexpr int DQ = 320;
    static constexpr int DQ_WIDTH = 2 * B_H_TMEM;
    static constexpr int DK_ROPE = 384;
    static constexpr int DK_ROPE_WIDTH = D_QK == 192 ? 64 : 0;
    static constexpr int DQ_ROPE = 448;
    static constexpr int DQ_ROPE_WIDTH = D_QK == 192 ? B_H_TMEM : 0;
    static constexpr int TOTAL_RESERVED = 480;
};
static_assert(tmem_cols::TOTAL_RESERVED <= 512);
```

The `Score` region reserves columns 256..319 for simple fixed addressing even
though the active width is only `B_H_TMEM`. This leaves a clean boundary for
`DQ` at 320.

## Shared Memory Plan

Use a `SharedMemoryPlan` close to forward, with explicit backward staging. Store
`dO` in transposed form `[D_V, B_H]` in SMEM so both `dP_t = V @ dO^T` and
`dV = Prob_t @ dO` can use compatible UMMA views without extra cross-warp
transpose.

Persistent/prologue data:

```text
q_nope        SmemLayoutQNoPE        [B_H, 128] bf16
q_rope        SmemLayoutQRoPE        [B_H, 64] bf16, D_QK=192 only
do_t          SmemLayoutDOTransposed [D_V, B_H] bf16
o             linear/UMMA layout     [B_H, D_V] bf16, only needed for sum_odo
lse           float[B_H]
neg_lse       float[B_H]
sum_odo       float[B_H]
attn_sink     float[B_H]
```

Double-buffered sparse data:

```text
k_nope[2]     SmemLayoutKNoPE        [B_TOPK, 128] bf16
v[2]          SmemLayoutV            [B_TOPK, D_V] bf16
indices[2]    int[B_TOPK]
valid[2]      char[B_TOPK / 8]
k_rope        SmemLayoutKRoPE        [B_TOPK, 64] bf16, D_QK=192 only
```

Score/derivative data:

```text
p_t           float[B_TOPK * B_H]
prob_t        bf16[B_TOPK * B_H]
dp_t          float[B_TOPK * B_H]
draw_t        bf16[B_TOPK * B_H]      // dRaw_t, MMA operand for dK/dQ
```

Output staging:

```text
dq_smem       bf16[B_H, D_QK]         // optional; direct global store is also OK
dk_vec        float vector scratch    // TMEM -> FP32 atomicAdd to dk_acc
dv_vec        float vector scratch    // TMEM -> FP32 atomicAdd to dv_acc
```

Worst-case SMEM byte budget before barrier objects, with aliasing:

| Buffer | Bytes | Notes |
| --- | ---: | --- |
| `q_nope [B_H,128]` | 8192 | persistent |
| `q_rope [B_H,64]` | 4096 | D_QK=192 only |
| `do_t [128,B_H]` | 8192 | persistent transposed dO |
| `o [B_H,128]` | 8192 | can alias with later staging after `sum_odo` |
| `k_nope[2] [64,128]` | 32768 | double-buffered |
| `v[2] [64,128]` | 32768 | double-buffered unless aliased with K load phases |
| `k_rope [64,64]` | 8192 | single-buffered v1 |
| `p_t [64,B_H] float` | 8192 | can alias with `dp_t` after use |
| `dp_t [64,B_H] float` | 8192 | sequential with `p_t` if copied carefully |
| `prob_t [64,B_H] bf16` | 4096 | needed for dV MMA |
| `draw_t [64,B_H] bf16` | 4096 | needed for dK/dQ MMA |
| `dq_smem [B_H,192] bf16` | 12288 | optional staging; direct store can skip it |
| `indices[2]` | 512 | double-buffered for epilogue address safety |
| `valid[2]` | 16 | double-buffered |
| scalar arrays | <1024 | `lse`, `neg_lse`, `sum_odo`, barriers metadata |

With unions (`o` with `dq_smem`, `p_t` with `dp_t` where legal, and output
scratch with score buffers), the target is about 110-125 KB, below SM100's
228 KB SMEM capacity. Add a compile-time `static_assert(sizeof(SharedMemoryPlan)
<= cutlass::arch::sm100_smem_capacity_bytes)` once layouts are concrete.

## SumOdO Strategy

WG0 computes `sum_odo[h] = dot(O[h,:], dO[h,:])` in the prologue.

Use all 128 threads in WG0:

```text
head = idx_in_warpgroup % B_H
part = idx_in_warpgroup / B_H
parts_per_head = 128 / B_H
```

Each thread handles a contiguous or strided slice of `D_V / parts_per_head`
elements:

```text
B_H=32 -> 4 threads/head, 32 dv values/thread
B_H=16 -> 8 threads/head, 16 dv values/thread
B_H=8  -> 16 threads/head, 8 dv values/thread
```

Write partial sums to `sum_odo_partial[128]`, synchronize WG0, then one thread
per head reduces `parts_per_head` partials into `sum_odo[h]`. This avoids
complicated sub-warp masks and gives a clear barrier before softmax derivative
computation.

## Warpgroup and Warp Assignment

Keep three warpgroups to match the forward kernel:

```text
WG0: compute/softmax/epilogue, warps 0..3
WG1: sparse KV + prologue loader, warps 4..7
WG2: tcgen05 MMA issuer, warps 8..11
```

WG0:

- Initialize barriers and TMEM from warp 0.
- Load `lse`, precompute `neg_lse`, load `attn_sink`, and compute `sum_odo`.
- Copy `P_t` / `dP_t` from TMEM Score into SMEM.
- Compute `prob_t` and `draw_t = prob_t * (dP_t - sum_odo) * sm_scale`.
- Compute `d_attn_sink[h] = -sink_prob[h] * sum_odo[h]` and FP32 `atomicAdd`
  into global `d_attn_sink[h]`.
- Read TMEM `dQ_t`, `dK`, and `dV` in vectorized chunks.
- Direct-store BF16 `dQ`; FP32 `atomicAdd` `dK/dV` into `dk_acc/dv_acc`.

WG1:

- Warp 4 loads sparse indices/valid masks and issues TMA gather for K/V NoPE.
- Warp 5 loads Q, O, and dO; write dO as transposed `do_t [D_V, B_H]`.
- Warps 6..7 load RoPE K rows with cp.async when `D_QK=192`.
- Keep `indices[2]` double-buffered because the optimized epilogue may still
  need indices for block `k-2` while WG1 prepares indices for block `k`.

WG2:

- Warp 8 issues all `tcgen05.mma` instructions using `elect_one_sync()`.
- Warp 9 can generate valid masks or prepare scatter addresses if WG1 becomes
  overloaded.
- Warps 10..11 can be idle in v1 or used for RoPE copy assistance.

Register directives:

```cpp
// Same starting point as forward.
warpgroup_reg_alloc<176>();   // WG0 compute/softmax/epilogue
warpgroup_reg_dealloc<80>();  // WG1 loader
warpgroup_reg_dealloc<80>();  // WG2 MMA issuer
```

Verify with `--ptxas-options=-v`. If WG0 spills, split WG0 into warps 0..1 for
score/softmax derivative and warps 2..3 for TMEM epilogue.

## Barrier Plan

For v1 correctness, keep score/MMAs mostly serial and use single logical
barriers:

```text
bar_kv_ready[2]       WG1 -> WG2, K/V NoPE tile ready
bar_k_rope_ready      WG1 -> WG2, K RoPE tile ready
bar_valid_ready[2]    WG1 -> WG0, valid mask and indices ready
bar_score_free        WG0 -> WG2, Score TMEM can be overwritten
bar_qk_done           WG2 -> WG0, P_t ready in Score TMEM
bar_dp_done           WG2 -> WG0, dP_t ready in Score TMEM
bar_ds_ready          WG0 -> WG2, prob_t/draw_t SMEM operands ready
bar_dv_done           WG2 -> WG0, dV TMEM accumulator ready
bar_dk_done           WG2 -> WG0, dK TMEM accumulator ready
bar_dq_done           WG2 -> WG0, dQ TMEM accumulator ready
bar_store_done[2]     WG0 -> WG1/WG2, indices/output staging reusable
wg0_sync              intra-WG0 named barrier
```

Introduce `[2]` variants for `bar_qk_done`, `bar_dp_done`, and score buffers only
in the optimized double-buffered score pipeline. Always use
`tcgen05_after_thread_sync()` before reading TMEM and `tcgen05_before_thread_sync()`
before releasing a TMEM region to the MMA warp.

## Pipeline

### Prologue

1. Initialize barriers and allocate 512 TMEM columns.
2. Load Q NoPE/RoPE to SMEM.
3. Load O and dO; store dO as `do_t [D_V, B_H]`.
4. Load `lse`, compute `neg_lse`, and load `attn_sink`.
5. WG0 computes `sum_odo[h]` and `d_attn_sink[h]`, then FP32 atomic-adds sink
   gradients to the global `[n_q_heads]` vector.
6. WG1 starts loading KV block 0, `indices[0]`, and `valid[0]`.

### V1 Serial Main Loop

For each sparse block `k`:

1. WG2 waits for `bar_kv_ready[cur]`, `bar_valid_ready[cur]`, and
   `bar_score_free`.
2. WG2 computes `P_t = K @ Q^T` into `tmem Score`.
   - If `D_QK=192`, compute RoPE partial first and NoPE partial with accumulate,
     matching forward's score semantics.
3. WG0 copies `P_t` TMEM -> `p_t` SMEM.
4. WG2 computes `dP_t = V @ dO^T` into `tmem Score`.
5. WG0 copies `dP_t` TMEM -> `dp_t` SMEM.
6. WG0 computes `prob_t` and `draw_t` for valid rows; invalid rows write zero.
7. WG2 computes gradient MMAs:
   - `dV [B_TOPK, D_V] = Prob_t [B_TOPK, B_H] @ dO [B_H, D_V]`.
   - `dK_nope [B_TOPK,128] = draw_t [B_TOPK,B_H] @ Q_nope [B_H,128]`.
   - `dK_rope [B_TOPK,64] = draw_t @ Q_rope` when `D_QK=192`.
   - `dQ_nope_t [128,B_H] += K_nope^T [128,B_TOPK] @ draw_t`.
   - `dQ_rope_t [64,B_H] += K_rope^T @ draw_t` when `D_QK=192`.
8. WG0 epilogue for this block:
   - Read `dV` and `dK` from TMEM and FP32 `atomicAdd` to `dv_acc/dk_acc` using
     `indices[cur]`.
   - Skip invalid rows.
9. Release KV/indices buffer `cur` and Score TMEM.

`dQ_t` accumulates across all sparse blocks in TMEM and is stored only after the
loop finishes.

### Optimized Steady State

Target overlap:

```text
WG1: load K/V(k+1), indices(k+1), valid(k+1)
WG2: compute P_t(k) and gradient MMAs for k
WG0: softmax/draw for k-1 and dK/dV epilogue for k-2
```

`NUM_BUFS=2` is sufficient for K/V SMEM because the dK/dV epilogue reads only
TMEM accumulators plus `indices[cur]`; it does not re-read K/V SMEM. Indices
must stay double-buffered so epilogue address generation cannot race with WG1's
next index load.

After v1 correctness:

- Add a second score TMEM region if the 512-column budget allows it cleanly.
- Double-buffer `bar_qk_done`, `bar_dp_done`, and Score ownership.
- Move `dV` earlier because it needs only `prob_t` and dO, while `dK/dQ` need
  `draw_t`.
- For `D_QK=192`, compute gradients in this order for overlap:
  1. `dK_nope` two 64-wide tiles.
  2. `dK_rope` one tile.
  3. `dQ_nope_t` two tiles.
  4. `dQ_rope_t` one tile.

## MMA Major Selection

The first implementation should use the following atom orientations. The exact
CuTe layouts should be written to match this table before coding the kernel.

| MMA | Result shape | A operand | A major | B operand | B major |
| --- | --- | --- | --- | --- | --- |
| `P_t = K @ Q^T` | `[B_TOPK, B_H]` | `K [B_TOPK,D]` | `K` | `Q [B_H,D]` | `K` |
| `dP_t = V @ dO^T` | `[B_TOPK,B_H]` | `V [B_TOPK,D_V]` | `K` | `do_t [D_V,B_H]` | `K` |
| `dV = Prob_t @ dO` | `[B_TOPK,64]` | `Prob_t [B_TOPK,B_H]` | `K` | `do_t tile [64,B_H]` | `MN` |
| `dK = draw_t @ Q` | `[B_TOPK,64]` | `draw_t [B_TOPK,B_H]` | `K` | `Q tile [B_H,64]` | `MN` |
| `dQ_t = K^T @ draw_t` | `[64,B_H]` | `K tile [64,B_TOPK]` | `MN` | `draw_t [B_TOPK,B_H]` | `MN` |

Sketch:

```cpp
using TiledMMA_QK = SM100_MMA_F16BF16_SS_NOELECT<
    bf16, bf16, float, B_TOPK, B_H, UMMA::Major::K, UMMA::Major::K>;

using TiledMMA_DPV = SM100_MMA_F16BF16_SS_NOELECT<
    bf16, bf16, float, B_TOPK, B_H, UMMA::Major::K, UMMA::Major::K>;

using TiledMMA_DV = SM100_MMA_F16BF16_SS_NOELECT<
    bf16, bf16, float, B_TOPK, 64, UMMA::Major::K, UMMA::Major::MN>;

using TiledMMA_DK = SM100_MMA_F16BF16_SS_NOELECT<
    bf16, bf16, float, B_TOPK, 64, UMMA::Major::K, UMMA::Major::MN>;

using TiledMMA_DQ = SM100_MMA_F16BF16_SS_NOELECT<
    bf16, bf16, float, 64, B_H, UMMA::Major::MN, UMMA::Major::MN>;
```

## dQ TMEM-to-Global Transpose

`dQ_t` is accumulated in TMEM as `[D_QK_tile, B_H]`, but global `dQ` is
`[B_H, D_QK]`. Use the same pattern as forward's O epilogue:

```cpp
for (int tile = 0; tile < D_QK / 64; ++tile) {
    int dq_pos = tile * 64 + warp_idx_in_wg0 * 32 + lane_idx;
    if (dq_pos < D_QK) {
        float dq_vals[B_H_TMEM];
        ku::tmem_ld_32dp32bNx<B_H_TMEM>(dq_tmem_col(tile), dq_vals);
        cutlass::arch::fence_view_async_tmem_load();
        for (int h = 0; h < B_H; ++h) {
            params.dq[s_q_idx, h, dq_pos] = bf16(dq_vals[h]);
        }
    }
}
```

For `D_QK=192`, this is three 64-wide passes. Warps 0..3 cover 128 positions per
pass capacity, so pass 0/1 use all four warps and pass 2 uses the first two
warps for positions 128..191.

## Global Accumulation and Convert Kernels

`dQ`:

- One CTA owns one query row's full topk list, so final BF16 `dQ` is a direct
  store after all sparse blocks have accumulated into TMEM.
- If future work splits topk across CTAs, add FP32 `dQ_acc` and a convert kernel.

`dK/dV`:

- Main backward kernel uses FP32 `atomicAdd` into:

```text
dk_acc[s_kv, D_QK] float
dv_acc[s_kv, D_V]  float
```

- Final `dK/dV` outputs are produced by a convert kernel:

```cpp
__global__ void convert_fp32_to_bf16(
    const float* __restrict__ src,
    bf16* __restrict__ dst,
    int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}
```

Launch one convert for `dk_acc` over `s_kv * D_QK` and one for `dv_acc` over
`s_kv * D_V`. Vectorize later if bandwidth shows up in profiling.

`d_attn_sink`:

- Main backward kernel uses FP32 `atomicAdd(&d_attn_sink[h], d_sink_h)`.
- Contention is `s_q` atomics per head. With `n_q_heads <= 32`, this is
  acceptable for v1. If profiling shows pressure, switch to per-CTA partials and
  a final head-wise reduction.

## Implementation Milestones

1. Add backward params.
   - Inputs: `q`, `kv`, `out`, `do`, `lse`, `indices`, `topk_length`,
     `attn_sink`.
   - Outputs/workspaces: BF16 `dq`, BF16 `dk`, BF16 `dv`, FP32 `dk_acc`, FP32
     `dv_acc`, FP32 `d_attn_sink`.
   - Remove `max_logits` from the backward interface.
2. Add `csrc/sm100/prefill/sparse/bwd/head_small/config.h`.
   - Define SMEM layouts for `do_t`, `prob_t`, `draw_t`, Q/K/V, and dQ epilogue.
   - Define TMEM columns and width static asserts.
   - Add SMEM capacity static assert.
3. Add a slow but structurally correct kernel.
   - Serial score/dP/draw/dV/dK/dQ sequence.
   - FP32 `atomicAdd` into `dk_acc/dv_acc`.
   - FP32 `atomicAdd` into `d_attn_sink`.
   - Direct BF16 `dQ` store.
4. Add convert kernels for `dk_acc -> dk` and `dv_acc -> dv`.
5. Add reference tests.
   - Compare against PyTorch dense gather reference.
   - Cover `D_QK=128`, `D_QK=192`, all `B_H` variants, invalid topk tails,
     cross-query reuse of the same KV row, and sink gradients.
6. Profile with NCU.
   - Check tensor pipe utilization, TMEM load/store stalls, shared bank conflicts,
     FP32 atomic pressure, and register spills.
7. Optimize pipeline.
   - Double-buffer Score and related barriers.
   - Overlap KV load, score MMA, softmax derivative, and dK/dV epilogue.
   - If FP32 atomics dominate, consider replacing `dk_acc/dv_acc` atomics with a
     contribution workspace plus segmented reduce.

## Correctness Checklist

- Invalid topk rows contribute zero to `prob_t`, `draw_t`, `dV`, `dK`, and `dQ`.
- `prob = exp(p - lse)` uses only forward `lse`; `max_logits` is not needed.
- `attn_sink` contributes only `d_attn_sink = -sink_prob * D`, not `dV/dQ/dK`.
- `dRaw_t` includes `sm_scale`; otherwise `dQ/dK` will be off by that factor.
- `dP_t = V @ dO^T` is `dProb`, not the final score gradient.
- `dV` accumulator layout is `[B_TOPK, D_V]` and scatters by `indices[cur][kk]`.
- Single-query indices have no duplicates; cross-query accumulation still uses
  FP32 atomics.
- `D_QK=192` stores NoPE dims `0..127` and RoPE dims `128..191` separately for
  both `dQ` and `dK`.
- `NUM_BUFS=2` is safe for optimized K/V buffering because dK/dV epilogue reads
  TMEM plus double-buffered indices, not K/V SMEM.

## Resolved Improve.md Items

- Fixed dV notation to use `[B_TOPK, D_V]` consistently.
- Added explicit TMEM widths and reservation boundaries.
- Removed `max_logits` from backward inputs; use only `lse`.
- Added concrete `sum_odo` WG0 reduction mapping.
- Added worst-case SMEM budget and aliasing notes.
- Added register allocation starting point.
- Added launch configuration.
- Added dQ transpose strategy.
- Added FP32 `d_attn_sink` atomic strategy.
- Added FP32-to-BF16 convert kernel deliverable.
- Clarified sink has no direct dV/dQ/dK contribution.
- Clarified why `NUM_BUFS=2` is compatible with the optimized pipeline.
- Resolved first-pass MMA major selections.
