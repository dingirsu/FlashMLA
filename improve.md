# Sparse Head-Small Backward Plan — Improvement Notes

Based on review of `plan.md`, the forward kernel `csrc/sm100/prefill/sparse/fwd/head_small/phase1.cuh`, and KernelWiki references.

---

## Issues to Fix

### 1. dP_t / dV notation inconsistency

In the "transposed representation" section:
```
dV_t = V-gradient as dV^T = dO^T @ Prob [dv, topk] or dV = ProbT @ dO
```
The two forms are inconsistent. `dV^T = dO^T @ Prob` gives `[D_V, B_TOPK]`, while `dV = ProbT @ dO` gives `[B_TOPK, D_V]`. Pick one consistently. Since the plan aims to match forward's transposed dataflow, the accumulator layout `dV [B_TOPK, D_V]` (same storage order as V) is more natural for scatter stores to the sparse KV rows.

**Recommendation:** State clearly that `dV` is accumulated in `[B_TOPK, D_V]` layout (matching V's storage in SMEM) via the MMA `dV += Prob_t^T @ dO` where `Prob_t^T` is `[B_TOPK, B_H]` and `dO` is `[B_H, D_V]`.

---

### 2. TMEM column accounting lacks explicit widths

The plan states `tmem_cols::DV = 0`, `DK = 128`, `Score = 256`, etc., but does not annotate how many columns each region actually consumes. This makes it impossible to verify correctness.

Concrete analysis for worst-case (B_H=32, B_TOPK=64, D_QK=192):

| Region | Start | Width | End | Contents |
|--------|-------|-------|-----|----------|
| DV | 0 | 128 | 127 | dV accumulator [B_TOPK=64, D_V=128], 64 rows × 2 tiles of 64 cols |
| DK | 128 | 128 | 255 | dK NoPE accumulator [B_TOPK=64, 128] |
| Score | 256 | 64 | 319 | P_t / dP_t workspace [B_TOPK=64, B_H≤32] padded to B_H_TMEM |
| DQ | 320 | 64 | 383 | dQ_t accumulator [D_QK_tile=64, B_H≤32] |
| DK_RoPE | 384 | 64 | 447 | dK RoPE accumulator [B_TOPK=64, 64] (D_QK=192 only) |
| Spare | 448 | 64 | 511 | Available for double-buffered score or DQ RoPE |

Total used: 448 columns (fits 512).

**Recommendation:** Add explicit width annotations and a `static_assert`:
```cpp
static_assert(tmem_cols::DV_WIDTH + tmem_cols::DK_WIDTH + tmem_cols::SCORE_WIDTH 
              + tmem_cols::DQ_WIDTH + tmem_cols::DK_ROPE_WIDTH <= 512);
```

Note: The width depends on `B_H_TMEM` (which is `H_Q == 24 ? 32 : H_Q`). Verify that Score width = B_H_TMEM (not B_TOPK) since P_t is stored as [B_TOPK, B_H] but the TMEM atom's M-dimension is B_TOPK (the "row" dimension in TMEM is the per-lane direction, which is B_TOPK for the QK MMA). Re-check: for `TiledMMA_P` with shape `[B_TOPK, B_H]`, the accumulator in TMEM uses B_H_TMEM columns. So Score width = B_H_TMEM ≤ 32 columns, not 64. This means total could be as low as 128+128+32+32+64 = 384 for D_QK=192, B_H=32. Even better.

---

### 3. Resolve `lse` vs `max_logits` — do not leave open

From the forward kernel code:
```cpp
params.max_logits[global_index] = real_mi * CUDART_LN2_F;  // natural log scale
params.lse[global_index] = fmaf(mi, CUDART_LN2_F, logf(li));  // = mi*ln2 + log(li)
```

For backward recomputation:
```
prob[kk,h] = exp(p_natural[kk,h] - lse[h])
```

The backward kernel only needs `lse`. The `max_logits` is an internal forward artifact for the streaming max-tracking during online softmax. The backward doesn't stream — it recomputes `P_t` from scratch and has the full B_TOPK block available, so it can compute its own block-local max for numerical stability.

**Resolution:** Remove `max_logits` from the backward kernel's input list. Use only `lse` for probability recomputation. If numerical stability during `exp(p - lse)` is a concern (underflow for very negative p values), use the same pattern as forward: compute block-local max, subtract it, exp, then scale by `exp(block_max - lse)`.

---

## Things to Add

### 4. `sum_odo[h]` reduction strategy

The plan says "Compute `sum_odo[h] = dot(O[h,:], dO[h,:])` cooperatively across lanes" but `D_V=128 > warp_size=32`. With `B_H ≤ 32` heads and 128 threads in WG0, you have `128/B_H` threads available per head.

Spell out the mapping:
- B_H=32: each head gets 4 threads, each computes 32 elements of the dot product, then reduce via `__shfl_xor_sync` across 4 threads (2 rounds).
- B_H=16: each head gets 8 threads, each computes 16 elements, 3 rounds of shfl reduction.
- B_H=8: each head gets 16 threads, each computes 8 elements, 4 rounds.

Alternative: assign warp 0 + warp 1 (64 threads) where each thread handles one head across two lanes for the reduction. Whichever approach, the sync/barrier structure must account for when `sum_odo` is ready for the softmax derivative computation.

**Recommendation:** Use `idx_in_warpgroup % B_H` as head index, `idx_in_warpgroup / B_H` as the partition index within the head. Each partition computes `D_V / (128/B_H)` elements. Reduce with shared memory (write partial sums → barrier → read and sum).

---

### 5. Shared memory byte budget

Add a concrete table. Worst-case (B_H=32, D_QK=192, NUM_BUFS=2):

| Buffer | Size (bytes) | Notes |
|--------|-------------|-------|
| q_nope [B_H, 128] bf16 | 8,192 | Persistent, UMMA K-major layout |
| q_rope [B_H, 64] bf16 | 4,096 | D_QK=192 only |
| dO [B_H, 128] bf16 | 8,192 | Persistent |
| O [B_H, 128] bf16 | 8,192 | For sum_odo, can alias with dq_smem after prologue |
| k_nope[2] [B_TOPK, 128] bf16 | 32,768 | Double-buffered |
| k_rope [B_TOPK, 64] bf16 | 8,192 | Single-buffered for v1 |
| p_t [B_TOPK × B_H] float | 8,192 | Score staging |
| dp_t [B_TOPK × B_H] float | 8,192 | Can alias p_t after ds_t produced |
| ds_t [B_TOPK × B_H] bf16 | 4,096 | MMA operand |
| lse/D/head_scale float[B_H] ×3 | 384 | |
| indices[2] int[B_TOPK] | 512 | |
| valid[2] char[B_TOPK/8] | 16 | |
| dq_smem [B_H, D_QK] float | 24,576 | Staging before global store |
| dk_smem [B_TOPK, D_QK] bf16 | 24,576 | Staging for scatter (aliases dq_smem) |
| Barriers (~15 barriers) | ~240 | |
| **Total (with aliasing)** | **~105 KB** | |

SM100 supports 228 KB SMEM per SM — fits comfortably. Apply the forward's union trick to alias:
- `O` with `dq_smem` (O only needed during prologue for sum_odo)
- `p_t` with `dp_t` (sequential use)
- `dk_smem`/`dv_smem` with score buffers (during final epilogue)

---

### 6. Register allocation directives

Forward uses:
- WG0 (compute/softmax): `warpgroup_reg_alloc<176>()`
- WG1 (loader): `warpgroup_reg_dealloc<80>()`
- WG2 (MMA): `warpgroup_reg_dealloc<80>()`

Backward WG0 carries more state: `sum_odo[B_H]`, score values, prob values, ds values, plus epilogue staging. Estimate:
- `sum_odo`: B_H floats = 32 regs max
- Score copy from TMEM: B_H_TMEM floats = 32 regs
- prob/ds computation: 2 × B_TOPK partial values (loop-carried)
- Epilogue: dQ/dK/dV read from TMEM = B_H_TMEM floats per tile

**Recommendation:** Keep `warpgroup_reg_alloc<176>()` for WG0 (should be sufficient). Verify with `--ptxas-options=-v` after initial compilation. If spilling occurs, consider splitting WG0's work across warp pairs (warps 0-1 for score exchange, warps 2-3 for epilogue).

---

### 7. Grid launch configuration

State explicitly:
```cpp
kernel<<<s_q, 384, smem_size, stream>>>(params, tma_params);
// __launch_bounds__(384, 1, 1)
// No cluster (1-SM mode), matching head_small forward
```

Each CTA processes one query row's full topk list, so grid = `[s_q, 1, 1]`.

---

### 8. dQ transpose strategy

The kernel computes `dQ_t [D_QK_tile, B_H]` in TMEM but needs to store `dQ [B_H, D_QK]` to global memory.

Forward's O epilogue pattern:
```cpp
// Read TMEM tile-by-tile, each lane gets B_H values for one dv position
float o_head[B_H_TMEM];
ku::tmem_ld_32dp32bNx<B_H_TMEM>(tmem_cols::O + tile*B_H_TMEM, o_head);
// Write transposed: o_head[h] goes to out[h, dv]
```

For dQ, apply the same pattern:
```cpp
// For each dq tile (64-wide):
float dq_vals[B_H_TMEM];
int dq_pos = warp_idx * 32 + lane_idx;  // covers 128 positions with 4 warps
int tile = dq_pos / 64;
ku::tmem_ld_32dp32bNx<B_H_TMEM>(tmem_cols::DQ + tile * B_H_TMEM, dq_vals);
// Write to dq_smem[h][dq_pos] or directly to global dQ
for (int h = 0; h < B_H; ++h) {
    dq_smem[h * D_QK + dq_pos] = dq_vals[h];
}
```

Since dQ is accumulated across all sparse blocks within one CTA (no cross-CTA reduction needed when grid = [s_q]), the final store is a direct BF16 write after FP32→BF16 conversion.

**Recommendation:** Adopt the forward's `tmem_ld_32dp32bNx` + transposed-write pattern. For D_QK=192, need 3 tiles of 64. With 4 warps (128 threads), each covering 32 positions, two passes cover all 192 positions (warp 0-3 cover 0..127 in pass 1, warps 0-1 cover 128..191 in pass 2).

---

### 9. attn_sink gradient reduction

`attn_sink` shape: `[n_q_heads]` (a global `nn.Parameter`). Each query CTA emits B_H partial gradient contributions.

For v1, use FP32 `atomicAdd`:
```cpp
// In epilogue, after computing dSink[h]:
float d_sink_h = sink_prob_h * (0.0f - sum_odo_h);  // * sm_scale if needed
atomicAdd(&params.d_attn_sink[head_idx], d_sink_h);
```

Contention analysis: `n_q_heads` atomics per CTA × `s_q` CTAs. With `n_q_heads ≤ 32` and `s_q` potentially large, each `d_attn_sink[h]` receives `s_q` atomic updates. This is acceptable for v1 — L2 atomic throughput on SM100 handles this scale.

**Recommendation:** Explicit atomicAdd for v1. If profiling shows atomic pressure (unlikely for 32 addresses), switch to a per-CTA partial buffer + final reduction kernel.

---

### 10. Convert kernel specification

The FP32 `dk_acc/dv_acc` workspace requires a final cast kernel:

```cpp
// convert_dk_dv_kernel<<<ceil_div(s_kv * D, 1024), 1024, 0, stream>>>
__global__ void convert_fp32_to_bf16(
    const float* __restrict__ src,  // dk_acc or dv_acc, shape [s_kv, D]
    bf16* __restrict__ dst,         // final dK or dV, shape [s_kv, D]
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}
```

This should be listed explicitly as a deliverable in Milestone 3. Consider vectorized loads/stores (float4 → bf16x4) for bandwidth efficiency.

---

### 11. Clarify that attn_sink has zero dV contribution

In the forward:
```
o[h,dv] = sum_kk prob[kk,h] * v[kk,dv] + sink_prob[h] * 0
```

The sink contributes to the denominator (normalization) but has zero value. Therefore:
- `dV` receives NO contribution from the sink term.
- `dQ` receives NO contribution from the sink term (sink doesn't depend on q directly — it's a learned bias on the logit that's head-specific but query-independent).
- `d_attn_sink[h]` is the only gradient for the sink parameter.

The formula:
```
d_attn_sink[h] = sink_prob[h] * (0 - D[h]) * sm_scale
```
where `sink_prob[h] = exp(attn_sink[h] - lse[h])` (or equivalently from forward's representation: `exp2(attn_sink_log2 - mi) / li`).

State this explicitly to avoid implementing unnecessary dV/dQ contributions from the sink.

---

### 12. Pipeline depth vs double-buffer compatibility

The optimized steady-state pipeline:
```
WG1: load K/V(k+1)
WG2: compute P_t(k) = K(k) @ Q^T
WG0: softmax/dS for k-1, epilogue/store dV/dK for k-2
```

This implies `k-2` epilogue runs while `k` occupies one buffer and `k+1` is being loaded. With `NUM_BUFS=2`, buffer `(k-2)%2 == k%2`, so the epilogue would conflict with the current MMA's KV data.

**Resolution:** The dK/dV epilogue (step 10 in the plan) reads from TMEM (dK, dV accumulators), NOT from K/V SMEM. The only SMEM data needed for scatter is the `indices` buffer. Since:
- `dK = dS_t @ Q` — uses Q (persistent) and dS_t (in separate SMEM region)
- `dV = Prob_t @ dO` — computed by MMA into TMEM, read from TMEM during epilogue

The K/V SMEM can safely be overwritten. Clarify this dependency chain in the plan to prevent confusion:

> The dK/dV epilogue reads TMEM accumulators and the persistent `indices` buffer. It does NOT re-read K/V SMEM. Therefore `NUM_BUFS=2` is sufficient for the 3-stage pipeline.

However, note that `indices` also needs double-buffering if the next block's indices are being loaded while the current epilogue still uses the old indices for scatter addresses. Add `indices[2]` double-buffering to the plan.

---

### 13. MMA atom Major selection — resolve before implementation

Critical constraint analysis:

**P_t = K @ Q^T** `[B_TOPK, B_H]`:
- K is `[B_TOPK, D_QK]` in SMEM (K-major along D_QK, i.e., reduction dim is contiguous) → `Major::K`
- Q is `[B_H, D_QK]` in SMEM, used as Q^T → reduction dim D_QK is along the row of Q → `Major::K`
- Same as forward ✓

**dP_t = V @ dO^T** `[B_TOPK, B_H]`:
- V is `[B_TOPK, D_V]` stored with D_V contiguous (same swizzled layout as K NoPE) → `Major::K`
- dO is `[B_H, D_V]`, used as dO^T → `Major::K`
- Atom: `SM100_MMA_F16BF16_SS_NOELECT<bf16, bf16, float, B_TOPK, B_H, Major::K, Major::K>`

**dV = Prob_t^T @ dO** → More precisely: accumulate `[B_TOPK, D_V]`:
- Need Prob_t^T `[B_TOPK, B_H]` × dO `[B_H, D_V]` → result `[B_TOPK, D_V]`
- Prob_t is already `[B_TOPK, B_H]` in SMEM — but we need it as the A-operand with reduction along B_H.
- SMEM layout for Prob_t: `[B_TOPK, B_H]` K-major (B_H contiguous) → already has `Major::K` semantics for this MMA.
- dO is `[B_H, D_V]` as B-operand with reduction along B_H → needs `Major::K` (B_H contiguous along the "K" dimension).
- But dO is stored as `[B_H, D_V]` with D_V contiguous (K-major along D_V, NOT along B_H). This means dO is `Major::MN` for this particular MMA.
- Atom: `SM100_MMA_F16BF16_SS_NOELECT<bf16, bf16, float, B_TOPK, 64, Major::K, Major::MN>`
- Alternatively: transpose dO to `[D_V, B_H]` in SMEM → then use `Major::K` for both. This matches the plan's notation "dO^T".

**dK = dS_t @ Q** `[B_TOPK, D_QK]`:
- dS_t is `[B_TOPK, B_H]` K-major → `Major::K`
- Q is `[B_H, D_QK]` K-major (D_QK contiguous) → this is `Major::MN` for the MMA (since B_H is the reduction dim).
- Wait: Q as B-operand `[B_H, 64]` per tile, reduction along B_H. Q is stored with D_QK contiguous, so the B_H stride is D_QK. The UMMA atom with `Major::MN` expects the MN-dimension to be the contiguous/swizzled one — but here D_QK (the MN output dimension) IS contiguous. So `Major::MN` is correct.
- Atom: `SM100_MMA_F16BF16_SS_NOELECT<bf16, bf16, float, B_TOPK, 64, Major::K, Major::MN>`

**dQ_t = K^T @ dS_t** `[D_QK, B_H]`:
- K^T: K is `[B_TOPK, D_QK]` K-major. Used as K^T → `[D_QK, B_TOPK]` with B_TOPK (reduction) not contiguous in original layout.
- This is problematic: the A-operand for this MMA is `[64, B_TOPK]` per tile, reduction along B_TOPK. K is stored `[B_TOPK, D_QK]` which makes B_TOPK the "row" and D_QK the contiguous "K" direction. Transposing means: A-operand has shape `[64, B_TOPK]` where 64 comes from D_QK and B_TOPK is the reduction dim. In the original layout, a [64]-wide slice of D_QK is contiguous — this is `Major::MN`.
- dS_t is `[B_TOPK, B_H]` as B-operand, reduction along B_TOPK (row of dS_t). B_H is contiguous → `Major::MN`.
- Atom: `SM100_MMA_F16BF16_SS_NOELECT<bf16, bf16, float, 64, B_H, Major::MN, Major::MN>`

**Summary table to add to the plan:**

| MMA | Shape | A operand | A Major | B operand | B Major |
|-----|-------|-----------|---------|-----------|---------|
| P_t = K @ Q^T | [B_TOPK, B_H] | K [B_TOPK, D] | K | Q [B_H, D] | K |
| dP_t = V @ dO^T | [B_TOPK, B_H] | V [B_TOPK, D_V] | K | dO [B_H, D_V] | K |
| dV = Prob_t^T @ dO | [B_TOPK, 64] | Prob_t [B_TOPK, B_H] | K | dO [B_H, 64] | MN |
| dK = dS_t @ Q | [B_TOPK, 64] | dS_t [B_TOPK, B_H] | K | Q [B_H, 64] | MN |
| dQ_t = K^T @ dS_t | [64, B_H] | K_tile [64, B_TOPK]^T | MN | dS_t [B_TOPK, B_H] | MN |

**Key implication:** dO needs to be stored in TWO layouts in SMEM:
1. `[B_H, D_V]` K-major (D_V contiguous) — for dP_t = V @ dO^T computation
2. `[B_H, D_V]` tiled as [B_H, 64] with B_H contiguous per tile — for dV computation (Major::MN)

Or equivalently, store `dO^T [D_V, B_H]` in K-major (B_H contiguous) and use:
- For dP_t: read dO^T as B-operand `Major::K` ✓
- For dV: read dO^T sliced as `[64, B_H]` with B_H as MN-dim (contiguous) → `Major::MN` on the transposed copy

This confirms the plan should explicitly state: **store dO in transposed form `[D_V, B_H]`** in SMEM to unify both use cases.

---

## Minor Suggestions

### 14. Simplify v1 barriers to single-buffered

The barrier naming `bar_qk_done[2]`, `bar_dp_done[2]` uses double-buffering indices, but the plan's first-version sequence is fully serial (step 1 → 2 → ... → 11). For v1 correctness:
- Use single barriers (no `[2]` array)
- Only introduce double-buffered variants during the optimization phase (Milestone 6)

This reduces the initial implementation complexity and barrier initialization code.

---

### 15. Pre-compute probability scaling in prologue

Since `lse[h]` is loaded once per query row during the prologue and remains constant across all sparse blocks, consider precomputing:
- `neg_lse[h] = -lse[h]` for fast probability computation: `prob = exp(p + neg_lse)`

This is a minor optimization but simplifies the inner loop code and matches the forward's precomputation of `head_mi`/`head_li`.

---

### 16. RoPE gradient ordering for D_QK=192

For `D_QK=192`, the dQ and dK gradients split into:
- NoPE part: dimensions 0..127 (128 wide, 2 tiles of 64)
- RoPE part: dimensions 128..191 (64 wide, 1 tile)

The forward computes "RoPE partial first, then NoPE with accumulate" for P_t. The backward should compute NoPE gradients first (larger, more important for pipeline overlap), then RoPE:

```
dK_nope = dS_t @ Q_nope    [B_TOPK, 128]  — 2 tiles
dK_rope = dS_t @ Q_rope    [B_TOPK, 64]   — 1 tile
dQ_nope_t = K_nope^T @ dS_t [128, B_H]    — 2 tiles
dQ_rope_t = K_rope^T @ dS_t [64, B_H]     — 1 tile
```

Order them to maximize overlap:
1. `dK_nope` (2 tiles, long) — WG0 can begin epilogue for previous block during this
2. `dK_rope` (1 tile)
3. `dQ_nope_t` (2 tiles)
4. `dQ_rope_t` (1 tile) — by this time, WG1 has loaded K(k+1) ready for next iteration

Ensure `dK_nope` and `dK_rope` use separate TMEM regions (128 + 384 respectively) so they accumulate independently across blocks without clobbering.

---

## Priority for Resolution Before Implementation

| Priority | Item | Risk if Deferred |
|----------|------|-----------------|
| P0 (must fix) | #2 TMEM widths | Implementation will hit silent corruption |
| P0 (must fix) | #3 lse vs max_logits | Wrong interface → wrong results |
| P0 (must fix) | #13 MMA Major selection | Wrong SMEM layouts → IMA errors or wrong results |
| P0 (must fix) | #8 dQ transpose | Affects TMEM→global store correctness |
| P1 (should fix) | #5 SMEM budget | Risk of exceeding 228KB or leaving performance on table |
| P1 (should fix) | #4 sum_odo strategy | Affects WG0 sync structure |
| P1 (should fix) | #12 Pipeline/buffer hazard | Potential data race in optimized version |
| P2 (nice to have) | #6 Register allocation | Can be tuned post-compilation |
| P2 (nice to have) | #14 Single-buffer v1 | Simplification, not correctness |
| P2 (nice to have) | #15 Precompute neg_lse | Minor optimization |
| P2 (nice to have) | #16 RoPE ordering | Performance tuning |
