# MXFP8 DSA Prefill Kernel — Bug-Fix Plan (D_QK=512, D_V=448)

> Target file: `csrc/sm100/prefill/sparse/mxfp8_fwd/head64/phase1.cuh`
> Companion file: `csrc/sm100/prefill/sparse/mxfp8_fwd/head64/config.h`
> API: `csrc/sm100/prefill/sparse/mxfp8_fwd/head64/phase1.h`
> Instantiation: `csrc/sm100/prefill/sparse/mxfp8_fwd/head64/instantiations/phase1_k512.cu`
>
> Scope of this revision: get `D_QK=512`, `D_V=448` (i.e. the `MX_FP8` variant) to
> compile, link against `csrc/api/mxfp8_sparse_fwd.h`, and produce numerically
> correct attention output. Performance is secondary — we only fix correctness,
> typed-in declarations, and the explicit TODO items that the kernel already
> flags.
>
> Reference (non-mxfp8) for sanity: `csrc/sm100/prefill/sparse/fwd/head64/`.

## Background

The kernel implements DSA (DeepSeek Sparse Attention) prefill with:

```text
Q :  e4m3 NoPE [B_H, D_V=448] + UE8M0 scales [B_H, 14] + bf16 RoPE [B_H, 64]
K  :  same layout as Q (448 NoPE e4m3, 8 bytes of scales incl. padding, 64 RoPE bf16)
V  :  reuses K's NoPE bytes (so V = K's NoPE, 448 e4m3 + scales)
P := QK^T  :  [B_TOPK, B_H]            ; online softmax, FP32 in TMEM
S := P*sm_scale_div_log2 → exp2 → quantized e4m3 [B_H, B_TOPK]  + UE8M0 scales
O := S @ V  :  [B_H, D_V]              ; in TMEM, then rescaled per block
```

Pipeline depth: `NUM_BUFS=3`, `B_TOPK=128`, `B_H=64`, `NUM_THREADS=384` (3
warpgroups).

Warpgroups:

```text
WG0 (warps 0..3)   : scale & exp + epilogue (TMEM rescale, O store via TMA)
WG1 (warps 4..7)   : KV NoPE producer (TMA gather4 + per-row scale scatter)
WG2 (warps 8..11)  : MMA issuer (UTCCP Q scales → tcgen05.mma, P / S / O MMAs)
                     warp 8  : Q UTCCP, P/QK/ROPE, SV
                     warp 9  : KV valid mask producer
                     warp 10,11 : K RoPE cp.async loader
```

## Catalogue of Bugs

The bugs are listed in roughly the order they need to be fixed, P0 first.

### P0 — `run_fwd_phase1_kernel` does not exist; no kernel is ever launched

`phase1.h:10` declares `run_mxfp8_fwd_phase1_kernel<D_QK>`, but `phase1.cuh:547`
defines `run_fwd_phase1_kernel` (different name). The instantiation in
`instantiations/phase1_k512.cu:6` instantiates `run_mxfp8_fwd_phase1_kernel<512>`.
Result: linker error.

Also, the `run_fwd_phase1_kernel<D_QK>` body in `phase1.cuh:548-622` builds the
TMA descriptors and then **falls off the end** without ever calling
`kernel<<<...>>>(params, tma_params)`. So even after the rename, nothing runs.

**Fix:**

1. Rename `run_fwd_phase1_kernel` → `run_mxfp8_fwd_phase1_kernel` in
   `phase1.cuh:547`.
2. Add the missing `TmaParams` struct to `mxfp8_fwd/head64/config.h`. It must
   hold the same six things the kernel reads:
   `shape_O`, `tma_O`, `shape_Q_scale`, `tma_Q_scale`, plus the three
   `CUtensorMap`s `tensor_map_q_nope`, `tensor_map_q_rope`, `tensor_map_kv_nope`,
   `tensor_map_kv_rope`. Model after `fwd/head64/config.h:17` and add the
   `tma_Q_scale` field.
3. Define `SmemLayoutOBuf_TMA = SmemLayoutOTiles<1>` in `config.h` (the
   `run_*` body references it at `phase1.cuh:566`).
4. Add the missing `ku::make_tensor_map_kv_nope` setup with `D_NOPE/8` int64 box
   (already drafted in `phase1.cuh:612-620` — leave as-is but feed it into
   `TmaParams`).
5. Add the missing `ku::make_tensor_map_kv_rope` setup with `D_ROPE` bf16
   (already drafted in `phase1.cuh:602-610`).
6. At the end of `run_mxfp8_fwd_phase1_kernel`, after the TMA builders,
   instantiate `TmaParams`, call
   `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)`,
   and launch `kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params, tma_params)`.
   The reference body is at `fwd/head64/phase1.cuh:664-670`.

### P0 — `SharedMemoryPlan` member names do not match the kernel

The struct in `config.h:113-143` exposes:

```cpp
union {
    struct { bf16 q_rope; e4m3 q_nope; } q;
    struct { e4m3 kv_nope[NUM_BUFS]; bf16 kv_rope; e8m0 kv_nope_scale[NUM_BUFS]; } kv;
    bf16 o[cosize_v<SmemLayoutO>];
} qkvo;
union { e4m3 s[B_H*B_TOPK]; e8m0 q_scale[cosize_v<SmemLayoutQScale>]; } s_q_scale;
```

The kernel, however, repeatedly reads these nonexistent members:

| Line | Bad expression | Correct expression |
| --- | --- | --- |
| `phase1.cuh:97`  | `plan.s_q_rope.q_tail.q_scale.data()` | `plan.s_q_scale.q_scale.data()` |
| `phase1.cuh:167` | `make_smem_ptr(plan.s)` | `make_smem_ptr(plan.s_q_scale.s.data())` (and use `SmemLayoutS` correctly) |
| `phase1.cuh:390` | `plan.s_q_rope.q_tail.q_scale.data()` | `plan.s_q_scale.q_scale.data()` |
| `phase1.cuh:472` | `make_smem_ptr(plan.s_q_rope.s)` | `make_smem_ptr(plan.s_q_scale.s.data())` |

**Fix:** Replace all four sites with the correct member access. (Optionally,
also add an `s_scale[NUM_QUANT_GROUPS]` element to `s_q_scale` if you want the
scale/exp logic to write its `e8m0 scale_g` values there; see P0 — tS_scale
below.)

### P0 — `bar_prologue_utccp_rope` / `bar_prologue_utccp_nope` do not exist

`phase1.cuh:108-109` initialises two barriers that are not declared in
`SharedMemoryPlan`:

```cpp
plan.bar_prologue_utccp_rope.init(1);    // line 108
plan.bar_prologue_utccp_nope.init(1);    // line 109
```

`config.h:134` only declares `bar_prologue_utccp_q_scale` and
`bar_prologue_utccp_k_scale`. The `bar_prologue_utccp_*` field is what the
UTCCP step later `umma_arrive_noelect`s into. The same comment also applies to
`phase1.cuh:412` (`umma_arrive_noelect(plan.bar_prologue_q_scale)` — this name
is also wrong; the field is `bar_prologue_utccp_q_scale`).

Also `phase1.cuh:116` says `bar_k_valid_ready[i].init(B_TOPK/8)` with
`// TODO: Check the NUmber here`. With `B_TOPK = 128`, that is 16 threads per
buffer producing validity, but only `lane_idx < B_TOPK/8 = 16` lanes run in
warp 9 (`phase1.cuh:490`), so `init(16)` is correct. Keep it. (Drop the TODO.)

**Fix:** Replace both init sites and the umma_arrive to use
`bar_prologue_utccp_q_scale` / `bar_prologue_utccp_k_scale` (matching the
struct).

### P0 — `B_H_TMEM` is undefined

`phase1.cuh:22` declares a templated function `rescale_O_t<B_H, B_H_TMEM, …>`.
`phase1.cuh:220` calls it as `rescale_O_t<Kernel>(plan.head_scale)` (no extra
args). `phase1.cuh:255-258` and the function body itself use
`Kernel::B_H_TMEM`, but the `Kernel` alias is not declared anywhere in the
mxfp8 namespace and `B_H_TMEM` is not defined in `config.h`. The `B_H` for
this kernel is fixed at 64, so `B_H_TMEM = B_H = 64` (one TMEM row per
`head_real_mi[h]`).

**Fix:** Add to `config.h`:

```cpp
namespace sm100::mxfp8_fwd::head64 {
constexpr int B_H_TMEM = B_H;        // 64; P / O tmem columns are 64-wide rows
}
```

(or use the same value through `Kernel` if you wrap things in a struct later).
Verify with the existing `tmem_ld_32dp32bNx<B_H_TMEM>` calls in
`rescale_O_t` (line 28, 34) and the load at `phase1.cuh:258`.

### P0 — `SmemLayoutO` and `SmemLayoutKNoPE` are sized for `D_V=512`, not `D_V=448`

`config.h:83`:

```cpp
using SmemLayoutO = SmemLayoutOTiles<8>;        // 8 * 64 = 512
```

`config.h:92`:

```cpp
using SmemLayoutKNoPE = SmemLayoutKTiles<8>;    // 8 * 64 = 512
```

But `D_V = 448`, so `D_V / 64 = 7`. The two `// TODO: WHY 8? For MODEL1 it
might should be 7?` comments at `config.h:83,92` flag this directly. The
SM100 `UMMA::Layout_K_SW128_Atom` swizzle and the TMA `tensor_map_kv_nope` box
of `D_NOPE/8 = 56` int64s (8 bytes per int64 = 448 bytes — already correct at
`phase1.cuh:613`) both match `D_V=448`. Only the swizzled SMEM layouts are
wrong.

`SmemLayoutV` is fine because it is built by composing `SmemLayoutKNoPE` with
`Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}`, so when
KNoPE is 7-tile (448 cols), V automatically becomes 7-tile too.

**Fix:** Change both `SmemLayoutO` and `SmemLayoutKNoPE` to `<7>`. The O TMA
store in `run_*` should also use `SmemLayoutOTiles<1>{}` (64-byte tile = one
TMA copy per `B_EPI=64` element) for the store, like the reference at
`fwd/head64/phase1.cuh:627`. The `D_V/B_EPI = 448/64 = 7` copy loop at
`phase1.cuh:281` already matches.

### P0 — `SmemLayoutKNoPE_TiledMMA` / `SmemLayoutKRoPE_TiledMMA` are undefined

`phase1.cuh:419-420` references both:

```cpp
Tensor sK_nope = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()), SmemLayoutKNoPE_TiledMMA{});
Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()),     SmemLayoutKRoPE_TiledMMA{});
```

These layouts are defined in `fwd/head64/config.h:92-102` (the dual-gemm
re-view of K: `B_TOPK*2 × D_V/2` for KNoPE and `B_TOPK*2 × 64/2` for KRoPE).
The mxfp8 config.h does not declare them.

**Fix:** Copy the two aliases verbatim from
`fwd/head64/config.h:92-102` into `mxfp8_fwd/head64/config.h`. (They use
`bf16` atom types — they are used here to build a 128-row × 64-col view of
the e4m3 bytes; this is a forged layout for CuTe address generation, exactly
as in the reference.)

### P0 — `tQ_scale` / `tK_scale` / `tS_scale` TMEM fragments

`phase1.cuh:78-79` does:

```cpp
tQ_scale.data().get() = tmem_cols::Q_Scale;
tK_scale.data().get() = tmem_cols::K_Scale;
```

But `tQ_scale` and `tK_scale` are not declared in scope yet — they are
declared locally at lines 394, 439 inside the `warp_idx == 8` branch. The
declarations at the function top level are missing.

In addition, `tS_scale` is used at `phase1.cuh:481` (in the `SV` MMA) but is
never declared and its TMEM region is never populated:

```cpp
ku::utcmma_blockscaled_ss(tiled_mma_O, sV, sS, tS_scale, tK_scale, tO, k == 1);
//TODO: We need to produce tS_scale in tmem in warpgroup idx 0
```

For the second argument of `tmem_cols::K_Scale = 338`, it is set at
`phase1.cuh:442` (correct). For `tS_scale`, the S_Scale region at
`tmem_cols::S_Scale = 356` is declared in `config.h:50` but nothing writes to
it. The scale/exp logic computes `s_scale[g] = e8m0(absmax_p[g] / FP8_MAX)` at
`phase1.cuh:203` and then the same `s_scale` array is immediately reused for
the next `g`. To populate the TMEM, this loop must additionally call
`SM100_UTCCP_*::copy` (matching the Q/K scale pattern) to send each `e8m0`
value into the S_Scale TMEM region.

**Fix:**

1. Hoist the `tQ_scale` / `tK_scale` declarations to the top of the kernel
   function (before line 80), so they can be assigned `data().get()`.
2. Add `tS_scale` as another TMEM fragment and assign
   `tS_scale.data().get() = tmem_cols::S_Scale;` at the same place.
3. In the scale/exp block (`phase1.cuh:200-212`), after computing
   `s_scale[g]`, push the `e8m0` value into the S_Scale TMEM via UTCCP. The
   simplest path: write `s_scale[g]` into a `e8m0[NUM_QUANT_GROUPS]` SMEM
   staging buffer, then issue one `SM100_UTCCP_4x32dp128bit_1cta::copy` (or
   similar `128dp32b`-style atom) into `tS_scale`. Until then, leave
   `tS_scale` out of the SV MMA and switch the call to
   `ku::utcmma_ss(tiled_mma_O, sS, sV, tO, k == 1)` (no block scale), so
   the kernel is at least numerically correct (S is already scaled by
   `1/scale_g` at `phase1.cuh:209`).
4. Remove the `//TODO: We need to produce tS_scale in tmem in warpgroup idx 0`
   comment once the choice is made explicit.

### P0 — The Q-scale copy uses `sQ_scale` that reads from a nonexistent `s_q_rope.q_tail`

`phase1.cuh:389-410` (in the `warp_idx == 8` Q-UTCCP branch) tries to build
`sQ_scale` from `plan.s_q_rope.q_tail.q_scale.data()`. As noted above, the
correct member is `plan.s_q_scale.q_scale.data()`. The downstream
`make_utccp_copy(... SM100_UTCCP_4x32dp128bit_1cta ...)` call then needs the
SFB fragment shape.

**Fix:** Replace the member access (already covered above) and verify that
`SmemLayoutQScale` and `TiledMMA_P::FrgTypeSFB` line up. For the typical
MXFP8 layout used elsewhere in the kernel, `TiledMMA_P::FrgTypeSFB` is the
correct type for the SFB fragment (see
`fwd/head64/.../sm100_blockscaled_mma_array_warpspecialized.hpp:652`).

### P0 — `Tensor sQ_rope = make_tensor(make_smem_ptr(...), ...)` and `sQ_nope` use `…` placeholders

`phase1.cuh:376` and `phase1.cuh:382` literally have `...` where a smem
pointer should be:

```cpp
Tensor sQ_rope = make_tensor(make_smem_ptr(...));
Tensor sQ_nope = make_tensor(make_smem_ptr(...));
```

**Fix:** Replace with the right SMEM pointers and layouts:

```cpp
Tensor sQ_rope = make_tensor(make_smem_ptr(plan.s_q_scale.q_rope.data()), SmemLayoutQRoPE{});
Tensor sQ_nope = make_tensor(make_smem_ptr(plan.qkvo.q.q_nope.data()),    SmemLayoutQNoPE{});
```

(Note: the `q_rope` portion currently does not exist in `s_q_scale`. The
`SharedMemoryPlan::s_q_scale` union only carries `s` and `q_scale`. Add
`q_rope` to the union, or alias it onto `qkvo.q.q_rope`. The simplest fix
that keeps smem union semantics is to also have a `q_rope` slot inside the
`s_q_scale` union so the struct still works at the `D_QK=512` size.)

### P0 — `K_SCALE_BYTES` and `K_QUANT_GROUP_SIZE` are not defined in this namespace

`phase1.cuh:344` (`e8m0 scale[K_SCALE_BYTES]`) and `phase1.cuh:25`
(`K_SCALE_DUP = K_QUANT_GROUP_SIZE / MXFP8_SCALE_VEC_SIZE;` in `config.h:25`)

`config.h:25` uses `K_QUANT_GROUP_SIZE` and `K_SCALE_DUP` but only
`K_SCALE_DUP` is defined. The reference is the `defines.h` file. Check
`defines.h` for `K_QUANT_GROUP_SIZE` — if absent, add a `constexpr int
K_QUANT_GROUP_SIZE = 64;` next to `MXFP8_SCALE_VEC_SIZE = 32;` in
`config.h:22`. Add `constexpr int K_SCALE_BYTES = D_NOPE / K_QUANT_GROUP_SIZE
+ /*padding*/ 1;` so that 8 bytes (one per quant group) plus padding fit
into 8 uint8_t — the K scale region in the SM100 layout has
`D_NOPE / 32 = 14` scale values per token, but the Q scale layout is `[B_TOPK,
B_H, D_NOPE/32]` and a single 32-byte-aligned `int4` (8 bytes) is loaded per
K-row. The exact value depends on the chosen `K_SCALE_BYTES`; pick the value
that matches the existing per-row `int4` load in `phase1.cuh:347` and the
sizeof `e8m0 scale[K_SCALE_BYTES]`.

**Action:** grep `K_QUANT_GROUP_SIZE` and `K_SCALE_BYTES` across `csrc/`
before deciding; for `D_NOPE=448, K_QUANT_GROUP_SIZE=64`, the natural values
are `K_SCALE_BYTES=8` (7 used + 1 padding). Document the choice inline.

### P0 — `FP8_MAX` is undefined

`phase1.cuh:203` uses `FP8_MAX` in `e8m0 scale_g = e8m0(absmax_p[g] / FP8_MAX);`
There is no `FP8_MAX` in scope. The standard E4M3 max is `448.0f`. The
commented `//TODO: change here to vectorized type conversion` suggests the
author already knows this is half-finished.

**Fix:** Add a `constexpr float FP8_MAX = 448.0f;` to `config.h:31` (next to
`MAX_INIT_VAL`).

### P1 — `is_k_valid` mask and `B_TOPK/8` indexing

`phase1.cuh:168`:

```cpp
uint64_t valid_mask = *(uint64_t*)plan.is_k_valid[k%NUM_BUFS];
```

`is_k_valid` is a `char[NUM_BUFS][B_TOPK/8]` array. A 64-bit read of 8
chars is fine, but the loop `kk = k + 32 * g` (line 176) iterates `k = 0..31`
inside `g = 0..3`, so it covers all 128 elements of the B_TOPK row. The
mask is read once at the top, before the inner `kk` loop. That is correct.

But `for (int k = 0; k < 32; ++k) { int kk = k + 32 * g; ... }` uses
`NUM_QUANT_GROUPS = B_TOPK/MXFP8_SCALE_VEC_SIZE = 128/32 = 4`. With
`MXFP8_SCALE_VEC_SIZE=32`, each scale covers 32 P elements. The MMA `kind::mxf8`
atom takes scales per 32 K elements along the K-dim, which matches
`B_TOPK=128`. The math is consistent.

`plan.is_k_valid[cur_buf][lane_idx] = k_validness_mask;` (line 503) writes
one byte per `lane_idx < B_TOPK/8 = 16` lane — the `init(16)` at line 116
matches. Keep the TODO removed.

### P1 — Stride check between `k` and `k_row` in the P transpose

`phase1.cuh:158`:

```cpp
int k_row = lane_idx + (warp_idx&3);  // 0..127
for (int h = 0; h < B_H; ++h) {
    plan.p_t[h + B_H * k_row] = p[h];
}
```

`p[h]` is loaded as `B_H` floats from TMEM at `tmem_cols::P + warp_idx *
NUM_ELEMS_PER_THREAD` (line 154). With `NUM_ELEMS_PER_THREAD = B_H = 64` and
4 warps, each warp loads a 64×64 piece of P. The transpose writes into
`p_t[h + B_H * k_row]` (column-major). The subsequent
`plan.p_t[kk*B_H + h]` (line 177) reads the transposed view. This is correct,
but the loop only iterates `B_H=64` heads and writes to the right column of
`p_t` for the lane's `k_row`. Multiple warps write to different `k_row`
ranges — no race. OK.

**Action:** No change; keep the `// How to improve the transpose efficiency
here?` comment as a perf TODO, not a correctness TODO.

### P1 — `k` is captured by reference inside the scale/exp branch

`phase1.cuh:214-222`:

```cpp
if (k > 0) {
    plan.bar_sv_done[(k-1)%Kernel::NUM_BUFS].wait(((k-1)/Kernel::NUM_BUFS)&1);
}
if (k > 0) {
    ku::tcgen05_after_thread_sync();
    rescale_O_t<Kernel>(plan.head_scale);
    ku::tcgen05_before_thread_sync();
}
```

`k` is the loop variable. This is fine because both blocks are inside the
same `for (int k = 0; k < num_k_blocks; ++k)` scope. The redundant `if (k > 0)`
should be merged into a single block; cosmetic, not a bug. The use of
`Kernel::NUM_BUFS` is fine once `Kernel` is defined (or replace with the
file-scope `NUM_BUFS` constant).

### P1 — Output epilogue: `B_H_TMEM` template arg of `rescale_O_t` and store order

`rescale_O_t<B_H, B_H_TMEM, TMEM_COL_START, D_V>` (line 22) takes
`D_V=448`. The body iterates `tile = 0 .. D_V/64-1` (= 0..6), loads
`B_H_TMEM=64` floats from `tmem_cols::O + tile * B_H_TMEM`, rescales, stores
back. That matches `tmem_cols::O = 0` and the kernel-wide convention that
`D_V` is laid out as 7 tiles of 64 columns of 64 rows. Good.

The epilogue at `phase1.cuh:254-264` then loads O with
`ku::tmem_ld_32dp32bNx<Kernel::B_H_TMEM>(Kernel::tmem_cols::O + tile*Kernel::B_H_TMEM, o_head)`
and writes to `plan.u.o.data()[h*Kernel::D_V + dv]`. The 7-tile SmemLayoutO
strides (with the swizzle atom) may not produce a contiguous `[B_H][D_V]`
view through `data()`. Verify that `data()` returns a pointer that lines up
with the `UMMA::Layout_K_SW128_Atom<bf16>` swizzle pattern of the bf16 O
output; for the 7-tile case the data() pointer is at the start of the
swizzled region, and the bf16 store at `h*D_V + dv` will collide with the
swizzle if the offset crosses a 128-byte swizzle boundary. The reference
kernel in `fwd/head64/phase1.cuh:300-352` uses `sO_addrs[i] = &sO(...)`
instead of `data()[...]` precisely to go through the swizzled layout.

**Fix:** Replace the manual offset with the proper smem layout, mirroring
the reference:

```cpp
Tensor sO = make_tensor(make_smem_ptr(plan.u.o.data()), SmemLayoutO{});
// inside the tile loop:
int dv = warp_idx*32 + lane_idx;
int tile = dv / 64;
CUTE_UNROLL for (int h = 0; h < B_H; ++h) {
    sO(h, dv) = bf16(o_head[h] * plan.head_scale[h]);
}
```

The TMA store loop at `phase1.cuh:280-289` already uses the smem layout; it
will work once `SmemLayoutO` is the 7-tile variant.

### P1 — `topk_length` and `is_k_valid` mask for partial topk tails

`phase1.cuh:51`:

```cpp
const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
```

`num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1)`. The validity
mask producer (`phase1.cuh:493-499`) calls `load_indices_and_generate_mask` with
the per-block absolute start `k*B_TOPK` and the global `topk_length`. So a
token with `topk_length=64` will produce valid mask 0xff for block 0 and 0x00
for block 1. The downstream `if (topk_length % B_TOPK != 0)` is automatically
handled by the mask. Good.

But the prologue TMA loads at `phase1.cuh:312`:

```cpp
if (k == 2) {
    plan.bar_prologue_utccp_nope.wait(0);
}
```

With `NUM_BUFS=3` and `k=2` matching `cur_buf=2`, the producer warp 4-7 is
checking that the UTCCP for K[2] (i.e. `q_nope`) has completed. The reference
kernel uses the same trick — `q_nope` (which is in SMEM) covers the bytes of
`k[2]` because the SMEM union is sized so that Q overlaps with the third K
buffer. The mxfp8 `SharedMemoryPlan` already has the same union structure
(`qkvo.q` is unioned with `qkvo.kv`), so the trick should work — but only if
the SMEM layout for `q_nope` covers the exact same byte range as
`k_nope[2]`. Verify the sizes line up:

- `q_nope`: `B_H*D_NOPE = 64*448 = 28672` bytes
- `k_nope[3]`: `3 * B_TOPK*D_NOPE = 3 * 128*448 = 172032` bytes

`q_nope` is much smaller than the union. So the producer's `k_nope[2]` data
will not collide with `q_nope` — but that means the original "since q_nope
coincidences with k[2]" justification is **not** true for this kernel, and
the `bar_prologue_utccp_nope.wait(0)` is irrelevant (no UTCCP wrote
to `k_nope[2]`). It is harmless (the wait will return immediately because
the UTCCP barrier was arrived in the prologue), but the comment is
misleading. Either:

- delete the `if (k == 2)` block and rely on `bar_kv_nope_ready[cur_buf].wait(...)`
  below to gate the MMA, or
- leave it in but fix the comment.

The simpler choice: remove the `if (k == 2)` block. The producer already does
`plan.bar_sv_done[cur_buf].wait((k/NUM_BUFS)&1^1)` (line 319) which
serialises the writes correctly across NUM_BUFS.

### P1 — `kk` declaration missing in the inner scale/exp loop

`phase1.cuh:206`:

```cpp
for (int k = 0; k < 32; ++k) {
    kk = k + g * 32;
```

`kk` is declared in the outer loop at `phase1.cuh:176`:

```cpp
for (int k = 0; k < 32; ++k) {
    int kk = k + 32 * g;
```

But the inner loop at line 205 **reuses** `kk` without re-declaring it. With
`g` iterated by the outer `for (int g = 0; g < NUM_QUANT_GROUPS; ++g)`, the
inner loop's `kk` is the outer-loop variable — which is out of scope. C++
will not compile this.

**Fix:** Re-declare `int kk = k + g * 32;` at the top of the inner loop at
`phase1.cuh:205`.

### P1 — The `kk` mask reads `valid_mask` in the inner `p_val` branch

`phase1.cuh:177`:

```cpp
float p_val = (valid_mask >> kk) & 1 ? plan.p_t[kk*B_H + h] : -CUDART_INF_F;
```

The mask is `uint64_t`, `kk` is `int`, and the shift amount is at most 127.
That is undefined behaviour in C++. Use `((valid_mask >> kk) & 1u) != 0u` or
extract the byte properly.

**Fix:** Replace the condition with a comparison.

### P1 — `bar_p_free` is unused

`phase1.cuh:422` does `plan.bar_p_free.wait(k&1^1);` but the prologue never
arrives on `bar_p_free` (it is init'd to 128 expectations at line 119 and
nothing increments it). The wait will deadlock or, in practice, will hang
the kernel.

Looking at the reference `fwd/head64/phase1.cuh:188`, `bar_p_free.arrive()`
is called by the scale/exp warp at the end of the `retrieve_mask_and_reduce_p`
helper. The mxfp8 kernel does its own P-loading at `phase1.cuh:154-160`
without ever arriving on `bar_p_free`. Either:

- call `plan.bar_p_free.arrive()` in the scale/exp warp at `phase1.cuh:162`
  (where `bar_k_valid_free` is currently arrived), or
- delete the `bar_p_free.wait(...)` in the MMA warp and the corresponding
  `bar_p_free.init(128)` / struct member.

**Fix:** Make the scale/exp warp arrive on `bar_p_free` once per block, after
`slot_bar_P_empty_arrival()` (the function passed as the helper is never
defined, so delete that call too). Note `phase1.cuh:157` calls
`slot_bar_P_empty_arrival();` but `slot_bar_P_empty_arrival` is not defined
anywhere — this is a build error.

**Fix (combined):** Replace the function-call placeholder at line 157 with
`plan.bar_p_free.arrive();` and add an `expect_tx` if the buffer also needs
an mbarrier to release the actual smem region. Verify that 128 threads
arrive exactly once per block (4 warps × 32 lanes).

### P1 — `bar_so_ready` is never signalled by the rescale step

`phase1.cuh:225` does `plan.bar_so_ready.arrive();` after writing S to
smem and rescaling O. The MMA warp waits on it at `phase1.cuh:476`. The
arrive should be a single thread, not 128. If the `bar_so_ready` is init'd
to 128 expectations, only one arrival would be needed per block (one
thread); if init'd to 1 expectation, 128 threads arriving would over-arrive
and the wait would be skipped, but the `cuda::memory::atomic_ref` overflow
might leave the barrier in a bad state.

**Fix:** Init `bar_so_ready` with `init(1)` and put the `arrive()` inside
`if (warp_idx == 0 && elect_one_sync()) { ... }`. Right now `init(128)` is
called at line 120 — change to 1, and gate the arrive with `elect_one_sync()`.

### P1 — `bar_prologue_q_scale` is not awaited on the consumer side

The producer at `phase1.cuh:98` issues
`ku::launch_tma_copy(tma_params.tma_Q_scale, gQ_scale, sQ_scale,
plan.bar_prologue_q_scale, ...)` — the TMA will arrive on
`bar_prologue_q_scale` automatically. The consumer at
`phase1.cuh:386-387` is:

```cpp
plan.bar_prologue_q_scale.arrive_and_expect_tx(B_H*(D_V/32)*sizeof(e4m3));
plan.bar_prologue_q_scale.wait(0);
```

The `arrive_and_expect_tx` here is wrong — this thread is the **consumer**
of the Q scale data, not the producer. The data arrives via the TMA. Drop
the `arrive_and_expect_tx` and just `wait(0)`.

The TX-byte estimate `B_H*(D_V/32)*sizeof(e4m3)` is also wrong: with
`MXFP8_SCALE_VEC_SIZE=32` and `D_V=448`, there are `D_V/32 = 14` scale
elements per row of Q, so the TX is `B_H*14*sizeof(e8m0) = 64*14*1 = 896`
bytes. The TMA copy issued by the producer should match. Verify the TMA
descriptor at `phase1.cuh:580-590` is set up with a box of `B_H × 14 × 1`
elements, which it is (line 572).

**Fix:** Remove the `arrive_and_expect_tx` call from the consumer.

### P1 — `bar_prologue_k_scale` is never signalled in the producer

`phase1.cuh:452` does `ku::umma_arrive_noelect(plan.bar_prologue_k_scale);`
— but no consumer ever waits on `bar_prologue_k_scale`. Instead, the MMA
warp at `phase1.cuh:435-451` builds `tK_scale` from the K-scale SMEM
region directly. So either:

- The wait is in the MMA warp at the right place (it currently is not
  present), or
- The umma_arrive is dead code and the MMA warp relies on the
  `bar_kv_nope_ready[cur_buf].wait(...)` to gate everything.

The current MMA warp builds `tK_scale` **after** `bar_kv_nope_ready.wait`,
which ensures that the K-NoPE data has arrived. The K-scale data is
written into `plan.qkvo.kv.kv_nope_scale[cur_buf]` by the producer (line
362) before the same producer arrives on `bar_kv_nope_ready` at line 454.
So the order is: scale data lands in SMEM → `bar_kv_scale_ready[cur_buf]
arrive()` (line 362) → `bar_kv_nope_ready.wait(...)` in MMA.

The `bar_prologue_k_scale` is therefore dead and should be removed — or
the MMA should `bar_prologue_k_scale.wait(0)` before building the
`tK_compact` (right after `bar_kv_scale_ready.wait` is implicit through
the ordering above).

**Fix:** Remove the dead `bar_prologue_utccp_k_scale` / `bar_prologue_k_scale`
arrive; the actual dataflow is `bar_kv_scale_ready` → `bar_kv_nope_ready`
→ MMA. Update the `SharedMemoryPlan` to drop the prologue barriers that
are not used and to keep only `bar_kv_scale_ready`.

### P1 — `bar_kv_scale_ready` is not in the MMA wait list

The producer (line 362) arrives on `bar_kv_scale_ready[cur_buf]` but the
MMA warp at `phase1.cuh:454` only waits on `bar_kv_nope_ready`. Without
an explicit `bar_kv_scale_ready.wait((k/NUM_BUFS)&1)` in the MMA warp
before reading `sK_scale`, there is a race: the TMA write to SMEM might
not have completed by the time the MMA reads it.

**Fix:** Add `plan.bar_kv_scale_ready[cur_buf].wait((k/NUM_BUFS)&1);`
before the `sK_scale` SMEM→TMEM UTCCP block in the MMA warp.

### P1 — `num_k_blocks` is `int`; loop bounds on TMEM columns

`num_k_blocks = max(ceil_div(topk_length, B_TOPK), 1)`. With
`B_TOPK=128` and `topk_length <= 4096` (typical DeepSeek prefill), this is
≤ 32. The TMEM columns (`O=0..127`, `P=400..463`) are reused per block. No
TMEM column is indexed by `k`, so no overflow risk. OK.

### P1 — `tP` and `tO` partition shape

`phase1.cuh:73-74`:

```cpp
Tensor tP = partition_fragment_C(tiled_mma_P, Shape<B_TOPK, Int<B_H>>{});
Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<D_V>, Int<B_H>>{});
```

For `TiledMMA_O` (D_V × B_H = 448 × 64), the partition needs a
constexpr shape. `Int<D_V>` should be a constexpr — `D_V` is `constexpr` in
the namespace (line 17). OK.

But the `tiled_mma_O` atom is `SM100_MMA_MXF8F6F4_SS_NOELECT<e4m3, e4m3, float, e8m0, D_V, B_H, ...>`.
The M-dim is `D_V=448`, which is `448/64=7` tiles along M. The MMA atom
requires M to be a power-of-two multiple of 16 — 448 is **not** a
power-of-two multiple of 16 (it is `7*64`, fine). Verify the cutlass
atom accepts `M=D_V=448`; if it only accepts M as 64/128/192/256, change to
`Shape<Int<D_V/2>, Int<B_H>>{}` and run two MMAs (the reference uses
this dual-gemm pattern). The reference `TiledMMA_O` in `fwd/head64/config.h:146`
uses M=256, suggesting the atom expects 64/128/192/256/512 along M.

**Fix:** Check the cutlass sm100 mxf8 atom's `M` parameter; if 448 is not
accepted, set `TiledMMA_O` to use M=224 (1SM shape limit) and run two
MMAs per O block, or change the M-dim of the O accumulator to 448 only if
the atom supports it. The
`flash_attention_4` style pattern uses 128×256 atom for O; we may need
`Shape<Int<256>, Int<B_H>>` and run the O accumulation in two
`256×64` passes (224 covers full D_V=448 with 224 = 7*32 not a power of
two; use 256 then mask or 192 + 256).

**Concrete recommendation:** keep the same M-dim selection as the
reference (`B_H × 256`), but view the O tile as `[256, 64]` and do two
MMAs per block (128×2). Specifically:

- Change `TiledMMA_O` to use M=256 (`B_H=64`, the N-dim of P) like the
  reference at `fwd/head64/config.h:145-147`.
- Update `tO` partition to `Shape<Int<256>, Int<B_H>>{}` (so two tiles of
  256 cover D_V=448 fully with 1 tile of 256 and one partial 192; for
  D_V=448 use two 256×64 MMAs and ignore the upper 64 columns of the
  second MMA).

This is a deeper change — defer to P2 unless `make_tiled_mma` fails at
compile time. The compile-time check will catch the wrong shape; if the
atom accepts 448, no change is needed.

### P2 — `bar_prologue_q_scale` is referenced as if it were the UTCCP barrier

Already covered in P0 / P1 above — the kernel conflates "TMA arrival" and
"UTCCP completion" barriers. The cleanest fix is to use one set of
`bar_prologue_*` for TMA, and a second set of `bar_prologue_utccp_*` for
UTCCP, and make all consumers / producers use the right one.

### P2 — `bar_k_valid_free[cur_buf].arrive()` count

`phase1.cuh:162` does `plan.bar_k_valid_free[k%NUM_BUFS].arrive();` once
per block, in the scale/exp warp. The barrier is init'd with
`init(128)` (line 117). 128 threads in the warpgroup will arrive, but the
code runs in 4 warps (each with 32 threads), all of which will execute
line 162 — so 128 arrivals per block. This matches the 128 expectations.
Good.

### P2 — `ku::tmem_ld_32dp32bNx` requires `B_H_TMEM` per row in TMEM

`tmem_ld_32dp32bNx<64>` loads 64 floats from one row in TMEM. The TMEM
atom is `32dp32b`, which is 32-wide. With B_H=64, we need two
consecutive 32-wide reads. Verify the helper handles the wider load
automatically (the reference `tmem_ld_32dp32bNx` is also used with
`B_TOPK/2=32` and `B_EPI=64`, so it must already work). OK.

### P2 — `CUTE_INVALID_CONTROL_PATH` vs the `__CLION_IDE__` guard

`phase1.cuh:540-541` is the `else` branch for non-SM100 architectures.
For IDE indexing the `#if` guard is true and the kernel is defined; for
real SM100 the kernel runs. OK.

### P2 — `k_rope` is `array_aligned<bf16, B_TOPK*D_ROPE>` but used as `SmemLayoutKRoPE` view

`SmemLayoutKRoPE` (config.h:101) is the swizzled UMMA layout. The smem
storage `kv_rope` is `array_aligned<bf16, B_TOPK*D_ROPE>` (i.e. 128*64 =
8192 bytes contiguous). The view is correct.

But the `k_rope` cp.async load at `phase1.cuh:510-534` uses
`ku::cp_async_cacheglobal<...>` with a per-row offset, which writes to
the smem directly. The strides used (`+ idx_in_group*8`, `+ local_row*NUM_GROUPS*32`)
need to match the swizzled layout's stride, not the contiguous stride.
Looking at the reference at `fwd/head64/phase1.cuh:545-571`, the same
pattern is used (without the swizzle) — for `SmemLayoutKRoPE =
UMMA::Layout_K_SW64_Atom<bf16>{}` and `B_TOPK=128`, a 64-byte swizzle
groups 32 bf16 elements per swizzle row. The cp.async writes 8 bf16
elements per group (`idx_in_group < 8`), so each thread writes 8 bf16
(16 bytes) per row — that is below the 32-byte cp.async size, which
underutilises the bandwidth. The reference does the same so this is
acceptable; keep it as-is.

## Plan to Apply the Fixes

### Step 1 — config.h changes

Edit `csrc/sm100/prefill/sparse/mxfp8_fwd/head64/config.h` to:

1. Add the missing constants:
   - `constexpr int B_H_TMEM = B_H;`
   - `constexpr float FP8_MAX = 448.0f;`
   - `constexpr int K_QUANT_GROUP_SIZE = 64;`
   - `constexpr int K_SCALE_BYTES = 8;` (7 used + 1 padding byte)
2. Add the `TmaParams` struct (mirroring `fwd/head64/config.h:17-22`,
   extended with `tma_Q_scale` and the RoPE/NoPE `tensor_map`s).
3. Add `SmemLayoutKNoPE_TiledMMA` and `SmemLayoutKRoPE_TiledMMA`
   (copy from `fwd/head64/config.h:92-102`).
4. Add `SmemLayoutOBuf_TMA = SmemLayoutOTiles<1>{}`.
5. Change `SmemLayoutO = SmemLayoutOTiles<8>` to `SmemLayoutOTiles<7>`.
6. Change `SmemLayoutKNoPE = SmemLayoutKTiles<8>` to `SmemLayoutKTiles<7>`.
7. Add `q_rope` to the `s_q_scale` union (so the kernel can find
   `plan.s_q_scale.q_rope`):
   ```cpp
   union {
       e4m3 s[B_H*B_TOPK];
       e8m0 q_scale[cosize_v<SmemLayoutQScale>];
       array_aligned<bf16, B_H*D_ROPE> q_rope;
   } s_q_scale;
   ```
8. Remove the dead `bar_prologue_utccp_q_scale` / `bar_prologue_utccp_k_scale`
   fields from `SharedMemoryPlan` (or rename to `bar_prologue_q_scale` /
   `bar_prologue_k_scale` if you prefer; the kernel uses both names). The
   cleanest is to keep the field name as `bar_prologue_utccp_q_scale` but
   fix the kernel references. Actually the field name is fine; only the
   *kernel* references the wrong name. See Step 2.

### Step 2 — phase1.cuh changes

Edit `csrc/sm100/prefill/sparse/mxfp8_fwd/head64/phase1.cuh` to:

1. Rename `run_fwd_phase1_kernel` → `run_mxfp8_fwd_phase1_kernel`
   (matches the header and the instantiation).
2. Hoist `tQ_scale` / `tK_scale` / `tS_scale` declarations to function
   top (before line 80), set their `data().get()` to the TMEM columns.
3. Replace the four `plan.s_q_rope.q_tail.q_scale(...)` /
   `plan.s_q_rope.s(...)` / `plan.s` references with
   `plan.s_q_scale.q_scale(...)` / `plan.s_q_scale.s(...)`.
4. Replace `plan.bar_prologue_utccp_rope.init(...)` →
   `plan.bar_prologue_utccp_q_scale.init(...)` (line 108).
5. Replace `plan.bar_prologue_utccp_nope.init(...)` →
   `plan.bar_prologue_utccp_k_scale.init(...)` (line 109).
6. Replace `plan.bar_prologue_q_scale.arrive_and_expect_tx(...)` (line 386)
   with just `plan.bar_prologue_utccp_q_scale.wait(0)`. The TMA hardware
   already arrived on `bar_prologue_q_scale`; the consumer just needs to
   wait.
7. Replace `plan.bar_prologue_k_scale.arrive_and_expect_tx(...)` (line 452
   expected location) with just the `umma_arrive_noelect`. The data is
   already in SMEM by virtue of the producer's `bar_kv_scale_ready`
   ordering.
8. Add the `bar_kv_scale_ready[cur_buf].wait((k/NUM_BUFS)&1)` in the MMA
   warp before building `tK_compact`.
9. Replace the `slot_bar_P_empty_arrival()` placeholder call (line 157)
   with `plan.bar_p_free.arrive();`.
10. Change `bar_so_ready.init(128)` → `init(1)` and gate the
    `arrive()` with `elect_one_sync()` and `warp_idx == 0` (line 120 and
    225).
11. Re-declare `int kk = k + g * 32;` at the top of the inner loop at
    line 205.
12. Fix the `valid_mask` shift expression at line 177 to use an
    unsigned compare.
13. Replace the `Tensor sQ_rope = make_tensor(make_smem_ptr(...))` and
    `Tensor sQ_nope = ...` `...` placeholders (lines 376, 382) with
    real smem pointers / layouts.
14. Replace the manual `plan.u.o.data()[h*D_V + dv] = ...` (line 262)
    with a swizzle-aware `sO(h, dv) = ...` write via `make_tensor(plan.u.o.data(), SmemLayoutO{})`.
15. Either remove `tS_scale` from the `utcmma_blockscaled_ss` call at
    line 481 and use the un-scaled `utcmma_ss`, or implement the
    S_Scale UTCCP in the scale/exp branch and add the
    `tS_scale.data().get() = tmem_cols::S_Scale;` at top level.
16. Remove the `if (k == 2) { plan.bar_prologue_utccp_nope.wait(0); }`
    block (line 312-314) — it is dead in the mxfp8 layout.
17. At the end of `run_mxfp8_fwd_phase1_kernel`, after all TMA
    descriptors, build the `TmaParams` struct, set the kernel attribute
    for max dynamic shared memory, and launch the kernel
    (`kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params, tma_params)`).
18. Drop the `//TODO: …` comments in `config.h:83, 92` and the
    `//TODO: How to wait for K scale ready in smem?` in `phase1.cuh:434`
    once the corresponding code is fixed.
19. Keep the `//TODO: change here to vectorized type conversion` comments
    at `phase1.cuh:203, 207, 209` — they are performance TODOs, not
    correctness. Document in the kernel that they are perf-only.

### Step 3 — verify the instantiation and call sites

`csrc/sm100/prefill/sparse/mxfp8_fwd/head64/instantiations/phase1_k512.cu`
already calls `run_mxfp8_fwd_phase1_kernel<512>` — no change needed.

`csrc/api/mxfp8_sparse_fwd.h` is the user-facing entry point; verify that
it calls `run_mxfp8_fwd_phase1_kernel<512>(params)` (or similar). If the
existing call uses a different name, update it to match the rename.

### Step 4 — `MmaParams` / `TiledMMA_O` M-dim sanity check

Compile with `nvcc -arch=sm_100 -c -std=c++17 …` and verify that
`SM100_MMA_MXF8F6F4_SS_NOELECT<e4m3, e4m3, float, e8m0, D_V=448, B_H=64,
UMMA::Major::MN, UMMA::Major::K>` is accepted. If not, fall back to
`TiledMMA_O = SM100_MMA_MXF8F6F4_SS_NOELECT<..., 256, 64, …>` and run
two MMAs per O block (256×64 + 192×64 with masking). This is the only
remaining open question that requires a build to resolve.

## Verification

1. Compile the new kernel with `nvcc -arch=sm_100` (or `sm_100a`).
   Confirm the renamed function links, the `TmaParams` struct compiles,
   and the SMEM `static_assert` (if added) shows `sizeof(SharedMemoryPlan)
   <= 228 * 1024` (SM100 smem cap).
2. Add a small test driver (`tests/test_mxfp8_dsa.py` or similar) that:
   - Allocates `Q`, `K`, `V` in MXFP8 + bf16 RoPE layout with known
     `topk_length`, `attn_sink`, and a deterministic set of `indices`.
   - Computes the reference attention output with PyTorch in FP32.
   - Computes the output of the kernel.
   - Compares the two to within `atol=2e-2, rtol=2e-2` (FP8 precision
     loss is expected).
3. Run the existing FlashMLA test suite
   (`tests/`) — verify the existing tests still pass.
4. Add an NCU profile (optional) to confirm the kernel is not blowing
   the SMEM or TMEM budget.

## Out of Scope (deferred)

- Vectorised FP8 store/load (the `//TODO: change here to vectorized`
  items in `phase1.cuh:203, 207, 209`).
- A `Kernel` template alias (the code uses `Kernel::B_H`, `Kernel::B_H_TMEM`,
  `Kernel::D_V`, `Kernel::NUM_BUFS`, `Kernel::NamedBarriers`,
  `Kernel::tmem_cols`, `Kernel::SmemLayoutO` — all of these can be
  replaced with the file-scope `B_H`, `B_H_TMEM`, `D_V`, `NUM_BUFS`,
  `NamedBarriers`, `tmem_cols`, `SmemLayoutO` constants once they exist).
  Adding the `Kernel` alias is a small refactor — do it after the
  correctness build passes.
- Performance tuning of the `p_t` transpose (`// How to improve the
  transpose efficiency here?` comment in `phase1.cuh:160`).
- Native block-scaled SV MMA (requires producing `tS_scale` via UTCCP
  from the scale/exp warp). The current un-scaled fallback is
  numerically correct but slower.
