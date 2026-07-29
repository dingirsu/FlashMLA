# MXFP8 x MXFP4 QK Problem

Status: unresolved. This file records the reproducible failure and the
observations made while changing the SM100 head64 sparse prefill kernel from
MXFP8 x MXFP8 to MXFP8 x MXFP4. No claim is made that the current QK MMA path
is correct.

Date of the last run: 2026-07-29

## Scope

The affected kernel is:

`csrc/sm100/prefill/sparse/mxfp8_fwd/head64/phase1.cuh`

The intended format is:

- Q data: MXFP8 E4M3, 512 values per token, 16 UE8M0 scales, group size 32.
- KV data: packed E2M1, 512 logical values (256 global bytes) per token.
- KV page-tail scales: 16 UE8M0 values per token, group size 32.
- KV global envelope: 272 bytes per token (`256` data bytes followed by the
  page-tail scale region).
- KV shared-memory footprint after `16U4_ALIGN16B` TMA expansion: 512 bytes
  per token.

The compact TMEM placement is intentionally retained as requested:

```text
Q_Scale = 256
K_Scale = 260
S_Scale = 264
P       = 268
V_Scale = 332
```

The source currently has the mixed-FP4 MMA type and the custom MMA-shaped
shared layouts in `config.h:115-149`, and the custom block-scaled helper in
`phase1.cuh:107-155`.

## Test environment

All GPU runs used physical GPU 5 only:

```text
CUDA_VISIBLE_DEVICES=5
GPU: NVIDIA B300 SXM6 AC
```

The extension compiled successfully with CUDA 13.0.3 using:

```bash
env CUDA_HOME=/share/apps/cuda/13.0.3 \
  PATH=/share/apps/cuda/13.0.3/bin:$PATH \
  LD_LIBRARY_PATH=/share/apps/cuda/13.0.3/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  python3 setup.py build_ext --inplace
```

## Controlled all-one experiments

The control encoding was:

```text
Q data       = 0x38 (E4M3 value 1)
K packed data= 0x22 (two E2M1 values 1 per byte)
Q scale      = 0x7f (UE8M0 value 1)
K scale      = 0x7f (UE8M0 value 1)
kv_scale_w   = 0x7f (UE8M0 value 1)
```

With all four quantities set to one, every QK dot product must be exactly
`512`. The kernel instead returned:

```text
expected QK = 512
kernel QK   = 2361183241434822606848
             = 512 * 2^62
```

The same result was obtained after a temporary A/B build that added a
`tcgen05.commit` plus mbarrier wait after the Q/V prologue scale copies and
after each K scale copy. That experiment did not fix QK, so those temporary
barrier fields were removed from the source rather than kept as an unverified
change.

The four single-variable controls were also run against random values for the
other inputs. The expected values below come from the PyTorch reference after
unpacking the exact packed buffers; `kernel max h0` is the first head's
returned QK maximum.

| Control | Expected QK sample / max | Kernel max h0 | Result |
| --- | ---: | ---: | --- |
| Q data forced to one | sample `0.037109375`, max `0.37890625` | `inf` | fail |
| K data forced to one | sample `61.5279541`, max `123.055908` | `50379556` | fail |
| Q scale forced to one | sample `-373.875`, max `7009.84375` | `inf` | fail |
| K scale and `kv_scale_w` forced to one | sample `22.9453125`, max `52.2314453` | `inf` | fail |

Therefore changing one global data or scale source at a time does not make the
QK path correct. It also rules out a simple omission of only one of the four
global loads.

## Scale sensitivity sweep

To check whether the MMA actually consumes the written scale values, Q and K
data were both kept at one while the UE8M0 scale bytes were changed. The
expected dot is `512 * q_scale * k_scale`.

| Q bits | K bits | Expected | Kernel |
| ---: | ---: | ---: | ---: |
| `0x7f` | `0x7f` | `512` | `2.36118324e21` |
| `0x7e` | `0x7f` | `256` | `192` |
| `0x80` | `0x7f` | `1024` | `1.18059162e21` |
| `0x7f` | `0x7e` | `256` | `5.90295810e20` |
| `0x7f` | `0x80` | `1024` | `768` |
| `0x7e` | `0x7e` | `128` | `inf` |

The output reacts to the scale bytes, but not with the expected multiplicative
factor. This is stronger evidence for a wrong scale-fragment/descriptor
mapping (or a wrong subset/order of scale factors) than for a missing global
load.

## `cute::print` observations

The kernel diagnostics are all `cute::print`; no `printf` diagnostics were
used. Representative output from the all-one run:

```text
Q data g/s h0 d0/64/128/256=38/38,38/38,38/38,38/38
Q h0 g0/4/8/12 raw=7f,7f,7f,7f mma=7f,7f,7f,7f
K FP4 g/s row0 d0/64/128/256=2/2,2/2,2/2,2/2
K scale g/s row0 0/1/4/7=7f/7f,7f/7f,7f/7f,7f/7f
Q scale TMEM row0 cols0/1/2/3=7f7f7f7f,7f7f7f7f,00000000,00000000
K scale TMEM row0 cols0/1/2/3=7f7f7f7f,7f7f7f7f,7f7f7f7f,7f7f7f7f
QK idesc=08900280 A(fmt/major)=5/0 B(fmt/major)=0/0 SF=1
```

The descriptor means that the QK instruction is configured as E2M1 A,
E4M3 B, K-major for both operands, with UE8M0 scale input. The first sixteen
descriptor/scale addresses printed by the kernel show the expected two-bit
scale-ID changes, for example:

```text
Kdesc=4000404000010840 Qdesc=4000404000010040 Ksf=00000104 Qsf=00000100
Kdesc=4000404000010842 Qdesc=4000404000010042 Ksf=40000104 Qsf=40000100
Kdesc=4000404000010844 Qdesc=4000404000010044 Ksf=80000104 Qsf=80000100
Kdesc=4000404000010846 Qdesc=4000404000010046 Ksf=c0000104 Qsf=c0000100
```

These reads prove that data and some TMEM locations contain the expected
bytes after the copy, but they do not prove that each MMA instruction uses the
same scale element at issue time. The scale sweep demonstrates that the
current mapping is nevertheless numerically wrong.

## What is already ruled out or made less likely

1. The global Q bytes are visible in shared memory with the expected values.
2. The packed FP4 global bytes and the TMA-expanded shared bytes are visible
   with the expected low nibbles. The zero bytes sampled between packed chunks
   are the expected 16U4_ALIGN16B padding, not logical K values.
3. The page-tail K scale bytes are visible in shared memory with the expected
   values.
4. The Q and K MMA descriptor formats/majors are not accidentally left as the
   old FP8/FP8 descriptor.
5. The failure persists when all data and all scales are constants, so random
   quantization or invalid-index masking is not needed to reproduce it.
6. A simple asynchronous-UTCCP completion race was tested with explicit
   mbarrier commits and did not change the all-one QK result.

## Most likely remaining fault

The remaining high-risk code is the QK-specific hierarchy and scale selection:

```text
SmemLayoutK_TiledMMA
SmemLayoutQ_TiledMMA
utcmma_blockscaled_ss_mma_layout()
```

The helper obtains `TiledMMA::make_fragment_A/B()` and walks nested K
coordinates with `idx2crd()`, then indexes the nested TMEM scale fragments with
the corresponding coordinate. This was introduced because the generic
`ku::utcmma_blockscaled_ss()` partitions an already MMA-shaped FP4 tensor a
second time. The scale sweep indicates that this custom correspondence is not
yet equivalent to the canonical CUTLASS block-scaled MMA correspondence.

The next investigation should be a minimal standalone 128 x 64 x 32 (then
512-K) SS MMA with all-one shared operands and explicitly initialized TMEM
scales. Compare, one instruction at a time:

1. `partition_fragment_A/B()` on the canonical FP4 shared layout.
2. `make_fragment_A/B()` on the MMA-shaped layout used here.
3. The exact nested coordinate used for the A/B scale fragment.
4. The resulting QK accumulator before any softmax or quantization.

The official CUTLASS `float_e2m1_unpacksmem_t` layout rules must be checked in
that minimal test; the physical-byte diagnostic layout (`uint8_t`) and the
logical unpacked-FP4 MMA layout must not be conflated.

Do not move the compact TMEM scale columns as part of this investigation; that
placement is an intentional constraint of this kernel.

## Relevant files and references

- `csrc/sm100/prefill/sparse/mxfp8_fwd/head64/config.h`
- `csrc/sm100/prefill/sparse/mxfp8_fwd/head64/phase1.cuh`
- `tests/mxfp8_test_utils.py` (packed E2M1/page-tail reference)
- `tests/test_mxfp8_sparse_prefill_head64.py` (32-wide tiled PyTorch oracle)
- `csrc/sm100/decode/mxfp8_head64/kernel.cuh` (working UTCCP barrier pattern)
- `csrc/cutlass/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf8_mxf4_f16_bf16_nt_layout.cu`
- `.agents/skills/KernelWiki/wiki/hardware/nvfp4.md`
- `.agents/skills/KernelWiki/wiki/kernels/nvfp4-gemm.md`
- `.agents/skills/KernelWiki/wiki/kernels/deepgemm.md`

## Current verification status

- Extension build: passed.
- Controlled GPU runs on GPU 5: reproduced the QK failure consistently.
- Full precision pytest: not passing because the first QK/max-logit stage is
  still wrong; later softmax/SV results are not meaningful until QK is fixed.
- `MXFP8_PREFILL_DEBUG_VALUES` is currently enabled in `config.h` for the
  diagnostics. Restore it to `0` (and leave markers disabled) before a final
  production build.

