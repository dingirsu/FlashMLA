# Dual MXFP8 Head64 Prefill 输入布局

本文描述 `dual_mxfp8/head64` prefill kernel 的 Q/K 输入、量化分组和
UE8M0 scale slot 布局。当前实现不区分 NoPE 和 RoPE，Q/K head dimension
均为 512。

## 数值格式

- Q 和 K 数据使用 E4M3。
- Q 和 K 每个逻辑量化 group 共用一个 UE8M0 scale。
- 每个逻辑 group 包含 64 个 E4M3 元素。
- tcgen05 MXFP8 MMA 的 scale vector 是 32 个元素。K 通过显式选择
  `b_sf_id`，让连续两条 K=32 MMA 使用同一个 K=64 scale。
- S 在转换为 E4M3 前乘以对应 K token 的 `k_sf0`，S 的 TMEM UE8M0
  scale 固定为 1。V 使用 `w1/w2` 提供的八个 64-D UE8M0 scale。

量化可表示为：

```text
scale_g = UE8M0(2 ^ ceil(log2(max(abs(x_g)) / 448)))
q_g     = E4M3(clamp(x_g / scale_g, -448, 448))
```

## Q 分组

每个 Q head 有 512 个元素。Q 的一个 64-element group 由两个相距
256 的连续 32-element chunk 组成：

| Scale | Q dimensions |
| --- | --- |
| `q_sf0` | `[0, 31]` 和 `[256, 287]` |
| `q_sf1` | `[32, 63]` 和 `[288, 319]` |
| `q_sf2` | `[64, 95]` 和 `[320, 351]` |
| `q_sf3` | `[96, 127]` 和 `[352, 383]` |
| `q_sf4` | `[128, 159]` 和 `[384, 415]` |
| `q_sf5` | `[160, 191]` 和 `[416, 447]` |
| `q_sf6` | `[192, 223]` 和 `[448, 479]` |
| `q_sf7` | `[224, 255]` 和 `[480, 511]` |

Q 的每个 head 是一个连续的 528B record：

```text
[512B E4M3 data]
[q_sf0 ... q_sf7, q_sf0 ... q_sf7]
```

因此 Q tensor 的接口形状为 `[s_q, 64, 528]`，head stride 必须是
528B，record 和 scale slot 必须至少 16B 对齐。

## K 分组

每个 K token 有 512 个元素。K 使用连续的 64-element groups：

| Scale | K dimensions |
| --- | --- |
| `k_sf0` | `[0, 63]` |
| `k_sf1` | `[64, 127]` |
| `k_sf2` | `[128, 191]` |
| `k_sf3` | `[192, 255]` |
| `k_sf4` | `[256, 319]` |
| `k_sf5` | `[320, 383]` |
| `k_sf6` | `[384, 447]` |
| `k_sf7` | `[448, 511]` |

每个 K token 的 scale 在 GMEM 中已经复制为一个 16B slot：

```text
[k_sf0 ... k_sf7, k_sf0 ... k_sf7]
```

16B slot 满足当前 TMA tensor map 的 inner-box 和对齐要求。当前实现让
K-scale gather4 直接使用原始 token index，每次收集四个完整的 16B slot
到 padded raw buffer，再由 warp10/warp11 做 pair-row 重排：

```text
raw row 0..15: four complete 16B token slots + 64B padding
warp10: final source rows 0..31
warp11: final source rows 32..63
```

每个 raw gather destination 使用 128B row stride，保证 TMA destination
对齐；raw map 使用 `SWIZZLE_NONE`，让 warp10/warp11 可以用普通 SMEM
load 读取 slot。重排完成后，CTA0 使用 CUTLASS 的
`SM100_UTCCP_2x64dp128bitlw0123_2cta` 写入 K-scale TMEM。

定义编译宏 `DUAL_MXFP8_K_SCALE_CP_ASYNC=1` 后，warp10/warp11 不再等待
raw gather buffer，也不执行 SMEM 到 SMEM 重排。每个 lane 负责一个
token，用四条 `cp.async.ca.4B` 把 GMEM 中 replicated 16B slot 的四个
word 直接写入最终 `k_scale_mma` 布局。未定义该宏时仍使用上述 TMA
gather4 加 SMEM 重排路径。测试脚本可通过下面的环境变量构建
cp.async 版本：

```bash
DUAL_MXFP8_K_SCALE_CP_ASYNC=1 ./compile_sm100.sh dual_mxfp8
```

此前直接把两个完整 slot slab 写到 `tmem_col + 0/+4` 的路径能够编译和
运行，但真实 K scale 数值错误。16B replication 只解决了 GMEM/TMA
搬运粒度，不能替代 `ScaleFactorDuplicated2by2` 要求的跨 token 物理
转置。失败原因和当前修复见下文。

## Prefill KV Page 布局

当前 prefill 接口把整个 KV allocation 视为一个 page。物理存储是两个
连续 plane，而不是逐 token 交错：

```text
page_base
  K_data[0]          512B E4M3
  K_data[1]          512B E4M3
  ...
  K_data[s_kv - 1]   512B E4M3
  K_scale[0]          16B UE8M0 slot
  K_scale[1]          16B UE8M0 slot
  ...
  K_scale[s_kv - 1]   16B UE8M0 slot
```

接口 tensor 形状为 `[s_kv, 1, 528]`，但这个 528B 只表示每个 token
在总 allocation 中占用的平均 envelope。kernel 分别以 512B data stride
和 16B scale stride 建立两个 TMA tensor map。

## K Scale 的 MMA 选择

K=512 被拆成 16 条 K=32 MMA。K 的 scale word 和 `b_sf_id` 选择为：

```text
MMA k:    0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15
b_sf_id:  0 0 1 1 2 2 3 3 | 0 0  1  1  2  2  3  3
word:     0 0 0 0 0 0 0 0 | 1 1  1  1  1  1  1  1
```

这样 TMEM 中只需要保存 `[k_sf0..k_sf3]` 和 `[k_sf4..k_sf7]`，不需要
将每个 K=64 scale 在 SMEM 中复制成两个 K=32 scale。

## S/V 的 Token Scale

`k_sf0`（维度 `[0, 63]` 的第一个 K scale）同时作为该 token 的 V
token scale。warp10/warp11 在把每个 raw 16B slot 重排到 K-scale MMA
SMEM 的同时，取 slot 的第 0 个 byte 写入
`v_token_scale[NUM_K_BUFS][B_TOPK]`。该数组按 K ring buffer 分 stage，
因此 softmax worker 在等待对应 `bar_QK_done` 后可以安全读取。

对每个 `S[h, t]`，kernel 执行：

```text
s_for_sv[h, t] = exp2(logit[h, t] - new_max[h]) * v_token_scale[t]
S_e4m3[h, t]   = E4M3(s_for_sv[h, t])
```

`li` 仍使用未乘 token scale 的 softmax 值累加；只有写入 S 并参与
`S@V` 的 E4M3 数据乘以 token scale。S 的 MMA scale factor 为全 1。
每个 token scale 会广播到该 token 对应的全部 `H=64` 个 S 行。

## V 的 Dimension Scale

接口新增两个 float 参数 `w1` 和 `w2`。它们不是数值意义上的 FP32
scale，而是分别用原始 32-bit float bit pattern 打包四个 UE8M0 byte：

```text
w1 bits = [v_sf0, v_sf1, v_sf2, v_sf3]
w2 bits = [v_sf4, v_sf5, v_sf6, v_sf7]
```

列表顺序是从 float raw bits 的最低 byte 到最高 byte。

第 `i` 个 scale 作用于 V 的维度 `[64*i, 64*i+63]`。逻辑 V scale
矩阵形状是 `[TOPK/32, D] = [2, 512]`，两个 TOPK/32 行使用相同的
dimension scale：

```text
V_scale[:, 64*i : 64*(i+1)] = v_sfi
```

CTA0 使用 `w1`，CTA1 使用 `w2`。每个 float 的 raw bits 已经是四个
UE8M0 组成的 32-bit word；四个 warp 分别覆盖四个 32-DP subpartition，
用 `tcgen05.st` 将 packed word 直接写入 V-scale 的四个 TMEM column。
这条路径不使用 SMEM，也不经过 UTCCP。

## 为什么 direct gather4 到最终 SMEM 布局不成立

这里讨论的 direct path（当前代码已替换为 raw gather + warp 重排）是：
每个 K token 在 GMEM 中保存一个已经复制
到 16B 的 scale slot，TMA gather4 将完整 slot 直接写入所谓的最终
SMEM slab，然后 UTCCP 不经 warp/register 重排直接写 TMEM。问题不在
16B transaction 本身，而在“每个 token 连续”与 MMA 所需
“跨 token 交错”是两种不同的布局。

### GMEM slot 和 UTCCP source row 不是同一种布局

设 token `t` 的八个 K=64 scale 为 `t.s0 ... t.s7`。GMEM 中一个 slot
按 token 连续保存：

```text
token t 的 16B slot

word 0          word 1          word 2          word 3
[t.s0..t.s3]    [t.s4..t.s7]    [t.s0..t.s3]    [t.s4..t.s7]
```

其中每个 `word` 是 4B。后 8B 是同一 token 内部的 replication，不是
另一个 token 的 scale。

CUTLASS 的 SFB SMEM 基本布局是 `(32, 4):(16, 4)`：逻辑 row 先按
`row % 32` 分到 32 个物理 source row，再把相隔 32 的逻辑 row 放到
同一个 16B source row 的下一个 4B word。对于本 kernel 的 64 个选中
K token，并结合 dual map 所需的物理重复，32 行 source row 的两个
K=256 half 必须分别组织为：

```text
UTCCP source row r, r in [0, 31]  (K groups 0..3)

word 0          word 1              word 2          word 3
[r.s0..r.s3]    [(r+32).s0..s3]     [r.s0..r.s3]    [(r+32).s0..s3]

UTCCP source row r+32, r in [0, 31]  (K groups 4..7)

word 0          word 1              word 2          word 3
[r.s4..r.s7]    [(r+32).s4..s7]     [r.s4..r.s7]    [(r+32).s4..s7]
```

也就是说，`r` 和 `r+32` 必须在进入 UTCCP 之前就被合并到同一个 16B
source row；同一个 K half 的第二组 word 是 duplicated2by2 的物理
重复。`SM100_UTCCP_2x64dp128bitlw0123_2cta` 会完成它定义的 DP 广播，
但不会把两个独立的 source row 再做一次上述转置。

### `ScaleFactorDuplicated2by2` 的关键地址关系

M=128 的 2CTA MXFP8 MMA 选择
`UMMA::TmemAllocMode::ScaleFactorDuplicated2by2`。CUTLASS 的
`tmem_sf_frg` 对 SFB 使用的核心 stride 是：

```text
Stride<Stride<Stride<1, 512>, 64, 32>, Stride<0, 128>>
```

这个布局同时编码了 32-DP 子分区、2x2 data path 的重复和 scale word
选择。对当前需要的逻辑 SFB row 而言：

- token `r` 和 token `r+32` 的 scale word 位于同一相关 DP 的不同
  32-bit word 中；
- duplicated physical half 位于 `DP + 64`；
- UTCCP 的 `2x64dp128bitlw0123` atom 从每个 source row 读取 128 bit，并按
  atom 规则写入/广播到这些物理位置。

因此，“逻辑 byte offset 为 4B”不能写成“TMEM base column 加 4”。前者
表示同一 DP 内的下一个 32-bit word；后者把整个 TMEM 起始地址推进了
四个 column，是另一个 scale tile 位置。

当前 direct path 的第二次 copy 是：

```cpp
SM100_UTCCP_4x32dp128bit_2cta::copy(desc, tmem_col + 4);
```

它把 token `32..63` 的完整 slot 放到 column `+4..+7`，并没有填入
token `0..31` 所在 atom 的 `word 1/word 3`。所以 MMA 在读取 token
`r+32` 的 scale 时会看到 token `r` slot 中的其他 scale 或 replication，
而不是 `r+32` 的 scale。所有 scale 都等于 1 时该错误会被完全掩盖。

### 为什么 TMA swizzle 不能顺便完成这个转置

当前 K-scale tensor map 的逻辑 box 是 `[16, 1]`，gather4 的每个 index
选择一个 token，结果仍是四个连续的 16B slot。`SWIZZLE_64B` 改变的是
SMEM atom 内的地址/Bank 映射，不会把下面的 token-major 矩阵：

```text
token r:      [A_r, B_r, A_r, B_r]
token r+32:   [A_s, B_s, A_s, B_s]
```

自动转成：

```text
pair row r:   [A_r, A_s, B_r, B_s]
```

gather4 可以选择四个离散 token index，但每个被选择的 box 仍作为一个
连续整体写出；它不是 4B-word scatter，也不能让一个 16B destination
row 的四个 word 分别来自两个 gather index。要让一次 16B gather 直接
成功，GMEM 自身就必须预先保存跨 token 的 pair-row 布局，而当前约定是
每个 token 自包含的 `[s0..s7, s0..s7]`。

### 已尝试但不能绕过该限制的办法

下表中的“报错”是本 kernel bring-up 时在当前 tensor map、swizzle 和
2CTA copy 配置下的实测结果，不应理解为对所有 TMA 用法的普遍限制。

| 尝试 | 目的 | 实际结果 / 原因 |
| --- | --- | --- |
| 两个完整 16B slab，分别 copy 到 `tmem_col + 0` 和 `+4` | 用第二个 slab 表示 token `32..63` | 能运行，但 `+4` 是四个 TMEM columns，不是同一 atom 内的第二个 4B word；真实 scale 数值错误。 |
| 把第二组 gather destination 放到 SMEM `+64B` | 利用 64B swizzle atom 的另一半交错写入 | 运行时报告 misaligned shared/local address；当前 gather4 destination 必须保持 128B 对齐。 |
| `UINT8` tensor map 使用 4B inner box | 每次只 gather 一个 4B scale word，再分别写到 word 0/1/2/3 | tensor map 创建失败，当前配置不接受该 4B box。 |
| `UINT32` tensor map 使用 `box_size={1,1}` | 用一个 `uint32_t` 表示一个 scale word | `cuTensorMapEncodeTiled` 仍拒绝该 map。 |
| 对 16B box 使用 `col_idx=4` | 从 token slot 的第二个 4B word 开始搬运 | 运行时命中 CUTLASS/device error；它也会使完整 16B box 跨出当前 slot，而不是选择一个独立 4B box。 |
| 调整 UTCCP descriptor `base_offset` | 在 source row 内选择 4B word | 该字段按 16B descriptor atom 工作，不能充当 4B word selector。 |
| 使用 `tmem_col - 1` 等非标准 TMEM 起点补偿 | 试图让第二次 copy 落到期望 word | 触发 misaligned-address，且没有改变 copy atom 的布局语义。 |

### 数值证据

旧 direct slab 实现有两个很有区分度的结果：

- K scale 全部设为 1 时测试通过。这说明 K E4M3 数据路径、gather 的 token
  index、barrier 和 QK MMA 的基本时序不是当前误差来源。
- 使用真实的逐 token K scale 后，单 tile 测试有 `112/128` 个元素不匹配，
  最大绝对误差约为 `0.6018`。因此剩余问题与 scale 的物理布局/别名一致，
  不能用“kernel 能运行”作为 direct layout 正确的证明。

当前 warp10/warp11 重排实现已经通过 `scripts/run_dual_mxfp8.py` 的三组
真实 Q/K scale 测试：one-tile、two-tile-pairs 和 five-tile-wrap。QK/LSE
均在测试阈值内，说明 raw gather、pair-row 重排和显式 `b_sf_id` 选择
现在是一致的。

### 可行方案

在保持当前 GMEM ABI 的前提下，kernel 现在采用显式 SMEM 重排：TMA
gather4 先得到每 token 16B slot，warp10/warp11 再将 token `r` 与
`r+32` 拼成 CUTLASS 所需的 post-transpose source row，最后执行
`SM100_UTCCP_2x64dp128bitlw0123_2cta`。这一步不是多余 copy，而是在
改变布局语义。由于 MMA 显式选择 `b_sf_id` 让两条 K=32 指令复用一个
K=64 scale，重排时每个 32-bit word 保留四个独立 scale byte，不能再把
每个 scale byte 复制两次。

另一条路线是重构 QK 的 MMA tiling，例如拆成 N=64 子 tile，或选择其他
不要求当前 `r/r+32` 交错的物理 SFB layout。这会改变 MMA 调度、TMEM
分配和 dual-map 映射，必须重新推导并用真实 scale 验证，不能只改 copy
offset。

如果允许修改 GMEM ABI，也可以让生产端直接存储 pair-row：每个 16B
slot 同时包含 token `r` 和 `r+32` 的 scale word。这样 gather4 才能直接
搬到最终 SMEM 布局，但它已经不再是“每 token 一个自包含 16B slot”。

## 布局依据和实验边界

以下结论直接来自当前 vendored CUTLASS 的布局定义：

- [`copy_traits_sm100.hpp`](../../../../../cutlass/include/cute/atom/copy_traits_sm100.hpp)：
  `SM100_UTCCP_4x32dp128bit_2cta` 和
  `SM100_UTCCP_2x64dp128bitlw0123_2cta` 的 source/destination layout；
- [`mma_traits_sm100.hpp`](../../../../../cutlass/include/cute/atom/mma_traits_sm100.hpp)：
  `ScaleFactorDuplicated2by2` 和 `tmem_sf_frg` 的 2x2 SFB stride；
- [`sm100_blockscaled_layout.hpp`](../../../../../cutlass/include/cutlass/detail/sm100_blockscaled_layout.hpp)：
  `deduce_smem_layoutSFB` 的 `(32, 4):(16, 4)` post-transpose SMEM layout。

两个对照实现也都假定 scale 在 UTCCP 前已经是 post-transpose SMEM
布局，而不是让 UTCCP 把 token-major slot 转置：

- CUTLASS example：
  `../gemm/blackwell/mxfp8/3_tcgen05_mxfp8_2cta_256n_overlap128_tma_store/`
  中的 SFA/SFB SMEM-to-TMEM copy 和 SF ID rotation；
- DeepGEMM：
  `../DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`
  中的 `utccp_required_smem_warp_transpose`、UTCCP copy 和 runtime SF ID
  选择。

对齐错误、tensor-map 创建失败和真实 scale 的误差数字属于当前实现的
实验结果。它们用于界定这条 direct path 为什么失败；如果 tensor map、
GMEM ABI 或 MMA tile 发生变化，需要重新验证这些实验结论。
