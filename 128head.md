# `head128/phase1.cuh`：SM100 Sparse Prefill 128-head kernel 设计拆解

本文分析的对象是 `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh`，以及它的
`config.h`、TMA/UMMA 封装和测试参考实现。它不是 Top-K indexer；`indices` 已经由上游
准备好。这个 kernel 的职责是对每个 query position，在 128 个 Q head 上完成稀疏 attention：

\[
P_{h,j}=Q_h K_{\mathrm{indices}[j]}^T,\qquad
O_h=\sum_j \operatorname{softmax}(P_h\cdot \mathrm{sm\_scale})_j V_{\mathrm{indices}[j]}.
\]

这里的 ``head128'' 指的是一次处理 `h_q == 128` 个 query head，不是 head dimension。
支持的 Q/K dimension 是 512 或 576，而 V dimension 固定为 512。

> 结论先说：这是一个以 **一个 query position / 一个 2-CTA cluster** 为基本工作单元的
> Blackwell UMMA kernel。它把 128 个 head 和一个 128-token Top-K tile 分别拆到两个 CTA，
> 用 TMEM 同时保存 `O`、`P` 和 Q 的 384 维尾段；再以 K 的两个 K-dimension 分段和 V 的两个
> token 分段构造细粒度的 producer/consumer pipeline。设计的核心不是“多做几个 GEMM”，而是
> 用 mbarrier 精确释放每一段 SMEM/TMEM，避免为整个 K/V tile 做传统的双缓冲。

## 1. 适用范围与调度入口

### 1.1 运行时契约

`run_fwd_phase1_kernel` 的断言位于
`csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:623`：

| 条件 | 原因 |
| --- | --- |
| `h_q == 128` | 整个 cluster 的 M 维就是 128 个 query head。 |
| `h_kv == 1` | 本实现只处理一个 KV head。 |
| `d_qk == 512 || d_qk == 576` | `D_sQ = D_QK - 384` 必须是 64 的倍数。 |
| `d_v == 512` | O 的 TMEM 和 SV layout 固定为 512 个输出 channel。 |
| `topk % 128 == 0` | 主路径不为 `params.topk` 的尾块做边界判断。 |

公开接口在 `csrc/api/sparse_fwd.h:262` 选择它。一个容易漏掉的分流是：当
`d_qk == 512 && topk <= 1280` 时，接口优先选择
`fwd_for_small_topk/head128`，因此本文件主要服务较大的 Top-K，或 `d_qk == 576` 的 128-head
prefill。

参考实现 `tests/ref.py:27`--`52` 给出的语义也很关键：K 用完整的 `d_qk` 参与 QK，但 V 只取
`gathered_kv[..., :d_v]`。所以在 `d_qk == 576` 时，最后 64 维只参与 score，不参与输出 V。

### 1.2 grid 和 cluster 的含义

启动参数在 `phase1.cuh:694`--`703`：

```cpp
dim3(2 * params.s_q, 1, 1),   // grid
dim3(Kernel::NUM_THREADS, 1, 1),
dim3(2, 1, 1),                // cluster
```

因此：

```text
grid.x = 2 * s_q

cluster for query position q:
  block 2*q + 0  -> CTA0, cta_idx = 0
  block 2*q + 1  -> CTA1, cta_idx = 1
```

`s_q_idx = blockIdx.x / 2`、`cta_idx = blockIdx.x % 2` 位于
`phase1.cuh:70`--`71`。两个 CTA 是同一个 cluster 中的协作成员，不是两个独立的 attention
block。它们各自有本地 SMEM 和本地 barrier 实体，但会通过 `shared::cluster` 远程到达、TMA
transaction accounting 以及 `tcgen05.mma.cta_group::2` 组成一次逻辑上的联合运算。

不要把 KernelWiki 中常见的“2-SM 是 `m256n256`”示例直接套到这里。本 kernel 的自定义 CuTe
atom 在 `config.h:120`--`132` 明确以 `M=B_H=128` 构造，且后续
`partition_fragment_C(..., Shape<64, ...>)` 说明每个 CTA 的本地 C fragment 是 64 个 head。
通用 2-SM 机制是架构背景；本 kernel 的精确逻辑 tile 和分布必须以本仓库的 CuTe trait 为准。

## 2. 两个 CTA 分别拥有哪一半数据

同一个 `k` 代表 128 个选择位置，即 `B_TOPK = 128`。CTA 的切分方式在 QK 与 SV 中不同，这是
这个 kernel 最容易被误读的地方。

| 对象 | CTA0 | CTA1 | 证据 |
| --- | --- | --- | --- |
| Q / 输出 head | 本地 head `0:64` | 本地 head `64:128` | Q TMA 的 `flat_divide(...)(_, cta_idx, _)`，`phase1.cuh:135`--`139`。 |
| QK 的 K token 供给 | 本 tile indices `0:64` | indices `64:128` | K producer 加 `cta_idx * (B_TOPK/2)`，`phase1.cuh:401`。 |
| SV 的 V channel 供给 | `V[:, 0:256]` | `V[:, 256:512]` | V TMA column offset，`phase1.cuh:471`；CuTe permutation 的源码注释，`config.h:131`。 |
| SV 的 token 分段 | 每个 CTA 都加载 token `0:64`，再加载 `64:128` | 同左，但取各自的 channel 半边 | `phase1.cuh:460`--`487`。 |

可以把两个联合 GEMM 看成：

```text
QK:  [128 heads, D] x [128 selected tokens, D]^T -> [128 heads, 128 tokens]
     Q head 由 CTA 分 64/64；K token 供给也由 CTA 分 64/64。

SV:  [128 heads, 128 tokens] x [128 tokens, 512 channels] -> [128 heads, 512]
     Q/output 仍按 head 分给两个 CTA；V 的 512 channel 则按 256/256 放到两个 CTA。
```

上图描述的是算法和数据供给。实际 C fragment 如何映射到 TMEM datapath 由
`SM100_MMA_F16BF16_2x1SM_*` 的 CuTe layout 决定，不能把它简化为普通行主序数组。
`csrc/kerutils/include/kerutils/device/sm100/gemm.cuh:271`--`277` 和
`:377`--`:383` 正是 CTA-local fragment 到 2-SM logical layout 的定义。

## 3. 资源预算：为什么恰好是 512 个 TMEM column

### 3.1 固定参数

`config.h:36`--`55` 定义：

```text
B_H = 128, B_TOPK = 128, NUM_THREADS = 512
D_tQ = 384
D_sQ = D_QK - 384 = 128 (D=512) 或 192 (D=576)
NUM_BUFS = 2
```

每个 CTA 是 512 threads，即四个 128-thread warpgroup。TMEM 一次通过
`cute::TMEM::Allocator2Sm().allocate(512, ...)` 整块申请，见
`phase1.cuh:142`--`146`。`Allocator2Sm` 的前置条件明确要求两个 CTA 使用同一 logical warp
参与分配，且传入相同的 SMEM destination offset；实现和约束见
`csrc/cutlass/include/cute/arch/tmem_allocator_sm100.hpp:124`--`145`。

TMEM 的物理 column 分配是：

| TMEM column 区间 | 大小 | 逻辑内容 | 标量格式 |
| --- | ---: | --- | --- |
| `[0, 256)` | 256 | `O` accumulator，逻辑上每 CTA 的 `64 x 512` 输出 fragment | FP32 |
| `[256, 320)` | 64 | `P = QK^T`，逻辑上每 CTA 的 `64 x 128` score fragment | FP32 |
| `[320, 512)` | 192 | Q 的尾部 384 维，供 TS UMMA 的 A operand 使用 | BF16 |

这不是估计，而是 `config.h:46`--`55` 的源代码常量。最后一段从
`q = 512 - D_tQ/2` 得到；对 BF16 的 TMEM pointer，两个 16-bit logical element 占一个 32-bit
物理 column，所以 384 个 K-dimension 只消耗 192 个物理 column。对于 P/O，2-SM fragment 的
datapath layout 也会使“逻辑 N 大小”和“所占 column 数”不同；例如 64-head 的 128-token P
恰好对应 64 columns。不要用 `M * N * sizeof(T)` 除以一个普通线性 buffer 来反推这里的地址。

这也解释了为什么 Q 只把最后 384 维搬进 TMEM：

```text
O 256 cols + P 64 cols = 320 cols
剩余                             192 cols
192 cols x 2 BF16 / col =          384 Q elements / row
```

`D=512` 的前 128 维、`D=576` 的前 192 维留在 SMEM，尾部统一为 384 维进入 TMEM。这样两种
head dimension 使用同一份 TMEM budget 和同一段 TS GEMM 代码。

### 3.2 SMEM aliasing 不是普通 double buffer

`SharedMemoryPlan` 的 union 在 `config.h:93`--`118`：

```text
union u
  q_full : [本 CTA 的完整 Q]
  s      : [sq = Q prefix] [v = 稍后复用 Q tail 的空间] [k = 独立于 q_full 的区域]
  o      : 最终 TMA store 的 staging buffer
```

关键的源码注释在 `config.h:99`--`101`：**K 不和 `q_full` overlap**。所以 K 可以在 Q 从 SMEM
复制到 TMEM 时提前 gather；相反 V 确实复用 Q tail 的位置，因此 V producer 必须等待
`bar_prologue_utccp`。输出 `o` 又在所有 Q/K/V 消费结束后复用整个 union。

`NUM_BUFS == 2` 主要是两组 barrier phase/validity-mask slot，而不是“完整 K/V 数据各有两块
SMEM”。K 和 V 的同一物理区域通过 `*_part_done` / `*_done` 精确释放后才被下一 tile 覆写。
这种做法省 SMEM，但使 barrier 生命周期成为正确性的核心。

## 4. 初始化阶段：从 Q TMA 到 Q-tail UTCCP

下面的时序严格对应 `phase1.cuh:106`--`148` 和 `:494`--`:520`。

| 顺序 | 执行者 | 动作 | 为什么需要 |
| ---: | --- | --- | --- |
| 1 | 两个 CTA 的 warp 0 | 初始化各自 local `SharedMemoryPlan` 中的 mbarrier | 每个 CTA 有 local barrier 地址，但后续会被 cluster 远程访问。 |
| 2 | 全 cluster | `cluster_sync()` | 源码明确注释：否则 CTA1 的 TMA 可能早于 CTA0 的 barrier 初始化。 |
| 3 | 两个 CTA 的 warp 0 各一线程 | 对各自的 64 个 Q head 发起 2-SM no-split TMA | Q 数据进入各自本地 `q_full`。 |
| 4 | 两个 CTA 的 warp 0 | `Allocator2Sm().allocate(512, ...)` | 两个 CTA 联合取得完整 TMEM column 空间；之后 CTA 内 `__syncthreads()` 让所有 warp 看见地址。 |
| 5 | CTA0, warp 12 的 elected thread | 在 `bar_prologue_q` 上登记 `128 * D_K * sizeof(bf16)` byte 并等待 | 这个 byte 数恰是两个 CTA 各自 `64 x D_K` Q TMA 的总和。 |
| 6 | CTA0, warp 12 | 对 Q 的最后 384 维发出 2-CTA `UTCCP`，SMEM -> TMEM | 之后 tail Q 不再占 SMEM。 |
| 7 | CTA0, warp 12 | multicast commit `bar_prologue_utccp` 给 CTA0/CTA1 | 两侧 V producer 现在可以安全覆写 Q tail。 |

这里的 TMA 封装不是标准 CuTe 的“自动二等分”版本，而是仓库自定义的
`SM100_TMA_2SM_LOAD_NOSPLIT`。其实现见
`csrc/kerutils/include/kerutils/device/sm100/tma_cta_group2_nosplit.cuh:23`--`34`：它会清除
barrier 地址的 peer bit，使两个 CTA 的 TMA transaction byte 都计入 CTA0 的 barrier。对应 Q
路径的总 byte account 由 CTA0 在 `phase1.cuh:505` 设置。

这也是为什么不能把这段改成两个普通 CTA-local TMA + 一个 `__syncthreads()`：后者既不能同步
另一个 CTA，也不能向 2-SM UMMA 表达联合 transaction completion。

## 5. 512 threads 的工作划分

`warpgroup_idx = threadIdx.x / 128`，所以每 CTA 有四个 128-thread group：

| group / warp | 职责 | 关键代码 |
| --- | --- | --- |
| WG0，warps 0--3 | 从 TMEM 读 P，mask、online softmax、写 BF16 S、按需缩放 TMEM O、最后导出 O/LSE | `phase1.cuh:150`--`386` |
| WG1，warps 4--7 | K gather producer；每个 warp 选一线程发 TMA gather4 | `phase1.cuh:387`--`446` |
| WG2，warps 8--11 | V gather producer；每个 CTA 负责 256 个 V channel | `phase1.cuh:447`--`489` |
| WG3，warps 12--15 | CTA0 warp12 的一线程发 UTCCP 和全部 UMMA；warp13 的 lane 0--15 生成 validity mask | `phase1.cuh:490`--`605` |

CTA1 的 warp12 不发第二份 MMA；只有 CTA0 warp12 的 elected thread 进入 `if (cta_idx == 0 &&
warp_idx == 12 && elect_one_sync())`。这不是漏掉了 CTA1 的计算：所发指令本身是
`tcgen05.mma.cta_group::2`，两个 CTA/SM 是同一协作操作的成员。自定义 SS/TS wrapper 的内联 PTX
可在 `gemm.cuh:231`--`240` 和 `:339`--`:348` 看到。

不同 group 的 register 上限也不同：WG0 `reg_alloc<144>`，K/V producer
`reg_dealloc<96>`，WG3 `reg_alloc<168>`，见 `phase1.cuh:150`、`:389`、`:449`、`:491`。它体现了
Blackwell 的典型 warp specialization：数据生产者压低 register 占用，softmax/MMA 控制路径保留更多
寄存器；TMEM 则承担 FP32 accumulator，避免把 O/P 长时间放在寄存器中。

## 6. 一个 Top-K tile 的完整事件图

令 `i` 为 Top-K tile 编号，tile 大小为 128。MMA issuer 的主循环实际执行
`k = 0 .. num_k_blocks`，并在 `k > 0` 时处理 `SV(k-1)`，见 `phase1.cuh:522`--`577`。

```text
K producer:      gather K0.prefix --- gather K0.tail --- gather K1.prefix --- ...
MMA issuer:                           QK0.prefix -> QK0.tail
WG0 softmax:                                             load P0 -> S0, rescale O
V producer:      gather V0.part0 --- gather V0.part1 --- gather V1.part0 --- ...
MMA issuer:                                                        SV0.part0 -> SV0.part1
                                                              QK1 ...
```

真实执行是异步交错的，不是上述每行串行；依赖关系如下。

### 6.1 QK(i)：K dimension 分成 prefix 和 tail

1. WG1 为 tile `i` gather K 的 `D_sQ` prefix。四个 warp 在每个 CTA 中各发多条
   `gather4`；两侧合起来正好覆盖 128 token。CTA0 对
   `bar_k_part0_ready[i % 2]` 登记 `128 * D_sQ * sizeof(bf16)` byte，并等待。
2. CTA0 warp12 执行 `utcmma_ss(tiled_mma_P_sQ, sQl, sKl, tP, true)`。
   `true` 表示第一个 K fragment 清零 P；helper 会在后续 16-wide K fragments 自动改为 accumulate，
   见 `csrc/kerutils/include/kerutils/device/sm100/helpers.cuh:18`--`43`。
3. 它 multicast commit `bar_qk_part_done`。此后 K producer 可以覆写 **prefix** K SMEM，而无需等
   tail 或 softmax。
4. WG1 gather K 的最后 384 维；CTA0 等待 `bar_k_part1_ready` 后执行
   `utcmma_ts(tiled_mma_P_tQ, tQr, sKr, tP, false)`。这里 A 是已在 TMEM 的 Q tail，B 是 SMEM
   的 K tail，`false` 表示继续累加 P。TS helper 的实现见 `helpers.cuh:97`--`119`。
5. `bar_qk_done` 的 2-CTA multicast commit 同时表示 P 已可读、K tail 已可覆写。

数学上它仍然只是一次完整 dot product：

\[
QK^T = Q_{[:,0:D_sQ]}K_{[:,0:D_sQ]}^T
     + Q_{[:,D_sQ:D_QK]}K_{[:,D_sQ:D_QK]}^T.
\]

这个拆分不是为了改变算法，而是为了把固定 384 维 Q tail 放在 TMEM，给两个支持维度一个统一且刚好
填满的 TMEM layout。

### 6.2 Softmax(i)：P 读完即可立刻释放

WG0 在 `bar_qk_done` 后：

1. 执行 `tcgen05_after_thread_sync()`，从 `tmem_cols::p` 用
   `tmem_ld_32dp32bNx<64>` 读出各自负责的 64 个 score；随后发 TMEM-load wait/fence。
2. 每个 CTA 的 128 个 WG0 thread 都执行
   `bar_p_free[i % 2].arrive(0u)`。这里的 `0u` **不是 transaction byte 数为零**，而是
   `ClusterBarrier::arrive(cta_id)` 的目标 CTA rank：它利用 `mapa.shared::cluster` 到达 CTA0 的
   barrier，见 `csrc/cutlass/include/cutlass/arch/barrier.h:380`--`384` 和 `:485`--`:504`。
3. 因此 CTA0 的 `bar_p_free` 收到 `128 * 2 = 256` 次到达后，才允许下一轮 QK 覆写同一 P TMEM
   区域。这个计数恰好覆盖两个 CTA 的所有 softmax consumer，见初始化
   `phase1.cuh:121`。
4. WG0 等 validity mask，做 max/reduction、online softmax；但在写 S 前还要等 `SV(i-1)` 完成，
   因为 S SMEM 和 O TMEM 都是复用资源。
5. 写完 S 且需要时缩放完 O 后，全部 256 个 WG0 thread 远程到达 CTA0 的
   `bar_so_ready`；MMA issuer 可开始 `SV(i)`。

所以 P 不是被“整个 attention 完成”才释放，而是在所有 consumer 从 TMEM load 完即释放。这让
`QK(i+1)` 能与 `softmax(i)` 和 `SV(i-1)` 重叠。

### 6.3 SV(i)：token dimension 分两半释放 V

V producer 在 prologue UTCCP 完成后运行。每个 CTA 对自己的 V channel half，先 gather token
`0:64`，后 gather `64:128`：

1. CTA0 等 `bar_so_ready[i]` 和 `bar_v_part0_ready[i]`，发
   `S[:,0:64] @ V[0:64,:]`。仅在 `i == 0` 的第一次 SV 发射时清零 O；源码中的循环变量为
   `k == 1`，因为它在外层 `k=1` 时计算 `SV(0)`。
2. `bar_sv_part_done[i]` 一到达，V producer 就能重用第一半 V SMEM 来装下一 tile。
3. CTA0 再等 `bar_v_part1_ready[i]`，发第二个 `S[:,64:128] @ V[64:128,:]`，累加 O。
4. `bar_sv_done[i]` 同时释放第二半 V，并允许 WG0 写下一 tile 的 S / 缩放同一 O accumulator。

这就是源文件顶部注释 `phase1.cuh:31`--`63` 中“QK 当前 tile、scale/exp 当前 tile、SV 前一 tile”
三段交错的具体实现。它不是完整的 K/V double buffering，而是 K-prefix、K-tail、V-part0、V-part1
逐段交接。

## 7. barrier 生命周期表

所有 transaction barrier 在两个 CTA 的 local plan 中分别初始化；`cluster_sync()` 后才允许跨 CTA
使用。`(k / NUM_BUFS) & 1` 是该 barrier slot 的 parity，避免复用 slot 时把上一轮到达误认为这一轮。

| barrier | producer / 到达者 | consumer | 释放的资源或状态 |
| --- | --- | --- | --- |
| `bar_prologue_q` | 两个 Q TMA，transaction byte 汇聚到 CTA0 | CTA0 warp12 | Q 完整到达，可做 Q-tail UTCCP。 |
| `bar_prologue_utccp` | CTA0 的 UTCCP commit，multicast 到两 CTA | 两侧 V producer | Q tail 已进 TMEM，V 可覆写它的 SMEM。 |
| `bar_k_part0_ready` | K prefix gather TMA | CTA0 MMA issuer | prefix K 可供第一个 SS QK。 |
| `bar_qk_part_done` | QK prefix 的 `tcgen05.commit`，multicast | 两侧 K producer | prefix K 已不再被 MMA 读取。 |
| `bar_k_part1_ready` | K tail gather TMA | CTA0 MMA issuer | tail K 可供 TS QK。 |
| `bar_qk_done` | QK tail commit，multicast | WG0、两侧 K producer | P 可读；tail K 可重用。 |
| `bar_p_free` | 两 CTA 的全部 WG0 thread 远程 arrive 到 CTA0 | CTA0 MMA issuer | 所有人已完成 P 的 TMEM load，可写下一 P。 |
| `bar_k_valid_ready/free` | warp13 的 16 个 lane 生产 16 byte mask；WG0 128 thread 消费 | WG0 / warp13 | mask 可读 / 对应 mask slot 可重写。 |
| `bar_so_ready` | 两 CTA 的全部 WG0 thread 远程 arrive 到 CTA0 | CTA0 MMA issuer | S 已写入且 O 已按新 pivot 缩放，可做 SV。 |
| `bar_v_part0_ready` / `bar_v_part1_ready` | 两 CTA 的 V gather transaction | CTA0 MMA issuer | 前/后 64 token 的 V 已到达。 |
| `bar_sv_part_done` / `bar_sv_done` | SV 第 1/2 段的 UMMA commit，multicast | V producer / WG0 | 前半 V 可重用；完整 V/S/O 依赖解除。 |

有两个常见误解：

1. `tcgen05.commit...mbarrier::arrive` 不是普通 CPU 式“发完指令立即保证结果已可读”的 flag。它是
   UMMA 异步工作和 mbarrier 的 completion handoff；consumer 在对应 `wait(parity)` 后仍按代码插入
   `tcgen05_after_thread_sync()` / async view fence。
2. `bar_p_free` 和 `bar_so_ready` 初始化为 256 不是多余的。它们要等两个 CTA 的每一个 WG0
   thread 都越过本地 TMEM/SMEM 访问点；若只让每 CTA 一个 lane 到达，P/O 或 S 的复用可能与仍在
   执行的 consumer 交叠。

## 8. CuTe/PTX 层：SS、TS、TMA 和 UTCCP 如何落地

### 8.1 两种 QK MMA

`TiledMMA_P_sQ` 是 2-SM **SS** atom，`TiledMMA_P_tQ` 是 2-SM **TS** atom，定义在
`config.h:120`--`126`。SS 表示 A/B 均来自 SMEM descriptor；TS 表示 A 从 TMEM，B 从 SMEM。
这就是 Q prefix 与 tail 的存储选择能直接落成两种 UMMA 的原因。

自定义 TS atom 最终发出的是：

```ptx
tcgen05.mma.cta_group::2.kind::f16 [tmem_c], [tmem_a], desc_b, ...;
```

它可直接在 `csrc/kerutils/include/kerutils/device/sm100/gemm.cuh:231`--`240` 查到。SS 版本则把
两个 operand 都传为 SMEM descriptor，见 `:339`--`:348`。`utcmma_ss` / `utcmma_ts` 不改变硬件
语义；它们负责按 CuTe fragment 的 K mode 逐块发 `cute::gemm`，并在首块后将
`ScaleOut::Zero` 改为 `ScaleOut::One`。

### 8.2 Q-tail 的 SMEM -> TMEM

CTA0 warp12 把 Q tail 描述为 `UMMA::Layout_K_SW128_Atom<bf16>`，然后对每个 `64 x 8` subtile
发 `SM100_UTCCP_2x64dp128bitlw0213_2cta::copy`，见 `phase1.cuh:495`--`519`。它不是 CUDA core
逐元素搬运；是 Blackwell 的 2-CTA `tcgen05.cp` 风格路径。tail 的 6 个 `64`-wide tile、每个 8
个 `64 x 8` subtile，刚好覆盖 `64 x 384` 的本 CTA Q tail。

### 8.3 不规则 K/V gather

K/V 不是连续 block，而是从 `indices` 指定的 token 行 gather。`tma_gather4_cta_group_2` 在
`csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh:36`--`52` 发出
`cp.async.bulk.tensor.2d ... gather4 ... cta_group::2`。每条请求携带四个 token row coordinate 和
一个连续 64-element column tile；这就是为何 K/V producer 以 `int4` 批量读 index。

KV tensor map 在 host 端以 BF16、64-element box、128-byte swizzle、L2 256B promotion 建立，见
`phase1.cuh:658`--`677`。`UMMA::Layout_*_SW128_Atom` 与该 TMA swizzle 一起保证 MMA operand
layout 相容；不要把它换成无 swizzle 的普通 SMEM layout。

### 8.4 架构背景与本 kernel 的边界

KernelWiki 的 [`wiki/hardware/tcgen05-mma.md`](.agents/skills/KernelWiki/wiki/hardware/tcgen05-mma.md)
（ID `hw-tcgen05-mma`，**verified**，依据 NVIDIA 官方 tuning guide 和 CUTLASS upstream code）给出的
最小 2-SM 模式是：

```ptx
tcgen05.mma.cta_group::2.kind::f16 [tmem_addr], desc_a, desc_b, idesc, 0;
```

它说明了为何本 kernel 可以让一个 elected thread 提交操作、让 TMEM 承接 accumulator。但该 wiki 的
通用形状示例不等于本文件的精确 `M=128/N=128 or 256` CuTe tiling；精确布局仍应以本仓库
`gemm.cuh` 和 `config.h` 为准。

## 9. Online softmax：`mi` 为什么不总是当前真实最大值

WG0 的数值逻辑位于 `phase1.cuh:154`--`285`。令

\[
z_j = P_j \cdot (\mathrm{sm\_scale}/\ln 2).
\]

代码维护的 invariant 是：

\[
L = \sum_{j\ \mathrm{seen}} 2^{z_j-m},\qquad
O = \sum_{j\ \mathrm{seen}} 2^{z_j-m}V_j.
\]

其中 `li` 是 `L`，`mi` 是当前使用的 base-2 pivot，`real_mi` 才是所有有效 token 的真实
`max(z)`。每个 thread 先处理 64 个 score，`idx_in_warpgroup ^ 64` 的配对 thread 处理同一 head 的
另 64 个 score；二者经 `rowwise_max_buf` 合并，见 `phase1.cuh:202`--`220`。

常规 online softmax 会在每个 tile 将 `m` 更新为新最大值，并重缩放旧 O。这里采用 lazy rescale：

```text
if cur_pi_max - mi <= 6:
    保持 mi，不读写旧 O，直接把新 tile 的 2^(z-mi) 加进来
else:
    new_m = max(cur_pi_max, mi)
    old_scale = 2^(mi-new_m)
    L *= old_scale
    O *= old_scale       // 从 TMEM load -> scale -> store
    mi = new_m
```

它仍满足上述 invariant：pivot 不变时旧项和新项都以同一个 `mi` 表示；pivot 改变时旧项和 O 同时乘
相同因子。阈值 6 的作用是允许新 exponent 最大到约 `2^6`，从而避免在最大值小幅上升时昂贵的
TMEM O read-modify-write。`real_mi` 独立更新，故最终 `max_logits` 仍是真实最大 logit，而不是
lazy pivot。

有效性 mask 由 warp13 的 16 个 lane 生成：每 lane 读 8 个 index，条件是
`0 <= index < s_kv && absolute_topk_position < topk_length`，见
`phase1.cuh:580`--`603`。WG0 把无效 P 置为 `-inf` 后才参与 max 和 exp。因此无效 gather 数据
不应进入数学结果。

最终：

```text
max_logits = real_mi * ln(2)
lse        = log(li) + mi * ln(2)       // 不包含 attn_sink
out        = O / (li + 2^(attn_sink * log2(e) - mi))
```

`attn_sink` 是一个 V 为零的虚拟项：它只增大输出的分母，不改变真实 token 的 `max_logits` 或写出的
`lse`。这和 `tests/ref.py:40`--`52` 的 reference 完全一致。若一个 query head 没有任何有效 index，
代码把 LSE 的 `-inf` 规范化成 `+inf`、输出置零，见 `phase1.cuh:289`--`337`。

## 10. Epilogue：TMEM O 到全局输出

所有 tile 的 `SV` 完成后，WG0 等最后一个 `bar_sv_done`，从 TMEM 分段读取 O：

1. 每个 thread 读 64 个 FP32 output component，乘最终 `1 / denominator`，转为 BF16；
2. 写到 alias 后的 `plan.u.o` SMEM staging；thread `0:63` 和 `64:127` 分别写输出 channel 的两个
   256-wide half；
3. 每轮 named barrier 后，warp0 和 warp1 的 elected thread 各发一个 TMA store，覆盖两个 64-channel
   tile；
4. `D_V=512` 共四轮，每 CTA 输出自己的 64 个 Q head。

对应代码为 `phase1.cuh:310`--`381`。TMA output map 是普通 `SM90_TMA_STORE`，而不是 2-SM TMA：
最终每个 CTA 各自把其本地 head 区间写回 `out[s_q_idx, h, :]`。

## 11. 可以从已编译 SASS 交叉验证什么

对工作区现有 `flash_mla/cuda*.so` 的 `sm_100f` code object 做过 SASS 检查。下面的命令可以再次定位
head128 的 512-dim 实例（若二进制与当前源码版本不同，结果只能作为旁证）：

```bash
/share/home/jintao/nvidia/bin/cuobjdump --dump-sass --gpu-architecture sm_100f \
  flash_mla/cuda.cpython-313-x86_64-linux-gnu.so 2>/dev/null | \
  sed -n '/identifier = .*fwd\/head128\/instantiations\/phase1_k512.cu/,/identifier = .*phase1_k576.cu/p'
```

观察到的 lowering 与源码设计相符：

| 源码抽象 | 在该二进制中观察到的指令族 |
| --- | --- |
| 2-SM Q TMA | `UTMALDG.3D.2CTA` |
| K/V irregular gather4 | `UTMALDG.2D.GATHER4.2CTA` |
| Q tail UTCCP | `UTCCP.T.S.2CTA.2x64dp128bit...` |
| transaction barrier wait/arrive | `SYNCS.PHASECHK.TRANS64.TRYWAIT`、`SYNCS.ARRIVE.TRANS64` |
| async-view / TMEM sequencing | `FENCE.VIEW.ASYNC.T`、`FENCE.VIEW.ASYNC.S` |
| output TMA store | `UTMASTG.3D` |

同一 resource dump 报告目标实例为 `REG:128 STACK:0 SHARED:1024 LOCAL:0`。其中 `SHARED:1024`
是 code object 的静态 resource 字段，**不能**解释为 kernel 总 SMEM 只有 1 KiB；本 kernel 的工作区是
`sizeof(SharedMemoryPlan)` 的动态 SMEM，并由 `cudaFuncSetAttribute` 在
`phase1.cuh:691`--`699` 申请。

## 12. 修改或调优时最容易破坏的约束

1. **不要删掉 prologue 的 `cluster_sync()`。** 这是跨 CTA barrier 初始化可见性的必要条件，源码本身已
   写明竞态原因。
2. **不要把 `arrive(0u)` 当成 no-op。** 它是 remote-arrive 到 CTA0；把它改成 local `arrive()` 会让
   CTA0 等不到 256 个 consumer。
3. **不要把 K 与 `q_full` overlap。** K 需要和 Q-tail UTCCP 并发；V 则必须等 UTCCP 后才可 reuse。
4. **不要在未等 `bar_sv_done(i-1)` 时写 `S(i)` 或缩放 O。** 这会直接让上一 tile 的 SV 读到被覆盖的
   S，或与 TMEM O 的 read/modify/write 冲突。
5. **不要删除 `tcgen05_after_thread_sync`、`tcgen05_before_thread_sync` 和 async-view fence。**
   UMMA/TMEM/TMA 不是普通同步 load/store；这些 fence 是 visibility/ordering 协议的一部分。
6. **`topk_length` 必须满足 `0 <= topk_length[q] <= params.topk`。** 当前代码用它计算
   `num_k_blocks`（`phase1.cuh:74`--`75`），但 host API 只检查 tensor shape，未在此路径 clamp。
   若它大于分配给 `indices` 的 `topk`，后续 K/V/index-mask 读取会越界。这是从代码审阅得出的输入
   合约/鲁棒性缺口，不是已经复现的运行时故障；现有测试生成的 `topk_length` 位于合法范围。
7. **保留 `topk % 128 == 0` 的契约，或补齐所有尾块路径。** `num_k_blocks` 虽然对
   `topk_length` 使用 `ceil_div`，但 index buffer 的静态容量和 regular kernel 的 host assertion 都假定
   原始 `topk` 对齐到 128。

## 13. Blackwell 资料来源与置信等级

以下资料用于解释硬件机制，而非替代本仓库源码作为本 kernel 行为的证据：

| 页面 | ID / 置信等级 | 本文采用的内容 |
| --- | --- | --- |
| [`wiki/hardware/tmem.md`](.agents/skills/KernelWiki/wiki/hardware/tmem.md) | `hw-tmem`，**verified** | TMEM 是 128 x 512 个 32-bit column 的显式分配空间；TMEM 读写须有正确 fence。 |
| [`wiki/hardware/tcgen05-mma.md`](.agents/skills/KernelWiki/wiki/hardware/tcgen05-mma.md) | `hw-tcgen05-mma`，**verified** | `tcgen05` 单线程提交、TMEM accumulator、`cta_group::2` 机制。 |
| [`wiki/hardware/2sm-cooperative.md`](.agents/skills/KernelWiki/wiki/hardware/2sm-cooperative.md) | `hw-2sm-cooperative`，**source-reported** | cluster 中两个 CTA 协作与 cluster barrier 的一般要求。 |
| [`wiki/hardware/tma.md`](.agents/skills/KernelWiki/wiki/hardware/tma.md) | `hw-tma`，**source-reported** | TMA 的异步 transaction/mbarrier 模式及 128B swizzle 背景。 |
| [`wiki/hardware/mbarrier.md`](.agents/skills/KernelWiki/wiki/hardware/mbarrier.md) | `hw-mbarrier`，**source-reported** | parity/phase 复用和 producer-consumer barrier 对的解释。 |
| [`wiki/techniques/warp-specialization.md`](.agents/skills/KernelWiki/wiki/techniques/warp-specialization.md) | `technique-warp-specialization`，**source-reported** | SM100 的 TMA/MMA/epilogue 分工模式。 |
| [`wiki/kernels/sparse-mla.md`](.agents/skills/KernelWiki/wiki/kernels/sparse-mla.md) | `kernel-sparse-mla`，**source-reported** | 稀疏 MLA 的“上游选 index，后续 gather attention”总体语义。 |

本文没有引用这些页面中的吞吐率数字，因为它们的 shape、dtype 和实现并不等同于这里的 BF16
128-head prefill kernel。

## 14. 一句话总结

`phase1.cuh` 的高层算法是标准 sparse FlashAttention，但它的实现把每个资源都压到 Blackwell 的
最小可复用粒度：Q tail 常驻 TMEM、P 在“所有 softmax thread 已读”时释放、K 按 K-prefix/tail
释放、V 按 token half 释放、S/O 通过一 tile 滞后的 SV 消费。2-CTA cluster、TMEM 512-column
精确切分、以及上述 barrier 图共同构成了它的设计；任何优化都应先保持这张依赖图不变，再讨论 tile、
register 或 TMA 策略。
