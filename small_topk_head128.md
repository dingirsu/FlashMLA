# `fwd_for_small_topk/head128/phase1.cuh` 设计分析，以及它和 regular head128 的区别

分析对象：

- small-Top-K：`csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.cuh`
- regular：`csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh`

本文只分析 small-Top-K 的 **BF16 prefill** 实例
`KernelTemplate<SparseAttnFwdMode::Prefill, 512>`。同一个 `.cuh` 还包含 FP8/decode/split-KV
分支，但这些分支在 prefill 实例中被 `if constexpr` 删除；不要因为文件中出现 `fp8_e4m3`、
`K_raw` 或 `extra_kv` 就把它们当作这条 prefill 路径的一部分。

> 核心结论：small-Top-K 不是 regular kernel 的“把 `B_TOPK` 从 128 改成 64”的缩小版。
> 它把完整 512 维 Q 搬进 TMEM，以 `D=256 + 256` 的 **dual GEMM** 形成两个 partial-P，并在
> softmax 前归并；它还把 K 和 V 复用为同一组四级 KV ring buffer，并通过 CLC 把一个
> 2-CTA cluster 变成可连续处理多个 query position 的 persistent worker。regular 版本则是
> 一个 query 一个 cluster，采用 Q-prefix SMEM + Q-tail TMEM 的两段 QK，以及 K/V 分离的
> 细粒度流水线。

## 1. 功能范围、入口和选择条件

两者的数学目标相同。给定上游已经选好的 `indices`，对每个 query position 和 128 个 Q head 计算：

\[
P_{h,j}=Q_hK_{\mathrm{indices}[j]}^T,\qquad
O_h=\sum_j\operatorname{softmax}(P_h\cdot\mathrm{sm\_scale})_jV_{\mathrm{indices}[j]}.
\]

它不是 Lightning Indexer 或 Top-K selection kernel；它只消费 `indices`。

### 1.1 实际 prefill 实例

small-Top-K 的 prefill 显式实例位于
`csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_prefill_k512.cu`。
虽然 `run()` 的通用模板写了 `D_QK == 512 || D_QK == 576`，但
`config.h:47` 有更强的 `static_assert(D_QK == 512)`，所以这个 head128 small-Top-K prefill
路径实际只支持：

```text
h_q = 128, h_kv = 1, d_qk = 512, d_v = 512, BF16 Q/KV
topk % 64 == 0
```

在 `csrc/api/sparse_fwd.h:262`--`276`，SM100f 上 `h_q == 128` 且 `d_qk == 512` 时，若
`topk <= 1280` 且 feature 检查通过，接口优先选择本路径；否则落到 regular head128。这个
`1280` 是 **dispatch policy**，不是本文件中可见的算法硬上限。

regular head128 则支持 `d_qk == 512 || 576`、固定 `d_v == 512`、要求 `topk % 128 == 0`，见
`csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:623`--`631`。

### 1.2 一页对比

| 维度 | small-Top-K head128 | regular head128 |
| --- | --- | --- |
| API 选择 | 通常 `d_qk=512, topk<=1280` | 较大 Top-K，或 `d_qk=576` |
| Top-K tile | 64 token | 128 token |
| cluster 生命周期 | CLC persistent，一个 resident cluster 连续处理多个 query | 一个 cluster 固定处理一个 query |
| Q 在 TMEM | 完整 512 维 | 固定 tail 384 维；prefix 为 128/192 维 SMEM |
| QK | TS dual GEMM，两个 partial-P 在 softmax 前相加 | prefix SS QK + tail TS QK，直接累加到同一 P |
| KV SMEM | 4 个真实的 `64 x 256` half-KV ring slot；同一 slot 同时作为 K 和 V | K/V producer 分开；通过 K-prefix/tail、V-part0/1 释放同一物理区 |
| P 处理 | TMEM 两个 partial P piece + `P_exchange` SMEM 归并 | TMEM 已是完整 P，直接 load/mask |
| query 间重叠 | 前一 query 的 O export 与当前 query 的 Q/TMEM/QK 重叠 | 只在单 query 内做 tile 流水 |
| 支持维度 | 仅 512 | 512、576 |

## 2. Launch 形状：同样是 2 CTA，但 small 版本是 persistent cluster

两者都使用：

```text
grid.x    = 2 * s_q
cluster   = (2, 1, 1)
threads   = 512 / CTA
cta_idx   = block_id_in_cluster().x   // 0 或 1
```

small 路径的 launch 位于 `phase1.cuh:1078`--`1098`。初始时，cluster 的 `blockIdx.x` 仍给出一个
query job；但 `run_outer_loop`（`:101`--`190`）在完成后会用 CLC 尝试取消尚未开始的 cluster，并把
被取消 cluster 的 ID 当作下一个 job：

```text
初始 job:  next_job = { valid, blockIdx.x, 0, 0 }
              s_q_idx = next_job.x / 2

完成当前 query 后:
  CTA0 warp10 发 CLC try_cancel（response multicast 给整个 cluster）
  所有参与 worker 等 response
  response.valid ? 用 response.x / 2 处理下一 query : 退出循环
```

因此，grid 里未驻留的 cluster 可以被已经完成工作的 resident cluster 抢占。它不是软件
`atomicAdd` 队列：仓库的 CLC wrapper 在
`csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh:266`--`318` 发出的是
`clusterlaunchcontrol.try_cancel.async ... multicast::cluster::all`，随后用
`clusterlaunchcontrol.query_cancel.get_first_ctaid::*` 解码结果。

KernelWiki 的
[`wiki/hardware/clc.md`](.agents/skills/KernelWiki/wiki/hardware/clc.md)（ID `hw-clc`，
**source-reported**）给出的控制流骨架与此完全对应：

```text
TILE_LOOP:
    // Request cancellation of a not-yet-launched cluster.
    clusterlaunchcontrol.try_cancel(response_smem, mbarrier)
    wait(mbarrier)

    has_work, tile_m, tile_n = clusterlaunchcontrol.query_cancel(response_smem)
    if (!has_work) return
```

这里的“tile”就是一个 2-CTA query-position job，而不是 QK 内部的 64-token tile。

### 2.1 `bar_clc_empty == 539` 不是魔数

`config.h:57` 定义 prefill 的：

```text
NUM_WORKER_THREADS = (128 + 4 + B_TOPK/8 + 1 + 128) * 2 + 1
                   = (128 + 4 + 8 + 1 + 128) * 2 + 1
                   = 539
```

这正好等于每个 outer job 后会调用 `bar_clc_empty.arrive(0u)` 的全部实际参与者：

| 参与者 | 每 CTA 数量 | 两 CTA 合计 |
| --- | ---: | ---: |
| WG0：Q fetch / O export | 128 | 256 |
| WG1：4 个 elected KV TMA issuer | 4 | 8 |
| WG2 warp9：8 个 mask producer lane | 8 | 16 |
| WG2 warp10：CLC loop participant | 1 | 2 |
| WG3：Scale/Exp | 128 | 256 |
| WG2 CTA0 warp8：唯一 UMMA issuer | - | 1 |
| **合计** | - | **539** |

CTA0 warp10 在处理当前 job 时先等上一轮的 `bar_clc_empty`，再发下一次 CLC query；所有 worker
在读完当前 CLC response 后向 CTA0 远程 arrive，见 `phase1.cuh:182`--`186`、`:743`--`:751`。
这保证 `CLCResponseObj` 不会在仍有角色读取上一响应时被覆写。若改动参与 `run_outer_loop` 的
warp/lane，而忘记同步更新这个计数，kernel 会永久等待或过早重用 response buffer。

regular kernel 没有 CLC 和这两条 barrier；它用静态 `blockIdx.x / 2` 永远绑定一个 `s_q_idx`。

## 3. TMEM 和 SMEM：small 用完整 Q 换取 dual-GEMM 形态

### 3.1 TMEM column 图

small 的 `config.h:75`--`83` 固定：

| 物理 TMEM column | 大小 | 内容 |
| --- | ---: | --- |
| `[0, 256)` | 256 | FP32 O accumulator |
| `[256, 384)` | 128 | 全部 512 维 BF16 Q |
| `[384, 448)` | 64 | FP32 partial-P workspace |
| `[448, 512)` | 64 | 逻辑上未使用 |

即使逻辑只需 448 columns，代码仍在 `phase1.cuh:63`--`66` 调用
`Allocator2Sm().allocate(512, ...)`。这是正确且必要的：`Allocator2Sm` 要求申请数是 32--512 的
二次幂，不能申请 448；其约束在
`csrc/cutlass/include/cute/arch/tmem_allocator_sm100.hpp:124`--`145`。

对比 regular：

```text
regular: [0,256) O | [256,320) P | [320,512) Q tail(384 BF16 dims)
small:   [0,256) O | [256,384) full Q(512 BF16 dims) | [384,448) partial P | slack
```

regular 恰好填满 512 columns；small 用完整 Q 以后留下 64 column slack，但获得了统一的
`d_qk=512` TS dual-GEMM 形式。

### 3.2 small 的 SMEM 是真实 4-stage KV buffer

prefill 下 `NUM_K_BUFS = 4`，`SharedMemoryPlan` 的主要 raw payload 为：

| 成员 | 原始大小（不含对齐/障碍物） | 生命周期 |
| --- | ---: | --- |
| `Q[64][512]` BF16 | 64 KiB | 先接收 Q，随后作为前一 query O 的 export staging。 |
| `K[4][64][256]` BF16 | 4 x 32 KiB = 128 KiB | 每 slot 是一个 64-token KV tile 的半个 feature plane；同时被 QK 当 K、被 SV 当 V。 |
| `S[64][64]` BF16 | 8 KiB | 当前 token tile 的 softmax 权重。 |
| `P_exchange[4][32][32]` FP32 | 16 KiB | 将 dual-GEMM 的两份 partial P 相加。 |
| `rowwise_max_buf` / `rowwise_li_buf` | 1 KiB | max、normalization 和 WG3 -> WG0 handoff。 |
| mask/coord/scale ring | 约 2 KiB | prefill 只实际使用 validity mask；其他成员为 decode 模板保留。 |

上述可见数组合计至少 224,288 B，`sizeof(SharedMemoryPlan)` 还会加入 barrier、CLC response 和
对齐。不要把 code object resource dump 里的静态 `SHARED` 字段误读成这个动态工作区大小；launch
明确以 `cudaFuncSetAttribute(..., sizeof(SharedMemoryPlan))` 申请动态 SMEM（`:1078`--`:1094`）。

这是两种设计最本质的 SMEM 差异：regular 用 union 和 K/V 分段释放来压低常驻数据；small 直接保留
四份 half-KV slot，换取 producer 可超前多个 64-token tile，以及跨 query 的 persistent 流水。

### 3.3 `smem.Q` 的三段生命周期

同一块 `smem.Q` 依次扮演：

```text
1. 当前 query 的 G -> S Q TMA destination
2. 当前 query 的 S -> T Q copy source
3. 前一 query 的 TMEM O -> S export staging，再 TMA store 到 global out
```

所以不是简单的 Q buffer。`bar_tQ_full` 确保当前 Q 已进 TMEM 后，WG0 才能把前一 query 的 O 写进
同一 SMEM；`bar_tQ_empty` 在当前 query 的最后一个 P 发射后表明 TMEM Q 已不再被 QK 使用，下一
query 才能覆写 TMEM Q。完整的跨 query 时序见第 8 节。

## 4. 512 threads 的 warpgroup 角色

每 CTA 仍是 4 个 128-thread warpgroup，但角色与 regular 完全不同：

| warpgroup | small-Top-K 角色 | 关键位置 |
| --- | --- | --- |
| WG0, warps 0--3 | 发 Q TMA；CTA0 发全 Q UTCCP；导出前一 query 的 O 并 TMA store | `phase1.cuh:192`--`396` |
| WG1, warps 4--7 | 每 warp 一条 elected TMA producer，装入 4-stage half-KV slot | `:397`--`:451` |
| WG2, warps 8--11 | CTA0 warp8 一线程发全部 QK/SV UMMA；warp9 lane 0--7 写 mask；warp10 管 CLC | `:533`--`:786` |
| WG3, warps 12--15 | 128 个 Scale/Exp thread，dual-P 归并、online softmax、写 S、缩放 O | `:787`--`:920` |

register policy 也相应变化：WG0/WG3 使用 `warpgroup_reg_alloc<176>`，WG1/WG2 使用
`warpgroup_reg_dealloc<80>`。这不是普通 CUDA block 中每个 warp 做同一件事的模式；TMEM 保留
accumulator，少数线程发异步 TMA/UMMA，大量线程处理 softmax 和输出。

regular 的映射恰好是：WG0 softmax/epilogue、WG1 K producer、WG2 V producer、WG3 CTA0 UMMA
+ mask。small 把 softmax 移至 WG3，把 Q/O 管线移至 WG0，并把 K/V 合并到 WG1 的同一 ring buffer。

## 5. 每个 query 的 prologue：完整 Q 入 TMEM

初始化由不同 warp 完成，然后所有线程执行：

```cpp
ku::barrier_cluster_arrive_relaxed();
ku::barrier_cluster_wait_acquire();
```

见 `phase1.cuh:37`--`90`。它替代 regular prologue 中的 `cute::cluster_sync()`，让两 CTA 的
barrier 初始化、TMEM allocation 和 descriptor prefetch 在进入 outer loop 前可见。

对一个 query，WG0 执行：

1. 两 CTA 的 warp0 elected thread 用 `SM100_TMA_2SM_LOAD_5D_NOSPLIT` 把各自 64 个 head 的
   **完整** Q 搬入本地 `smem.Q`（`:330`--`:345`）。Q tensor map 是 5D 的 128B-swizzled map；
   注释 `:978` 说明它故意把 `Q[0:64]` 和 `Q[256:320]` 分组，以适配后续 dual GEMM 的 UTCCP layout。
2. CTA0 对 `bar_sQ_full` 登记两个 CTA 合计的 `128 * 512 * sizeof(bf16)` transaction bytes，
   等 Q 完整到达（`:347`--`:350`）。
3. CTA0 等 `bar_tQ_empty`，然后对 4 个 128-dimension tile、每 tile 4 个 subtile 发
   `SM100_UTCCP_128dp256bit_2cta::copy`，共 16 次，把所有 512 Q dimension 放到
   `tmem_cols::Q=256` 起始处（`:352`--`:377`）。
4. `bar_tQ_full` 以 2-CTA multicast commit 通知其他角色：TMEM Q 可被 QK 消费，Q SMEM 可在
   合适时机改作前一输出的 staging。

regular 只对 tail 384 dimension 发 UTCCP，prefix 仍留在 SMEM，并因此要做 SS + TS 两段 QK。
small 的全 Q UTCCP 更重，但之后 QK 的 A operand 完全来自 TMEM，不需要 regular 的 `D_sQ` prefix
路径。

## 6. 64-token KV slot 与 dual-GEMM P

### 6.1 每个 `K[k_buf]` 实际保存什么

small 的 `B_TOPK=64`。WG1 的四个 elected TMA issuer 在每 CTA 合计覆盖 64 个 index；对于每个
token，CTA0 取 feature `[0:256)`，CTA1 取 `[256:512)`：

```cpp
// cta_idx 决定 512 维 KV 中哪一个 256-wide half
local_col * 64 + cta_idx * (D_K / 2)
```

见 `phase1.cuh:410`--`446`。两 CTA 的 index 列表是同一组 `gIndices + k*64`，并非像 regular QK
那样 CTA0/CTA1 分别装 token `0:64` 和 `64:128`。

该 slot 随后被同一块 SMEM 以两种 layout 解释：

```text
QK: sK = canonical K-major [64 tokens, 256 local K dims]
SV: sV = canonical MN-major [256 local V channels, 64 tokens]
```

代码分别在 `phase1.cuh:566`--`:570` 和 `:583`--`:591`。这就是“同一 `K[]` 同时是 K 和 V”的
精确含义：MLA 的 per-token KV vector 在 QK 后可直接以 V layout 重解释，无须再从 global memory
单独 gather 一份 V。

每个 slot 的生命周期为：

```text
bar_KV_empty(slot)
  -> WG1 TMA gather
  -> bar_KV_full(slot)
  -> QK(slot)
  -> softmax produces S(slot)
  -> SV(slot)
  -> bar_KV_empty(slot)  // 直到 SV 已消费 V 后才可重写
```

`NUM_K_BUFS=4` 是真实的四级 pipeline depth；`RingBufferState` 的 `stage=block % 4`、
`phase=(block/4)&1` 定义可见于 `csrc/utils.h:58`--`81`。

### 6.2 为什么 `TiledMMA_P` 的 N 是 `B_TOPK * 2`

配置是：

```cpp
SM100_MMA_F16BF16_2x1SM_TS_NOELECT<..., H_Q, B_TOPK*2, ...>
// *2 for dual gemm
```

见 `config.h:116`--`118`。这个 `*2` 不是“本 tile 实际有 128 个 Top-K token”。实际输入仍是 64
token；它表达的是 Q/K 512 dimension 被 dual-GEMM layout 组织成两个 partial-P piece。

最直接的源码证据在
`csrc/sm100/prefill/sparse/common_subroutine.h:47`--`66`：该 helper 明说 dual gemm 在 TMEM
中产生两份 P，一份占 rows `0:63`，另一份占 rows `64:127`，随后需要归并为单一 P。

`tQ` 和 `sK` 都以 `D_Q/2 == D_K/2 == 256` 创建：

```cpp
Tensor tQ = ... partition_shape_A(..., Shape<64, 256>);
Tensor sK = ... canonical_k_major_layout<64, 256, 128>();
```

见 `phase1.cuh:541`--`:570`。据此可得到算法层面的结果：

\[
P = P^{(0)} + P^{(1)},\qquad
P^{(r)} = Q^{(r)}(K^{(r)})^T,\quad r\in\{0,1\},
\]

其中两份 partial 的精确 datapath/TMEM-row 映射由 CuTe 的 2-SM dual-GEMM layout 决定，而不是普通
row-major array；“两个 256-dimension partial dot-product 最后相加”是由上述 fragment shape 与
`retrieve_mask_and_reduce_p` 的源码联合推得的结论。

### 6.3 P 的读取、mask 和归并

WG3 的每个 thread 拿到 32 个 score。`retrieve_mask_and_reduce_p`：

1. 按 local warp 从 `P` 和 `P+32` 读取本地 P / peer P；
2. 立即完成 TMEM load fence，并 remote-arrive `bar_P_empty`；此时后续 QK 已可覆盖 TMEM P；
3. 先把无效 token 对应的本地 P 写成 `-inf`；
4. warps `(0,2)`、`(1,3)` 通过 `P_exchange` 的两个 64-thread named barrier 交换 peer P；
5. 做 `p += p_peer`，得到完整 512-dimension score。

关键实现片段在 `common_subroutine.h:87`--`125`：

```cpp
ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(TMEM_COL_START, p);
ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(TMEM_COL_START + NUM_ELEMS_PER_THREAD, p_peer);
// ... mask p before reduction ...
cur_p[0] = ku::float2_add(cur_p[0], t[0]);
cur_p[1] = ku::float2_add(cur_p[1], t[1]);
```

这一步是 small 与 regular 数值路径的最大结构差异：regular 的 prefix SS 与 tail TS 都累加在同一
TMEM P fragment，softmax 读取时已经是完整 score；small 必须先通过 `P_exchange` 把两份 partial
score 合为一份，才可计算 max/exp。

## 7. 单个 64-token tile 的稳态流水

WG2 CTA0 warp8 的 UMMA issuer 的主循环在 `phase1.cuh:550`--`615`：

```text
for k = start .. end:
    issue P(k),              if k < end
    release TMEM Q,          if k is the final P issue
    issue O += S(k-1)V(k-1), if k > start
```

对 tile `i` 的依赖图是：

```text
WG1:       KV[i] TMA -> KV_full(slot i)
WG2/CTA0:                QK dual-GEMM(i) -> QK_done
WG3:                                      load/mask/reduce P(i) -> S(i), scale O
WG2/CTA0:                                                              SV(i)
WG1:                                                                         KV_empty(slot i)
```

在相邻 tile 上，`QK(i+1)` 与 `softmax(i)`、`SV(i-1)` 异步交错。四个 KV slot 允许 WG1 在
`SV(i)` 尚未结束时继续装入后续 tile；但单一 TMEM P、单一 S 与单一 O 仍分别由 barrier 严格串行
复用。

### 7.1 barrier 语义表（prefill）

| barrier | 到达者 | 等待者 | 保护的资源 |
| --- | --- | --- | --- |
| `bar_sQ_full` | 两 CTA 的 Q TMA transaction 汇聚 | CTA0 Q-to-TMEM issuer | Q SMEM 已完整就绪。 |
| `bar_tQ_empty` | 最后一个 QK P 发射后的 2-SM commit | 下一 query 的 CTA0 Q-to-TMEM issuer | 旧 TMEM Q 不再被 QK 读取。 |
| `bar_tQ_full` | 完整 Q UTCCP commit，multicast | UMMA issuer、WG0 output staging | TMEM Q ready；Q SMEM 可被 O staging 复用。 |
| `bar_KV_empty[4]` | SV commit，multicast | WG1 producer | 해당 KV slot 的 K/V 都已消费，可重写。 |
| `bar_KV_full[4]` | gather4 TMA transaction | CTA0 QK issuer | 一个 `64 x 512` logical KV tile 已由两 CTA 的 256-half 拼齐。 |
| `bar_QK_done` | dual QK 2-SM commit | WG3 | partial P 已可读。 |
| `bar_P_empty` | 两 CTA 的 128 WG3 thread remote arrive，合计 256 | CTA0 QK issuer | 所有 TMEM P load 已发出，P 可覆写。 |
| `bar_SV_done` | SV 2-SM commit | WG3 | 旧 S 已消费，O 写/缩放不会和 SV 冲突。 |
| `bar_S_O_full` | 两 CTA 的 128 WG3 thread remote arrive，合计 256 | CTA0 SV issuer | S 已写，且旧 O 若需要已按新 pivot 缩放。 |
| `bar_tOut_full` | 所有 SV 完成后的 2-SM commit | WG0 | 当前 query 的 O accumulator 可以 export。 |
| `bar_tOut_empty` | 两 CTA 的全部 WG0 thread 在最后一次 TMEM O load 后 remote arrive，合计 256 | 下一 query 首次 SV issuer | 前一 O 已从 TMEM 读出，可清零/写当前 O。 |
| `bar_li_full` / `bar_li_empty` | WG3 的 64 个 row owner / WG0 的 128 consumer | WG0 / 下一 WG3 round | output scale 从 softmax 交给 O export，随后 `rowwise_li_buf` 可复用。 |
| `bar_valid_coord_scales_full/empty[4]` | warp9 的 8 个 mask lane / WG3 的 128 consumer | WG3 / warp9 | 64-token validity mask 的四级 ring slot。 |
| `bar_clc_full/empty` | CLC transaction / 全部 539 persistent participants | 所有 outer-loop 角色 / CTA0 CLC producer | CLC response 的可读与可覆写。 |

`arrive(0u)` 的 `0u` 仍然是 **目标 CTA rank 0**，不是“零 transaction byte”。
`ClusterBarrier::arrive(uint32_t cta_id, ...)` 会用 `mapa.shared::cluster` 做 remote arrive，见
`csrc/cutlass/include/cutlass/arch/barrier.h:380`--`384`、`:485`--`:504`。

regular 同样使用远程 256-arrival barrier 保护 P 和 S/O，但没有 4-slot `bar_KV_*`，也没有
跨 query 的 `bar_tQ_*`、`bar_tOut_*`、`bar_clc_*` 体系。

## 8. 跨 query 的 Q/O ping-pong

small 的 persistent 设计最值得注意的地方是它不等 O export 完才加载下一 query：

```text
query q:
  Q(q) G->S -> T(Q) -> QK/S/ SV -> tOut_full(q)

query q+1:
  Q(q+1) G->S -> T(Q) -----------------------------+
                                                     |
WG0:                   O(q) T->S(smem.Q) -> TMA store out(q)
WG2:                                      wait tOut_empty(q), then first SV(q+1) clears O
```

具体顺序：

1. WG0 为当前 `args` 完成 Q 的 TMA 与 UTCCP；
2. 若有 `last_args`，WG0 调 `perform_o_copy_out(last_args, false)` 导出前一 query 的 O；
3. 它在第一次向 `smem.Q` 写 O 前等待当前 query 的 `bar_tQ_full`，因此不会覆写仍在用于 UTCCP 的 Q；
4. `bar_tOut_empty` 在所有 WG0 thread 发起最后一段 TMEM O load 后到达 CTA0。当前 query 的第一
   个 SV 在清零/写 O 前等待它，而无需等待全局 TMA store 完成；此时 O 已在寄存器/SMEM staging 中；
5. 当 CLC 无更多 job，WG0 最后一次导出 O，`is_last_o=true` 时还调用
   `cudaTriggerProgrammaticLaunchCompletion()`（`:214`--`:227`）。它是程序化 dependent-launch
   信号；是否存在可重叠的后继 launch 取决于外部 launch 链，不能仅凭本文件推断。

regular 的 output epilogue 只发生在该 query 的最后一个 SV 后，不存在“前一 query O 与当前 query
QK”这一层 ping-pong，也没有用 Q SMEM 作为前一 O staging。

## 9. Softmax 的数值不变量：算法相同，P 来源不同

WG3 的 `mi`、`li`、`real_mi` 逻辑在 `phase1.cuh:795`--`919`，与 regular 的 online softmax
不变量相同。令：

\[
z_j=P_j\cdot(\mathrm{sm\_scale}/\ln 2),
\quad L=\sum_j 2^{z_j-m},
\quad O=\sum_j 2^{z_j-m}V_j.
\]

每个 64-token tile 后：

```text
if cur_max - mi <= 6:
    保留 mi；不读写旧 O；直接添加新 tile 的权重
else:
    new_m = max(cur_max, mi)
    old_scale = 2^(mi - new_m)
    li *= old_scale
    O  *= old_scale       // TMEM load -> multiply -> store
    mi = new_m
```

`real_mi` 始终独立记录真实最大值，故最终：

```text
max_logits = real_mi * ln(2)
lse        = log(li) + mi * ln(2)       // 不含 attn_sink
out        = O / (li + exp2(attn_sink * log2(e) - mi))
```

`attn_sink` 是零 V 的虚拟项，只影响 output denominator；`max_logits` 和写出的 `lse` 仍只描述真实
KV token。这与 `tests/ref.py:40`--`52` 一致。无有效 index 时，代码将 LSE 的 `-inf` 规范成
`+inf`，O 输出为零。

两者数值路径的不同点仅在 softmax 前：small 先 mask + 归并 partial P；regular 读取的 P 已是
完整 QK score。两者都在 P/O 中使用 FP32 accumulation，并将 S 转成 BF16 供 SV UMMA 使用。

## 10. SASS 交叉检查

对工作区已有的 `flash_mla/cuda*.so` 的 `sm_100f` code object 可用：

```bash
/share/home/jintao/nvidia/bin/cuobjdump --dump-sass --gpu-architecture sm_100f \
  flash_mla/cuda.cpython-313-x86_64-linux-gnu.so 2>/dev/null | \
  awk '
    /identifier = .*fwd_for_small_topk\/head128\/instantiations\/phase1_prefill_k512.cu/ { in_section=1 }
    in_section {
      if (seen && /^identifier =/) exit
      print
      seen=1
    }'
```

该二进制中可观察到以下对应关系（若二进制落后于当前源码，只能作为 lowering 旁证）：

| 源码机制 | 观察到的 SASS 指令族 |
| --- | --- |
| 5D 2-CTA Q TMA | `UTMALDG.5D.2CTA` |
| 完整 Q 的 16 个 2-CTA UTCCP subtile | 多条 `UTCCP.T.S.2CTA tmem[0x100..0x178]` |
| 64-token KV irregular gather | `UTMALDG.2D.GATHER4.2CTA` |
| output TMA store | `UTMASTG.5D` |
| mbarrier parity wait | `SYNCS.PHASECHK.TRANS64.TRYWAIT` |
| TMEM/SMEM async view ordering | `FENCE.VIEW.ASYNC.T`、`FENCE.VIEW.ASYNC.S` |
| warpgroup register policy | `USETMAXREG.TRY_ALLOC.CTAPOOL`、`USETMAXREG.DEALLOC.CTAPOOL` |

CLC 的最直接证据是源码 wrapper 的内联 PTX，而非猜测一个反汇编 mnemonic：
`intrinsics.cuh:280`--`287` 发出
`clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128`。

## 11. 修改时的高风险点

1. **不要删除 partial-P 归并。** `TiledMMA_P` 的 `B_TOPK*2` 是 dual-GEMM layout；直接把两份
   TMEM P 中的一份喂给 softmax 会少掉一半 K dimension 的 dot product。
2. **不要把 four-slot K buffer 当成只读 K cache。** 一个 slot 还要充当 V，必须等 `bar_SV_done` 后
   的 `bar_KV_empty` 才能重写。
3. **不要改变 `NUM_WORKER_THREADS` 的参与集合而不重算 539。** 这是 CLC response 复用的正确性
   条件，不是 tuning 参数。
4. **不要提前复用 `smem.Q`。** 输出 export 在同一块 SMEM 上发生；必须保留
   `bar_tQ_full`、`tma_store_wait` 和 named barrier 的次序。
5. **不要删除 `bar_tOut_empty`。** 它允许“旧 O 已从 TMEM 发起读取”后开始清空当前 O；删掉会让
   当前首次 SV 与旧 O export 竞争同一 TMEM accumulator。
6. **保留 `topk_length` 合约：`0 <= topk_length[q] <= params.topk`。** small 路径用
   `ceil_div(topk_length, 64)` 计算 tile 数（`:154`--`:164`），但 host API 对其数值没有在此路径
   clamp。若它大于 `indices` 分配的 `topk`，后续 index/KV/mask 读取可能越界。这是代码审阅得到的
   输入合约/鲁棒性缺口，不是已复现的故障；regular 路径有同类问题，只是 tile 大小为 128。
7. **不要因为文件含有 decode FP8 代码就混用其 barrier count。** prefill 的 `K_raw` 数量为零，
   `bar_valid_coord_scales_full` 预期 8 次到达；decode 的 producer/consumer 数完全不同。

## 12. Blackwell 背景资料与置信等级

以下资料解释架构机制；本 kernel 的具体行为以上述源码、辅助封装和 SASS 为主证据。

| 页面 | ID / 置信等级 | 本文采用的内容 |
| --- | --- | --- |
| [`wiki/hardware/tmem.md`](.agents/skills/KernelWiki/wiki/hardware/tmem.md) | `hw-tmem`，**verified** | TMEM 是 128 x 512 个 32-bit column 的显式分配空间；读取/写入需要正确 fence。 |
| [`wiki/hardware/tcgen05-mma.md`](.agents/skills/KernelWiki/wiki/hardware/tcgen05-mma.md) | `hw-tcgen05-mma`，**verified** | single-thread UMMA submission、TMEM accumulator、`cta_group::2` 的架构背景。 |
| [`wiki/hardware/2sm-cooperative.md`](.agents/skills/KernelWiki/wiki/hardware/2sm-cooperative.md) | `hw-2sm-cooperative`，**source-reported** | 两 CTA cluster 协作和 cluster-scope synchronization。 |
| [`wiki/hardware/tma.md`](.agents/skills/KernelWiki/wiki/hardware/tma.md) | `hw-tma`，**source-reported** | TMA transaction/mbarrier 和 128B swizzle 背景。 |
| [`wiki/hardware/mbarrier.md`](.agents/skills/KernelWiki/wiki/hardware/mbarrier.md) | `hw-mbarrier`，**source-reported** | parity/phase ring-buffer 复用的语义。 |
| [`wiki/hardware/clc.md`](.agents/skills/KernelWiki/wiki/hardware/clc.md) | `hw-clc`，**source-reported** | `try_cancel` 驱动的 persistent cluster 调度。 |
| [`wiki/techniques/persistent-kernels.md`](.agents/skills/KernelWiki/wiki/techniques/persistent-kernels.md) | `technique-persistent-kernels`，**source-reported** | persistent loop 的 response/wait/exit 模式。 |
| [`wiki/hardware/pdl-gdc.md`](.agents/skills/KernelWiki/wiki/hardware/pdl-gdc.md) | `hw-pdl-gdc`，**source-reported** | `cudaTriggerProgrammaticLaunchCompletion()` 所属的 PDL/GDC 背景。 |

本文没有引用 wiki 中的性能数字；它们不是本仓库这个 BF16、64-token、128-head prefill 实例的实测结果。

## 13. 最终理解

regular head128 把一个 query 的资源压到极致：固定 128-token tile、Q 的 384-dim tail 驻留 TMEM、
K/V 在同一 query 内按片段释放。small-Top-K 则为小 Top-K 形状选择另一种平衡：64-token tile、
完整 Q TMEM、partial-P reduction、四级真实 KV ring buffer，并把 Q load、O export、下一个 query
的 CLC job 获取编织成 persistent cluster pipeline。

因此，两者虽实现相同 attention 方程，但优化方向不同。读 small 的正确顺序是：先看 CLC outer loop
如何让一个 cluster 连续拿 query，再看 `smem.Q` 的跨 query 生命周期，最后看 dual-GEMM partial P
在 softmax 前如何归并；若按 regular 的“每 CTA 一个 query、P 已完整”的心智模型去读，最容易误判
barrier 和 TMEM 内容。
