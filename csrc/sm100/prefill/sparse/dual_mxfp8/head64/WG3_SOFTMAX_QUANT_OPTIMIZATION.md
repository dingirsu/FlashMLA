# WG3 softmax 与 S quant 优化记录

日期：2026-08-28

## 测试方法

编译：

```bash
PYTHON=/usr/bin/python3 NVCC_THREADS=16 ./compile_sm100.sh dual_mxfp8
```

测速：

```bash
/usr/bin/python3 scripts/run_dual_mxfp8_rank1.py \
  --benchmark --warmup 30 --iterations 150
```

固定形状为：

- prefill: `sq=32768, sk=32768, topk=512`
- decode: `sq=4, sk=32768, topk=512`

表中的 prefill 时间取同一台 SM100 GPU 上的 CUDA event median。decode 只有
4 个 query，28--31 us 区间的 launch、split-KV 和系统噪声占比较大，因此只用来
检查是否出现明显回退，不把小于约 1 us 的变化视为有效收益。

## 最终结果

| 版本 | MXFP8 prefill median | 相对基线 | BF16 对照 | MXFP8/BF16 |
|---|---:|---:|---:|---:|
| 基线 | 约 1927 us | - | 约 3321 us | 约 1.72x |
| 最终版本 | 约 1855 us | -3.7% latency | 约 3321 us | 约 1.79x |

最终版本保持：

- prefill: 128 registers，0 spill store/load
- decode: 128 registers，原有 4-byte store / 32-byte load spill，没有新增 spill

## 尝试记录

| 尝试 | 结果 | 处理 | 说明 |
|---|---|---|---|
| 用 `float2` 处理 `p * scale - new_max` | 变快 | 保留 | 明确生成 `FFMA2`，把每两个元素的乘加融合为一条指令。 |
| 用 `float2` 累加 softmax sum | 变快 | 保留 | 明确生成 `FADD2`；和 `float2` 的 V token scale 乘法一起，单 accumulator 版本约 1904 us。 |
| 用 `float2` 乘 V token scale | 变快 | 保留 | 明确生成 `FMUL2`，同时使用 64-bit load/store 搬运相邻两个元素。 |
| 两路交错 sum/max accumulator | 变快 | 继续扩展 | 约 1882 us；缩短展开循环内 `FADD2` 和 `FMNMX` 的串行依赖链。 |
| 四路交错 sum/max accumulator | 变快 | 保留 | 约 1855 us，是本轮最好结果，且没有增加 spill。 |
| 八路交错 sum/max accumulator | 变慢 | 回退 | 约 1870 us；额外 live values 和最终归并成本超过依赖链缩短收益。 |
| 把初始 32-element `p` max 改成两路 max | 变慢 | 回退 | 四路 softmax/quant 之前的版本从约 1904 us 变成约 1908 us；标量 `get_max` 更好。 |
| 使用 `max.f32x2` | 无法编译 | 删除 | 当前 PTX ISA 没有该指令，`ptxas` 报 `Unexpected instruction types specified for 'max'`。max 仍使用标量 `FMNMX`。 |
| UE8M0 reciprocal 先在 exponent byte 上做 `254-exp` | 单独无可测收益 | 保留 | 避免 FP32 reciprocal；与 CUTLASS UE8M0 fast reciprocal 的定义一致。 |
| UE8M0 reciprocal 直接左移 23 bit 构造 FP32 | 单独无可测收益 | 保留 | UE8M0 是纯指数，`(254-exp)<<23` 正好是 reciprocal power-of-two 的 FP32 bit pattern，移除 UE8M0 -> BF16 -> FP32 转换链。 |
| 用 exponent byte 表示 `1e-20` 下限 | 单独无可测收益 | 保留 | clamp 到 UE8M0 exponent byte 61，保持原先 `max(scale, 1e-20)` 后再向上量化的行为。 |
| E4M3 x2 conversion | 原来已经最优 | 不改 | 原代码的 `__nv_cvt_float2_to_fp8x2` 已生成 packed `cvt...e4m3x2`，无需另写 PTX。 |
| `exp2f` 向量化 | 无可用指令 | 不做 | PTX/SASS 只有标量 `ex2.approx` / `MUFU.EX2`，不存在对应的 FP32x2 指令。通过交错 accumulator 隐藏其延迟。 |

## 数值回归

最终代码使用以下命令比较 Torch CUTLASS 模拟与 kernel：

```bash
/usr/bin/python3 scripts/run_dual_mxfp8_rank1.py
```

需要保持基线的 LSE、prefill 和 decode 误差范围，不允许出现 NaN。最终实测值记录在
下方，后续修改时更新本节和上面的尝试表。

<!-- FINAL_PRECISION_RESULTS -->

```text
LSE: max_abs=0.000671387, mean_abs=0.000113487
prefill [0:256]:   max_abs=16,  mean_abs=0.00163746
prefill [256:512]: max_abs=16,  mean_abs=0.00238043
decode [0:256]:    max_abs=16,  mean_abs=0.00125122
decode [256:512]:  max_abs=0.5, mean_abs=0.0000305176
NaN: none
```

## 实现说明

UE8M0 scale byte `e` 与正常 FP32 的 biased exponent 一致。对正常 power-of-two：

```text
scale_bits      = e << 23
reciprocal_exp  = 254 - e
reciprocal_bits = (254 - e) << 23
```

这里的 scale exponent 在使用前 clamp 到 61，因此不会进入 UE8M0 zero、NaN 或
FP32 subnormal 的特殊编码范围。
