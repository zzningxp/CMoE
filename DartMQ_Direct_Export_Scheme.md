# GPTQ 直接导出量化数据方案 — Q*_D 类型家族 (最终版)

> 最后更新: 2026-03-10
> 状态: 最终方案，针对 DartMQ 适配

---

## 1. 问题分析

### 1.1 当前方案的问题

```
现有链路（存在两次精度损失）:

GPTQ量化(Int4) → 反量化为FP16 → llama.cpp 用 min-max 重新量化为 Q4_1
     ↓                ↓                        ↓
  scale/zero      精度损失1                  精度损失2
  q_int索引     (rounding误差)         (参数重算，±1索引偏差)
```

**核心问题**：
1. **反量化 rounding 误差**：GPTQ 的 `scale * (q - zero)` 反量化为 FP16 时引入浮点截断
2. **参数重算不一致**：llama.cpp 的 Q4_1 使用 min-max 方式重新计算 d/m，与 GPTQ 的 Hessian 优化 scale/zero 不同
3. **整数索引偏差**：两次量化导致典型的 ±1 索引漂移，直接体现为 PPL 下降

### 1.2 新方案的核心思想

**直接保存 GPTQ 量化的原始结果 (scale, zero, q_int)，在 Python 端完成参数映射和 bit packing，直接写入 GGUF，跳过一切反量化/再量化过程。**

```
新链路（零精度损失）:

GPTQ量化 → 保存 scale/zero/q_int (.pt) → 参数映射+打包 → 直接写入 GGUF
                                              ↓
                                      d = scale            (纯数学转换)
                                      m = -zero * scale    (纯数学转换)
                                      qs = pack(q_int)     (bit排列，无损)
```

### 1.3 数学等价性证明

```
GPTQ 反量化公式:  value = scale * (q_int - zero)
                       = scale * q_int - scale * zero
                       = q_int * scale + (-zero * scale)

Q4_1 反量化公式:  value = q_int * d + m

令 d = scale, m = -zero * scale，则两式完全等价。
```

**结论**：GPTQ 的量化数据可以通过纯数学参数映射无损地表示为 llama.cpp 的 block 格式，整个过程不涉及任何量化/反量化操作。

---

## 2. 方案范围：Q*_D 类型家族 (DartMQ专属)

### 2.1 为什么需要一组新类型

GPTQ 支持多种位宽 (3/4/5/6-bit)，而 llama.cpp 现有类型的兼容情况如下：

| GPTQ 位宽 | maxq | 现有最接近类型 | 能否直接复用？ | 原因 |
|---|---|---|---|---|
| 3-bit | 7 | Q3_K (super-block=256, 多级scale) | **不能** | Q3_K 结构完全不同 |
| 4-bit | 15 | Q4_1 (block=32, d+m+qs) | **能** | 二进制格式完全一致 |
| 5-bit | 31 | Q5_1 (block=32, d+m+qh+qs) | **能** | 格式兼容，需拆分第5位 |
| 6-bit | 63 | Q6_K (super-block=256, 多级scale) | **不能** | Q6_K 结构完全不同 |

- Q3_K / Q6_K 使用 256 元素的 super-block 加多级量化 scale，与 GPTQ 的简单 (scale, zero, q_int) 格式根本不兼容
- 为统一性和可维护性，建立完整的 Q3_D / Q4_D / Q5_D / Q6_D 家族

### 2.2 统一设计原则

所有 Q*_D 类型共享：
- **反量化公式**: `value = q_int * d + m`
- **参数来源**: `d = GPTQ_scale`, `m = -GPTQ_zero * GPTQ_scale`
- **block_size**: 32 (与 GPTQ groupsize=32 对齐)
- **存储方式**: `[d:FP16][m:FP16][qs:packed_bits]`

### 2.3 各类型 block 结构

#### Q3_D (3-bit, 16 bytes/block)

采用 **低2位 + 第3位分离存储**（SIMD 友好）：

```c
#define QK3_D 32
typedef struct {
    ggml_half d;        // scale                   (2 bytes)
    ggml_half m;        // min = -zero * scale     (2 bytes)
    uint8_t ql[8];      // 低2位, 4 values/byte    (8 bytes)  32×2bit = 64bit
    uint8_t qh[4];      // 第3位 bitmap, 8 bits/byte (4 bytes) 32×1bit = 32bit
} block_q3_d;           // Total: 16 bytes
static_assert(sizeof(block_q3_d) == 16, "wrong q3_d block size");
```

提取逻辑:
```
q_int[j] = ((ql[j/4] >> (2*(j%4))) & 0x03)   // 低2位
          | (((qh[j/8] >> (j%8)) & 0x01) << 2) // 第3位
// q_int ∈ [0, 7]
```

#### Q4_D (4-bit, 20 bytes/block)

与 Q4_1 **二进制格式完全一致**：

```c
#define QK4_D 32
typedef struct {
    ggml_half d;        // scale                   (2 bytes)
    ggml_half m;        // min = -zero * scale     (2 bytes)
    uint8_t qs[16];     // 4-bit nibbles, 2/byte   (16 bytes) 32×4bit = 128bit
} block_q4_d;           // Total: 20 bytes
static_assert(sizeof(block_q4_d) == 20, "wrong q4_d block size");
```

打包规则 (与 Q4_1 一致):
```
qs[j] = q_int[j] | (q_int[j+16] << 4)   // j = 0..15
// 低 nibble = 前半元素, 高 nibble = 后半元素
```

**重要**: 这是 Q4_1 的打包顺序，不是相邻对打包。

#### Q5_D (5-bit, 24 bytes/block)

与 Q5_1 **二进制格式完全一致**：

```c
#define QK5_D 32
typedef struct {
    ggml_half d;        // scale                   (2 bytes)
    ggml_half m;        // min = -zero * scale     (2 bytes)
    uint8_t qh[4];      // 第5位 bitmap, 8 bits/byte (4 bytes) 32×1bit = 32bit
    uint8_t qs[16];     // 低4位 nibbles, 2/byte   (16 bytes) 32×4bit = 128bit
} block_q5_d;           // Total: 24 bytes
static_assert(sizeof(block_q5_d) == 24, "wrong q5_d block size");
```

提取逻辑 (与 Q5_1 一致):
```
low4_0 = qs[j] & 0x0F             // 前半元素低4位
low4_1 = qs[j] >> 4               // 后半元素低4位
bit5_0 = (qh_u32 >> j) & 1        // 前半元素第5位
bit5_1 = (qh_u32 >> (j+16)) & 1   // 后半元素第5位
q_int[j]    = low4_0 | (bit5_0 << 4)   // ∈ [0, 31]
q_int[j+16] = low4_1 | (bit5_1 << 4)
```

#### Q6_D (6-bit, 28 bytes/block)

采用 **低4位 + 高2位分离存储**：

```c
#define QK6_D 32
typedef struct {
    ggml_half d;        // scale                   (2 bytes)
    ggml_half m;        // min = -zero * scale     (2 bytes)
    uint8_t ql[16];     // 低4位 nibbles, 2/byte   (16 bytes) 32×4bit = 128bit
    uint8_t qh[8];      // 高2位, 4 values/byte    (8 bytes)  32×2bit = 64bit
} block_q6_d;           // Total: 28 bytes
static_assert(sizeof(block_q6_d) == 28, "wrong q6_d block size");
```

提取逻辑:
```
low4_0 = ql[j] & 0x0F                        // 前半低4位
low4_1 = ql[j] >> 4                           // 后半低4位
hi2_0  = (qh[j/4] >> (2*(j%4))) & 0x03       // 前半高2位
hi2_1  = (qh[(j+16)/4] >> (2*((j+16)%4))) & 0x03  // 后半高2位
q_int[j]    = low4_0 | (hi2_0 << 4)   // ∈ [0, 63]
q_int[j+16] = low4_1 | (hi2_1 << 4)
```

### 2.4 汇总

| 类型 | 位宽 | block_size | block 字节 | 压缩比(vs FP16) | 有等价现有类型？ | C++实现策略 |
|---|---|---|---|---|---|---|
| Q3_D | 3 | 32 | 16 | 4.0x | 无 | **新写** |
| Q4_D | 4 | 32 | 20 | 3.2x | Q4_1 | 转发Q4_1 |
| Q5_D | 5 | 32 | 24 | 2.67x | Q5_1 | 转发Q5_1 |
| Q6_D | 6 | 32 | 28 | 2.29x | 无 | **新写** |

---

## 3. 实现方案

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                  Phase 1: Python 端 (DartMQ)                │
│                                                             │
│  GPTQ量化 ──→ fasterquant() 中额外保存:                     │
│               scale (FP16), zero (FP16), q_int (未打包整数)  │
│               ──→ 保存为 .pt 文件                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 2: convert_hf_to_gguf.py                 │
│                                                             │
│  加载 .pt ──→ 参数映射: d=scale, m=-zero*scale              │
│           ──→ 按 Q*_D block 格式打包 qs                     │
│           ──→ add_tensor(raw_dtype=Q4_D) 写入 GGUF          │
│                                                             │
│  非量化张量 (embedding, norm 等) ──→ 正常 FP16/FP32 路径     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         Phase 3: llama.cpp (Q*_D 类型注册与推理)             │
│                                                             │
│  ggml.h:          注册 GGML_TYPE_Q3_D/Q4_D/Q5_D/Q6_D       │
│  ggml-common.h:   定义 block_q*_d 结构体                    │
│  ggml-quants.c:   实现 dequantize 函数                      │
│  ggml.c:          注册 type_traits                          │
│  ggml-cpu/:       实现 vec_dot (Q4_D/Q5_D 转发现有实现)     │
│                                                             │
│  推理: value = q_int * d + m  (与GPTQ结果完全一致)          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Phase 1: Python 端 — GPTQ 量化数据保存 (DartMQ 适配)

#### 3.2.1 关键挑战：参数收集
`DartMQ/gptq_utils.py` 中的 `fasterquant` 采用动态计算 `scale/zero` 的方式。一旦循环进入下一组，上一组的参数会被覆盖。因此，**必须在循环内部收集参数**。

#### 3.2.2 修改 gptq_utils.py 的 fasterquant 方法

**1. 初始化容器 (在循环开始前)**
```python
# [新增] 初始化收集容器
all_scales = []
all_zeros = []
```

**2. 循环内收集 (在 find_params 之后)**
在 `if groupsize != -1:` 分支内，当参数被更新时立即记录：
```python
# 约第 235 行附近
if groupsize != -1:
    if not static_groups:
        if (i1 + i) % groupsize == 0:
            end_idx = min((i1 + i + groupsize), self.columns)
            self.quantizer.find_params(W[:, (i1 + i):end_idx], weight=True)
            
            # [新增] 收集当前组参数
            if hasattr(self, '_export_gptq_data') and self._export_gptq_data:
                all_scales.append(self.quantizer.scale.clone())
                all_zeros.append(self.quantizer.zero.clone())
```

**3. 末尾导出逻辑 (在 actorder 恢复之后)**
在 `fasterquant` 结束前（约第 283 行之后）：

```python
# ========== 新增：导出 GPTQ 原始量化数据 ==========
if groupsize > 0 and hasattr(self, '_export_gptq_data') and self._export_gptq_data:
    import os
    
    # 获取量化位宽
    bits = int(torch.log2(self.quantizer.maxq + 1).item())
    maxq = int(self.quantizer.maxq.item())
    
    # 3.1 合并 scale/zero
    # 注意：all_scales 是 list of (out, 1, 1, 1)，需处理维度
    all_scale_tensor = torch.cat([s.flatten(1) for s in all_scales], dim=1)  # (out, num_groups)
    all_zero_tensor = torch.cat([z.flatten(1) for z in all_zeros], dim=1)
    
    # 3.2 恢复 q_int (利用反量化后的 Q 和原始 scale/zero)
    # Q 是已经反量化回来的 FP16 权重: Q = scale * (q - zero)
    # 逆运算: q = round(Q / scale + zero)
    
    export_q_ints = []
    num_groups = all_scale_tensor.shape[1]
    
    for g_idx in range(num_groups):
        gi = g_idx * groupsize
        ge = min(gi + groupsize, self.columns)
        
        group_q = Q[:, gi:ge]  # 反量化后的值
        g_scale = all_scale_tensor[:, g_idx:g_idx+1]
        g_zero = all_zero_tensor[:, g_idx:g_idx+1]
        
        # 核心：使用原始参数恢复整数索引，保证 0 误差
        q_int = torch.clamp(
            torch.round(group_q / g_scale + g_zero), 0, maxq
        ).to(torch.uint8)
        
        export_q_ints.append(q_int)
        
    all_q_int = torch.cat(export_q_ints, dim=1)
    
    # 3.3 保存
    export_dir = "gptq_export"
    os.makedirs(export_dir, exist_ok=True)
    safe_name = name.replace('.', '_').replace('/', '_')
    torch.save({
        'name': name,
        'shape': tuple(W.shape),
        'scale': all_scale_tensor.cpu().to(torch.float16),
        'zero': all_zero_tensor.cpu().to(torch.float16),
        'q_int': all_q_int.cpu().to(torch.uint8),
        'groupsize': groupsize,
        'bits': bits,
    }, f"{export_dir}/{safe_name}.pt")
    
    print(f"[GPTQ Export] Saved {name} (bits={bits})")
# ==========================================================
```

#### 3.2.3 启用导出

在调用 `fasterquant` 之前设置标志：

```python
gptq_obj._export_gptq_data = True
```

#### 3.2.4 .pt 文件格式规范

每个 .pt 文件包含一个 dict：

```python
{
    'name': str,                    # HF 张量名, 如 "model.layers.0.self_attn.q_proj.weight"
    'shape': tuple,                 # 原始形状 (out_features, in_features)
    'scale': Tensor[float16],       # (out_features, num_groups)
    'zero': Tensor[float16],        # (out_features, num_groups)
    'q_int': Tensor[uint8],         # (out_features, in_features), 值域 [0, 2^bits-1]
    'groupsize': int,               # 组大小，通常 128 (DartMQ 默认)
    'bits': int,                    # 量化位数 (3/4/5/6)
}
```

### 3.3 Phase 2: GGUF 转换 (混合精度支持)

DartMQ 的一个核心特性是**动态混合精度**（不同层甚至不同算子使用不同的 bit）。由于 `.pt` 文件中已保存了 `bits` 字段，转换脚本可以自动识别并使用对应的打包函数。

#### 3.3.1 gguf-py 类型注册

**文件: `gguf-py/gguf/constants.py`**

在 `GGMLQuantizationType` 枚举中添加:

```python
class GGMLQuantizationType(IntEnum):
    # ... 现有类型 ...
    Q3_D  = 40    # DartMQ 3-bit
    Q4_D  = 41    # DartMQ 4-bit
    Q5_D  = 42    # DartMQ 5-bit
    Q6_D  = 43    # DartMQ 6-bit
```

#### 3.3.2 Python 打包函数 (gptq_pack.py)

需实现 `pack_q3_d`, `pack_q4_d`, `pack_q5_d`, `pack_q6_d`。具体实现逻辑见前文 `gptq_pack.py` 代码。

#### 3.3.3 修改 convert_hf_to_gguf.py

支持自动混合精度打包：

```python
def process_tensor_with_gptq(self, name, tensor_data, gptq_dir):
    pt_path = os.path.join(gptq_dir, f"{safe_name}.pt")
    if not os.path.exists(pt_path): return None
    
    gptq = torch.load(pt_path)
    bits = gptq['bits']  # 自动获取该张量的 bit 数
    
    # 动态选择打包函数
    pack_fn = PACK_FUNCTIONS[bits]
    packed_data, _ = pack_fn(...)
    
    # 动态选择 GGUF 类型
    dtype_map = {3: Q3_D, 4: Q4_D, 5: Q5_D, 6: Q6_D}
    
    self.gguf_writer.add_tensor(
        name=mapped_name,
        tensor=packed_data,
        raw_dtype=dtype_map[bits]  # 自动适配
    )
```

### 3.4 Phase 3: llama.cpp C++ 端

#### 3.4.1 类型枚举注册

**文件: `ggml/include/ggml.h`**

```c
enum ggml_type {
    // ... 现有类型 ...
    GGML_TYPE_Q3_D    = 40,
    GGML_TYPE_Q4_D    = 41,
    GGML_TYPE_Q5_D    = 42,
    GGML_TYPE_Q6_D    = 43,
};
```

#### 3.4.2 Block 结构体定义

**文件: `ggml/src/ggml-common.h`**

```c
// ============ Q*_D: DartMQ Types ============
// 具体结构见 2.3 节
```

#### 3.4.3 反量化函数与 SIMD

- **Q4_D / Q5_D**: 直接复用 Q4_1 / Q5_1 的 SIMD 内核（零工作量）。
- **Q3_D / Q6_D**: 需实现新的反量化逻辑和 SIMD 优化（可暂用标量实现跑通流程）。

---

## 4. 实施路线

### Phase 1: 端到端验证（仅 Q4_D，最小改动）

**目标**: 验证 DartMQ 默认配置 (128 groupsize, 4-bit) 的导出链路。

1. [ ] 修改 `DartMQ/gptq_utils.py`: 实现参数收集与 `.pt` 导出。
2. [ ] 编写 `DartMQ/gptq_pack.py`: 实现 `pack_q4_d`。
3. [ ] 修改 `convert_hf_to_gguf.py`: 加载 `.pt` 并以 `Q4_1` 类型写入（Q4_D 兼容 Q4_1）。
4. [ ] 测试: llama.cpp 加载 → 推理 PPL。

### Phase 2: 建立完整 Q*_D 体系

1. [ ] 注册 Q3_D/Q5_D/Q6_D 类型。
2. [ ] 实现 C++ 端反量化逻辑。
3. [ ] 验证混合精度模型（如部分层 3-bit，部分 5-bit）的加载与推理。

---

## 5. 风险与注意事项

### 5.1 Groupsize 128 适配
DartMQ 默认 `groupsize=128`，而 block 大小固定为 32。
这意味着每 4 个 block (4 * 32 = 128) 共享同一个 scale/zero。
打包函数 `gptq_pack.py` 中已包含 `blocks_per_group` 逻辑来处理此情况，确保 scale/zero 被正确复制/广播。

### 5.2 混合精度对齐
DartMQ 可能对同一层的不同算子使用不同 bit（如 `v_proj` 用 4-bit，`down_proj` 用 6-bit）。
本方案完全支持此特性，因为每个 tensor 独立保存 `.pt` 文件，且包含自己的 `bits` 元数据。GGUF 格式天然支持异构精度的 tensor 存储。

### 5.3 显存占用
导出过程中需要保存所有参数，显存占用会略有增加。但由于是逐层处理（sequential），且 `.pt` 保存后即释放显存，整体影响可控。

---

## 6. 实现附录（GPU 友好 + Q/K 语义修复）

本节记录了已在代码中实现的内容，重点关注：
- `Q3_D/Q4_D/Q5_D/Q6_D` 的 CUDA 后端友好性
- Llama Q/K 排列的 GPTQ 导出转换正确性

### 6.1 CUDA 后端：GPU 友好支持策略

设计目标：
- 保持修改最小化且安全
- 在二进制兼容时复用现有优化的 CUDA 路径
- 避免将不支持的类型路由到 MMVQ/MMQ 内核

已实现的行为：

1. `Q4_D/Q5_D` 快速路径复用
- `Q4_D` 被别名为 `Q4_1` CUDA MMVQ/MMQ 行为
- `Q5_D` 被别名为 `Q5_1` CUDA MMVQ/MMQ 行为
- 文件：
  - `ggml/src/ggml-cuda/mmvq.cu`
  - `ggml/src/ggml-cuda/mmq.cu`
  - `ggml/src/ggml-cuda/mmq.cuh`

2. `Q3_D/Q6_D` GPU 反量化 + cuBLAS 路径
- 为 `Q3_D` 和 `Q6_D` 添加了 CUDA 反量化内核
- 在所有相关转换器入口点添加了转换路由：
  - `to_fp16`
  - `to_fp32`
  - `to_fp16_nc`
  - `to_bf16_nc`
  - `to_fp32_nc`
- 文件：
  - `ggml/src/ggml-cuda/dequantize.cuh`
  - `ggml/src/ggml-cuda/convert.cu`

3. MMVQ/MMQ 调度安全门控
- 在 CUDA 后端添加了类型白名单辅助函数（`ggml_cuda_type_supports_mmvq`）
- 阻止 `Q3_D/Q6_D` 进入不支持的 MMVQ/MMQ 分支
- 确保不支持快速内核的类型回退到 CUDA 反量化 + cuBLAS 路径
- 文件：
  - `ggml/src/ggml-cuda/ggml-cuda.cu`

4. 后端算子兼容性更新
- 为 CUDA 算子能力检查添加了 `Q*_D` 支持条目（`MUL_MAT` 相关路径 + `CPY` F32<->Q*_D 兼容性）
- 文件：
  - `ggml/src/ggml-cuda/ggml-cuda.cu`

### 6.2 Q/K 排列语义修复（对 PPL 至关重要）

问题：
- GPTQ 导出直接写入路径原本绕过了 Llama 特定的 Q/K 排列语义
- 即使模型正确加载，这也可能产生极端的 PPL 飙升（例如数百）

修复：

1. 添加了通用的 GPTQ 条目转换钩子
- 新钩子方法：`transform_gptq_export_entry(...)`
- 在 GPTQ 打包/写入之前调用
- 文件：
  - `convert_hf_to_gguf.py`

2. 为 Q/K 张量实现了 Llama 覆盖
- 对于 `q_proj.weight` 和 `k_proj.weight`，对以下内容应用相同的 Llama 排列：
  - `scale`
  - `zero`
  - `q_int`
- 使用现有的 `LlamaModel.permute(...)` 语义，配合正确的 head / kv-head 设置
- 文件：
  - `convert_hf_to_gguf.py`

结果：
- GPTQ 直接导出转换路径现在与基准 Llama 张量语义匹配
- 这解决了已知的 Q/K 语义不匹配类错误，并恢复了预期的困惑度行为

### 6.3 已知的非阻塞日志消息

加载 `Q*_D` 模型时，可能会出现如下日志行：
- `llama_model_loader: unknown type q6_d`

含义：
- 此警告来自文件类型猜测 switch 覆盖范围
- 它**不**意味着 Q6_D 张量加载失败
- 实际的张量统计（例如 `type q6_d: ... tensors`）是权威信号
