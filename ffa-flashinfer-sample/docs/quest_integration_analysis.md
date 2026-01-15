# Quest 与 FlashInfer 集成逻辑分析

## 1. 概述

Quest 是一个查询感知的稀疏注意力库，它通过巧妙地集成 FlashInfer 来实现高效的长上下文 LLM 推理。Quest 的核心思想是：**先用低成本的方法筛选出重要的 KV 页面，再用 FlashInfer 的高性能 kernel 只计算选中的页面**。

## 2. 整体架构

Quest 采用三层架构与 FlashInfer 集成：

```
┌─────────────────────────────────────────────────────────────┐
│                    Python API 层                             │
│  quest/utils/__init__.py                                     │
│  - prefill_forward()  调用 FlashInfer Prefill               │
│  - decode_estimate()  调用自定义估计 kernel                  │
│  - decode_topk()      调用自定义 TopK kernel                 │
│  - decode_sparse_attn() 调用 FlashInfer BatchDecode          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   PyBind11 绑定层                            │
│  quest/ops/csrc/bsk_ops.cu                                   │
│  - PYBIND11_MODULE 定义所有 Python 可调用的函数              │
│  - BatchDecodeWithPagedKVCachePyTorchWrapper 类封装          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CUDA Kernel 层                            │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │    Quest 自定义      │    │      FlashInfer 核心        │ │
│  │  - estimate.cu      │    │  - decode_handler.cuh       │ │
│  │  - topk.cu          │    │  - prefill.cuh              │ │
│  │  - page.cu          │    │  - BatchDecodeWithPaged...  │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 3. 关键文件说明

### 3.1 FlashInfer 依赖位置

```
quest/kernels/3rdparty/flashinfer/
└── include/
    └── flashinfer/
        ├── attention/
        │   ├── decode.cuh        # BatchDecode 核心实现
        │   └── prefill.cuh       # Prefill 核心实现
        ├── page.cuh              # 分页 KV 缓存数据结构
        └── handler.cuh           # BatchDecodeHandler
```

### 3.2 Quest 自定义 Kernel

```
quest/kernels/include/
├── decode/
│   ├── decode_handler.cuh    # 封装 FlashInfer 的 BatchDecodeHandler
│   ├── decode_attn.cuh       # 添加 MaxPossibleSample 估计 kernel
│   └── decode_page.cuh       # 页面操作
├── prefill/
│   └── prefill.cuh           # 封装 FlashInfer 的 Prefill
└── topk/
    └── decode_select_k.cuh   # Top-K 选择算法
```

### 3.3 PyBind11 绑定

```
quest/ops/csrc/
├── bsk_ops.h            # C++ 接口声明
├── bsk_ops.cu           # PyBind11 模块定义
├── approx_attn.cu       # BatchDecodeWrapper 实现
├── batch_prefill.cu     # Prefill 封装
├── estimate.cu          # 注意力分数估计
└── topk.cu              # Top-K 筛选
```

## 4. 核心集成逻辑

### 4.1 Prefill 阶段：直接调用 FlashInfer

**文件**: `quest/ops/csrc/batch_prefill.cu`

```cpp
#include "flashinfer/attention/prefill.cuh"

torch::Tensor prefill_with_paged_kv_cache(...) {
    // 直接调用 FlashInfer 的 BatchPrefillWithPagedKVCache
    flashinfer::BatchPrefillWithPagedKVCache<
        PageStorage::kIndices,
        KV_LAYOUT,
        c_type, c_type, int32_t
    >(q_ptr, paged_kv, o_ptr, ...);
}
```

**逻辑**：Prefill 阶段需要处理完整序列，直接使用 FlashInfer 的高性能实现。

### 4.2 Decode 阶段：三步流程

```
Step 1: 估计注意力分数 (Quest 自定义)
           ↓
Step 2: Top-K 筛选 (Quest 自定义)
           ↓
Step 3: 稀疏注意力计算 (FlashInfer)
```

#### Step 1: 估计注意力分数

**文件**: `quest/ops/csrc/estimate.cu`

```cpp
// 使用 FlashInfer 的 MaxPossibleSample kernel
// 从每个页面的 min/max K 值估计注意力分数
flashinfer::MaxPossibleSampleWithPagedKVCache<...>(
    q, paged_kv, o, num_heads, RotaryMode::kNone
);
```

**逻辑**：对每个 KV 页面，使用预存的 min/max K 值快速估计该页面的注意力分数。

#### Step 2: Top-K 筛选

**文件**: `quest/ops/csrc/topk.cu`

```cpp
// 使用 RAFT 库的 radix_topk 算法
raft::matrix::detail::select::radix::select_k<...>(
    handle, in, in_idx, n_rows, n_cols, k, out, out_idx, ...
);
```

**逻辑**：选择估计分数最高的 K 个页面。

#### Step 3: 稀疏注意力计算

**文件**: `quest/ops/csrc/approx_attn.cu`

```cpp
class BatchDecodeWithPagedKVCachePyTorchWrapper {
    flashinfer::BatchDecodeHandler handler_;  // FlashInfer 处理器

    void Forward(...) {
        // 构建只包含选中页面的 paged_kv
        paged_kv_t<...> paged_kv(
            num_kv_heads, page_size, head_dim, batch_size,
            page_budget,  // 只选中的页面数量
            paged_kv_last_page_len, paged_kv_last_page_idx,
            paged_kv_data,
            paged_kv_indices,  // Top-K 选中的页面索引
            paged_kv_indptr
        );

        // 调用 FlashInfer 的 BatchDecode
        flashinfer::BatchDecodeWithPagedKVCacheWrapper<...>(
            &handler_, q, paged_kv, o, ...
        );
    }
};
```

**核心技巧**：通过修改 `paged_kv_indices` 只包含 Top-K 选中的页面，让 FlashInfer 只计算这些页面的注意力。

## 5. CMake 编译配置

**文件**: `quest/ops/CMakeLists.txt`

```cmake
# 关键配置
target_include_directories(_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/../../kernels/include          # Quest 自定义 kernel
    ${CMAKE_SOURCE_DIR}/../../kernels/3rdparty/flashinfer/include)  # FlashInfer

# 编译选项
target_compile_options(_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)

# 链接库
target_link_libraries(_kernels PRIVATE
    ${TORCH_LIBRARIES}
    raft::raft          # 用于 Top-K
    Python::Python
    pybind11::module)
```

## 6. Python API 层

**文件**: `quest/utils/__init__.py`

```python
import quest._kernels as _kernels  # 导入编译的 C++ 模块

# 封装 Prefill
def prefill_forward(q, iController, layer_idx, ...):
    return _kernels.prefill_with_paged_kv_cache(
        q, iController.kv_cache.buf_layer(layer_idx), ...
    )

# 封装 Decode 三步流程
def decode_estimate(q, iController, layer_idx):
    # 调用估计 kernel
    _kernels.estimate_attn_score(q, o, metadata_cache, ...)
    return o

def decode_topk(estimated_attn_score, iController):
    # 调用 Top-K kernel
    _kernels.topk_filtering(estimated_attn_score, ...)

def decode_sparse_attn(q, iController, layer_idx, topk_indices, ...):
    # 调用 FlashInfer BatchDecode，只处理选中的页面
    iController._decode_handler.forward(
        q, o, kv_cache, topk_indices, ...
    )
    return o
```

## 7. 数据流示意图

```
                    Decode 阶段数据流

    Query (q)
        │
        ▼
┌───────────────────┐     ┌────────────────────┐
│  Metadata Cache   │────▶│   estimate_attn    │
│  (min/max K 值)   │     │   (Quest 自定义)    │
└───────────────────┘     └────────────────────┘
                                   │
                                   ▼ estimated_scores
                          ┌────────────────────┐
                          │    topk_filtering   │
                          │   (Quest + RAFT)    │
                          └────────────────────┘
                                   │
                                   ▼ topk_indices
┌───────────────────┐     ┌────────────────────┐
│    KV Cache       │────▶│  BatchDecode       │
│   (完整页面)       │     │   (FlashInfer)     │
└───────────────────┘     └────────────────────┘
                                   │
                                   ▼
                              Output (o)
```

## 8. 关键设计原则

### 8.1 职责分离
- **Quest 负责**: 稀疏性策略（估计、筛选）
- **FlashInfer 负责**: 高性能注意力计算

### 8.2 接口兼容
- 通过 `paged_kv_t` 数据结构与 FlashInfer 通信
- 使用索引数组控制要计算的页面

### 8.3 最小修改原则
- FlashInfer 作为第三方库直接引入，不修改其源码
- 仅在 `decode_handler.cuh` 中添加必要的封装

## 9. 你的项目集成建议

对于你的 `attn_sample4_fp16` 算法，建议采用类似的架构：

```
你的 Triton Kernel (筛选)
    ↓ 输出 kept_indices
FlashInfer Python API (精确计算)
```

由于你的算法是 Triton 实现的，可以直接在 Python 层集成，无需写 C++/CUDA：

```python
# 1. 你的 Triton kernel 做快速筛选
o_sparse, kept_indices = your_triton_stage1(q, k_sample, k_full, v, ...)

# 2. 对于需要精确计算的 block，使用 FlashInfer
o_precise = flashinfer.batch_decode_with_paged_kv_cache(
    q, paged_kv_cache[kept_indices], ...
)

# 3. 合并结果
o = merge_outputs(o_sparse, o_precise, kept_indices)
```

这种方式的优点：
1. 保留 Triton 的开发效率
2. 利用 FlashInfer 的高性能 kernel
3. 无需编译 C++ 代码
