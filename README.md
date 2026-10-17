# llama.cpp

A fork of https://github.com/ggml-org/llama.cpp

# fixds4 分支简介

针对在 **Ampere 系 GPU（CMP 170HX / RTX 3080 / A4000）** 上运行 **DeepSeek-V4-Flash（IQ2_XXS / Q2_K 专家权重）** 的推理场景，为 llama.cpp 的 ggml-cuda 后端加入了一批手写算子快速路径。目前的算子改进实现了 **prefill速度 +45%**，decode速度无可感改进。

（巧的是项目开发期间llama.cpp也有性能优化，大概decode +3%左右、prefill +20%，而且能和我的优化算子效果叠加，所以我直接同步到尽量新的llama.cpp版本了）

目标模型：https://hf-mirror.com/antirez/deepseek-v4-gguf 中的 DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf

优化遵循「**不动原实现、前面加可开关的 gated fast path**」原则：只在全部触发条件满足时走快速路径，否则原样落回通用 kernel，**zero regression risk**。所有快速路径按**功能**命名。

## 集成的优化点

### 1. 统一运行时开关基础设施（`common.cuh` / `ggml-cuda.cu`）

新增一套运行时开关，便于整体 / 单算子 A/B：

- `GGML_CUDA_FAST_OPS` —— 总开关（硬开关），`=0` 全关，即完全 stock 行为；
- `GGML_CUDA_FAST_SOFTMAX`、`GGML_CUDA_FAST_MOE_MM` —— 单算子覆盖；
- `GGML_CUDA_FAST_MOEMM_O1O2`、`GGML_CUDA_FAST_MOEMM_O5` —— MoE MM 的逐优化项子开关。
- 开关状态在 CUDA 后端初始化时打印一次（ggml INFO，默认 verbosity 可见）。

### 2. Softmax 快速路径（`softmax.cu`）

为「连续、f32、scale==1、无 mask、无 sinks、无 ALiBi」的常规 SOFT_MAX 场景（目标模型中会用到的）按列宽分档特化：

- `NARROW8`：ncols==8 专项，一个 warp 并行处理 16 行、每行 2 lane，寄存器驻留 + 向量化；
- `WARP-REG` / `TINY` / `REG` / `HYBRID` / `ONLINE`：覆盖中等到超大列宽的寄存器 / shared-memory / online 归约变体；
- 附决策日志，可确认真实模型实际命中的快速路径与回退原因。

收益集中在 **kernel 级访存带宽占比**（如 ncols=8 在大 nrows 下三卡 14.9× / 17.2× / 21.3×，%copy 89~94%）；端到端上 softmax 不是这几张卡的瓶颈（<0.5%）。

### 3. MoE 量化专家 matmul（`mmq.cu` / `mmq.cuh` / `mmq-load-tiles.cuh`）

针对目标模型的专家设置（每层256个专家，每token激活6个）

MoE MM 的瓶颈是**分层**的，逐层做了三项优化：

- **O1 —— 消除调度层 sync-fallback 悬崖**：

  MoE ids 分支在 token 数 T≥64 时，stock 会掉进一条「调度悬崖」：`mul_mat_id_needs_sync` 返回 true，触发 **2×stream sync + CPU 往返 + 逐 expert 发射 kernel + 禁用 CUDA graph 捕获** 的兜底路径，发射/同步开销巨大。

  O1 让 `mul_mat_id_needs_sync` 对该场景（src1 为 f32、dst 为 f32、src0 为量化类型、`ne[2] ≥ MMQ_DP4A_MAX_BATCH_SIZE`、且 `should_use_mmq` 成立）**直接返回 false**，从而跳过整条兜底路径，改由 `mul_mat_id` 直接发射 mmq kernel（J=8..128 逐档），并保住 CUDA graph 捕获。收益来自砍掉发射/同步开销，与 kernel 本身无关。

- **O2 —— 消除工作形状层 mma 浪费**：

  stock 用「全局 J≥T」发射 mmq：J-tile 必须盖住最大的活跃行数，而每个 expert 实际只分到 col_diff 行，于是每个专家都付出 `J/col_diff` 的 **mma 空转**（时间 ∝ E_act×J）。

  O2 在宿主端按每个 expert 的实际活跃行数（col_diff）把专家**分进 4 个桶**，每桶用「刚好 ≥ 桶内最大 col_diff 的 J」发射（`switch_J` 逐档选 J），把空转压到每桶只多一点点。为让 mmq kernel 保持快速连续加载，y 按桶**重排（gather）**成局部紧凑布局、每 k-block 补 pad=128 填充行，只有权重 expert id 需要间接寻址（expert_ids）。收益集中在**削减每个专家付出的 mma 空转**，是端到端收益的第二个主体。
  
- **O5 —— 消除数据解包层 ALU 瓶颈**：Q2_0 向量化解包，一次 4 字节加载解出 16 个值（原 2 字节解 8 个），shared-memory tile 布局与原 `__byte_perm` 路径逐字节一致，`ldmatrix+mma` 可原样消费。

## 端到端收益（真实模型实测）

- **prefill**：3 卡 KV Q8 `262.54 → 379.9`（**~1.45×**），即提交信息中的 **prefill +45%**（基于开发中的llama.cpp版本基线）。同步新的llama.cpp版本之后，prefill更高一些。
- **decode**：基本持平（±1% 噪声内），MoE MM 只改善 prefill（decode 走 mmvq 路径）。
- 收益主体是 MoE MM 的 O1/O2；softmax 加速端到端无感，O5 单独在当前 IQ2_XXS/Q2_K 模型上也无端到端收益（它覆盖的 Q2_0 不是当前模型类型，但理论提升挺大的所以先保留了）。

## 构建与用法

- 集成目标测试通过**CUDA 13.3**+`CMAKE_CUDA_ARCHITECTURES="80;86"`（没有用到什么高级特性，其他CUDA版本和GPU架构理论上也可以用）。
- A/B：`GGML_CUDA_FAST_OPS=0`（stock）对照 `GGML_CUDA_FAST_<OP>=0`（单算子）。

## 对于新架构的GPU的适配程度分析
- Softmax：纯 thread-level CUDA（warp 每行、寄存器驻留、float4 加载、expf），无架构专属指令，可直接迁移；但收益只在 kernel 级访存带宽，端到端非瓶颈，新卡上大概率依旧无感。
- MoE MM O1：纯宿主端调度优化（跳过 2×sync + CPU 往返 + 逐 expert 发射），架构无关；新卡算力越快这类发射/同步开销占比越大，最可能在新卡上保持甚至放大提速。
- MoE MM O2：分桶削 J/col_diff 的 mma 浪费，机制所有 tensor-core 架构通用；但分桶阈值/J≤128/pad 是 Ampere 调的，新卡要重调参才拿满。
- MoE MM O5：__byte_perm+4 字节加载解 16 值，NVIDIA 全系可移植；但只覆盖 Q2_0，当前模型是 IQ2_XXS/Q2_K，任何架构端到端都无收益。

所以优先推荐复现 **MoE MM O1**，其次愿意做一点测试确认最优参数的话复现 **MoE MM O2**。

## 说明

- 本分支的fork起点已经更新到 llama.cpp 基线 [`2539badcb`](https://github.com/ggml-org/llama.cpp/commit/136887b665180c13c6209a4ce0673637b6cd3afd) & https://github.com/ggml-org/llama.cpp/releases/tag/b11221。
- 改动集中在 `ggml/src/ggml-cuda/`；不涉及 `ggml.h` / `ggml.c` / `src/llama-graph.cpp`。
