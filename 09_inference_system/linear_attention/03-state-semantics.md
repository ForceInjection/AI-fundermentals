# 状态语义深水区：rollback、fork、重放一致性、TP 状态

> KV 是只读快照：拷贝、复用、丢弃都不会错，任何时刻从任意位置续算结果都一样。KDA 的循环状态是「会变的寄存器」：投机解码要回滚它、beam search 要分叉它、前缀缓存要核对重放它、张量并行要切分它。KV 时代这些操作都有现成语义，状态域里每一样都要重新定义。
>
> 2026-09 | 基于 vLLM（`43d691ec6b`）与 SGLang（`f7101b0ae6`）源码验证；本系列第三篇（[机制](01-kda-mechanism.md)、[调度](02-chunkwise-scheduling.md)）

---

## 一、状态的三个性质

把 KDA 状态与 KV 对比，三个差异决定了后续所有语义问题：

**不可分割**。KV 是 token 序列，可以部分命中、部分逐出、按 token 续算。状态是固定大小的矩阵（每头 d_k×d_v），要么整体在、要么整体不在，没有「半个状态」。

**会变且要回滚**。KV 是前向计算的副产品，追加永远安全；状态参与后续计算，写错了要能撤销。投机解码验证失败、beam 分支被剪、请求被抢占重排，都需要把状态精确恢复到某个历史点。

**按头本地化**。KV 的 TP 语义复杂（MLA 的 latent KV 要跨卡共享或复制）；KDA 的状态按 head 切分后，注意力计算是纯本地的：每个 attention 头独立更新自己的状态，不需要跨卡交换（见第五节）。

## 二、投机解码：状态的回滚是拷贝，不是重算

投机解码在 KDA 模型上的基本流程：draft 模型草拟 K 个 token，target 模型一次验证。验证失败时，KDA 状态必须从「验证点」重新出发，但**状态回滚不是重算而是拷贝**：验证点之后的所有状态写入被丢弃，把验证点的状态复制回来即可。

两引擎的实现印证了「拷贝」这个本质：

**SGLang**：spec decode 的 draft worker 要为每个草稿候选维护一份状态，一份新请求或 radix COW 事件会让「清除/拷贝 conv-state 池槽位」的操作在每个 draft head 的 pool 上重放一遍。这个重放循环原本是多个 launch-bound 的小 kernel（每个 conv tensor 一次），SGLang 用 `mamba_slot_fused`（`mem_cache/mamba_slot_fused.py`）把它们折成单次 launch：传入每个 tensor 的指针、stride、特征长度数组，一次清/拷全部 conv 槽位。draft 场景是这套 fused kernel 存在的直接动机（文件头部注释自述）。temporal state（形状不同）由调用方用普通 indexed op 处理。

**vLLM**：`MambaSpecDecodeGPUContext`（`worker/mamba_utils.py:481`）是 spec decode + hybrid + align mode 下 fused postprocess 路径的 GPU 侧上下文。它预先计算每个状态张量的基址、块 stride、元素大小、conv 宽度，让 GPU kernel 直接做状态拷贝而无需 CPU-GPU 同步；conv state（滑窗，offset 拷贝）与 temporal state（整块拷贝）分类型处理。`num_accepted_tokens` 等结果直接写回 GPU 缓冲。

两条实现都指向同一结论：**状态回滚在引擎里被实现为「精确的块拷贝」**。代价不再是重算（省了算力），而是拷贝的编排复杂度（每个状态的布局元数据、分类型拷贝、多 draft 头扇出），这正是两引擎各自写一个专用上下文/专用 kernel 的原因。

## 三、分叉与共享：状态为什么关掉 COW

beam search、并行分支、树状探索都会让多个候选从同一前缀分叉。KV 域的做法是 radix tree + copy-on-write：分叉点之前的 KV 多候选共享，各自的增量分叉写。

状态域的问题在于：续算一个候选会**原地更新**状态（S_t = f(S_{t-1})），共享的状态矩阵会被第一个写入的候选污染，必须为每个候选复制一份。SGLang 的续算路径显式关掉了状态侧的 COW（`scheduler.py:2695` `init_next_round_input(..., cow_mamba=False)`），KV 可以 COW，mamba 状态不能。这带来两个后果：

1. 分叉成本是**全状态复制**：每个候选一份固定大小的状态，分支越多复制越贵。好在状态大小与序列长度无关（每头 128×128），复制成本是常数，不随上下文增长；
2. 前缀命中在状态域的形态因此不同：KV 命中省的是「重算 + 显存」，状态检查点命中省的是「从检查点重放」，而分叉点之后的每个候选仍要独立持有一份状态。

调度器看到的图景：KDA 请求的并发成本不是「共享前缀 + 独占总和」的 KV 模型，而是「前缀共享检查点 + 每候选独占状态」的固定租金模型。这与[总览 §2.3](linear-attention-serving.md) 的「状态池硬顶」互相印证。

## 四、重放一致性：一条源码注释留下的隐患

前缀缓存的核心承诺：命中后跳过重算，结果与未命中时一致。KV 域这个承诺几乎免费（同样的块同样的结果）；状态域的重放一致性是个必须核对的问题：**从检查点续算得到的输出，与原始一路算下来的输出，是否逐位一致？**

SGLang 的 `kimi_linear.py` 里有一条注释暴露了隐患的源头（forward 内，`kimi_linear.py:405-411` 附近）：

```python
# Prefill passes raw gates to chunk KDA; decode and target-verify kernels
# apply the activation internally.
```

prefill（chunk kernel）与 decode/verify（recurrent kernel）对门控的激活位置不同：prefill 路径先把 β sigmoid 化再交给 chunk kernel，decode 路径由 kernel 内部激活。同一段序列，用 chunk kernel 算和用 recurrent kernel 算，浮点累积路径不同，结果在小数位上有差异的可能存在。对自回归生成，这个差异会被后续 token 放大，**重放一段历史的输出，可能与原始输出不完全一致**。

这不是说引擎有 bug：检查点的设计目标本来就是近似续算而非逐位复现。但部署方要意识到：状态域的前缀命中是有语义代价的，命中率与输出一致性的权衡是新的调参维度（高命中率场景若对输出复现敏感，需要核对检查点间隔与 kernel 选择）。这个方向值得实测验证（见文末待办），目前源码层能确认的是：两种 forward 模式确实走了不同的激活路径。

## 五、TP 状态：按头切分，但「零通信」要划清边界

KDA 的 TP 切分与全注意力有本质差异。看 SGLang 的投影布局（`kimi_linear.py`）：

- q/k/v 与 β 走 column parallel（按 head 切，`qkvb_sizes` 中 `projection_size` 与 `num_heads // tp_size`）；
- 门控低秩投影 f_a/g_a 走 replicated（每卡全算，`fg_sizes = [head_dim, head_dim]`，`split_sizes` 里 `2 * head_dim` 不分卡）；
- 每个 attention head 独立维护自己的状态，head 级切分后本卡只算本卡的 head（`_get_kda_local_num_heads`，`kimi_linear.py:56`）。

需要划清的是「无通信」的边界：**本地的是注意力计算本身**（状态更新与 chunk 内 attention 都在本卡 head 上完成，不交换 KV 或状态）；**层的输出投影不是**——`o_proj` 是 `RowParallelLinear`（`kimi_linear.py:324`），forward 末尾 `return self.o_proj(core_attn_out)[0]`（:431）仍有一次 all-reduce，与全注意力层相同。所以 KDA 层在 TP 下的**层间通信模式与其他层并无差别**，差别在注意力计算内部：全注意力的计算要处理跨卡可见的 KV（MLA 的 latent 在 TP 下冗余复制），KDA 的状态天然按 head 切开、零共享。

这也让「TP 不能缩短串行步数」的因果关系更清楚（[总览 §2.1](linear-attention-serving.md)）：步数 = ⌈序列长度 / chunk 大小⌉，与并行度无关，加卡只缩短每步延迟。而 KDA 每步的加速空间受 head 数约束：head 是天然的切分维度，TP 超过 head 数后这一维度用尽，进一步的并行只能走 head_dim 切分或其他策略——这与 MLA 的对照点不同（MLA 的瓶颈是 latent KV 在 TP 下冗余复制，而非切分维度耗尽，见 [MLA 的 TP 切分分析](../vllm/module_analysis/mla_tp_kv_redundancy.md)）。

## 六、结语

把四个语义问题连起来：状态的不可分割性让分叉与命中都变成「整块操作」，会变性让回滚变成「精确拷贝工程」，激活路径的分裂让重放一致性成为需要核对的调参维度，按头本地化让 TP 的注意力计算无跨卡依赖（但层的通信模式不变）。引擎的适配代码（`MambaSpecDecodeGPUContext`、`mamba_slot_fused`、cow_mamba=False）已经为前两件事交了学费，后两件事还是开放问题。KV 时代「缓存是安全的」这个默认信念，在状态域每一条都要重新验证。

## 待实测（挂 InferenceX 排期）

同一多轮 trace 在「检查点开/关」下的输出一致性比对：chunk kernel 与 recurrent kernel 的数值差、检查点续算 vs 原始计算的首个分歧 token 位置分布。

## 源文件索引

| 文件                                              | 关键符号                                                            | 引用点                          |
| ------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------- |
| `vllm/v1/worker/mamba_utils.py`                   | `MambaSpecDecodeGPUContext`                                         | :481 起（GPU 侧状态拷贝元数据） |
| `vllm/v1/core/kv_cache_manager.py`                | mamba align 组                                                      | :320、:853                      |
| `python/sglang/srt/mem_cache/mamba_slot_fused.py` | fused conv-slot 清/拷 kernel                                        | 全文（draft worker 扇出动机）   |
| `python/sglang/srt/managers/scheduler.py`         | `init_next_round_input(cow_mamba=False)`                            | :2695                           |
| `python/sglang/srt/models/kimi_linear.py`         | `_get_kda_local_num_heads`、fused 投影布局、`o_proj`（RowParallelLinear）、prefill/decode 激活分支 | :56、:324、:349、:364、:405-411、:431 |
| `python/sglang/srt/mem_cache/memory_pool.py`      | `MambaSlotAllocator`、`free_mamba_cache`                            | :1237、:1477                    |

_版本：vLLM `43d691ec6b`（2026-08-07）、SGLang `f7101b0ae6`（2026-08-18）。_
