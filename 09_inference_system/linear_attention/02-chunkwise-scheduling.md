# chunkwise 串行 prefill 的调度：两个「串行」要分清

> [总览](linear-attention-serving.md) 说线性层的 prefill 是本质串行的，[KDA 机制拆解](01-kda-mechanism.md) 讲了串行的来源（chunk 间只传状态）。这篇回答工程问题：引擎的调度器怎么处理这种串行？「chunk 间隙插 decode」到底是已解决的问题，还是缺原语？
>
> 2026-09 | 基于 vLLM（`43d691ec6b`）与 SGLang（`f7101b0ae6`）源码验证

---

## 一、先分清两个「串行」

讨论调度之前，必须把两个被混用的「串行」拆开：

**模型级串行**。KDA 的 chunkwise 递推：一个请求的 1M token 输入被模型切成 64-token 的 chunk，chunk N 必须等 chunk N-1 的状态算完。这一步是模型架构强制的，kernel 内部顺序执行，调度器改变不了。

**调度级切块**。引擎为了让长 prefill 让位给交互请求，把 prefill 切成多个调度单元、分多轮完成。这一步是调度器自选的，切在哪、切多大都由引擎决定。

普通 attention 模型只有第二种串行，所以「chunk 间隙插 decode」是已解决的老问题：vLLM V1 用统一的 token budget 轮转（`scheduler.py` 的 `token_budget = max_num_scheduled_tokens`，prefill 与 decode 在同一 budget 内按 token 数竞争，chunk 化 prefill 让 decode 插进轮间），SGLang 用 `chunked_req` 单例（一次只调度一个被切请求，每轮 chunk 算完 stash 出 batch，见下）。

KDA 模型的 prefill 同时有这两种串行，且它们叠加：**调度器每轮放行的是一段输入，这一段输入在 kernel 内部还要走完自己的模型级串行步**。调度切块解决的是「decode 的饥饿」，模型级串行解决的是「这一段输入的 TTFT」。两者互不替代。

## 二、现状：引擎已经做了什么

### 2.1 SGLang：chunked_req 单例 + 检查点粒度强制对齐

SGLang 长 prefill 的处理（`scheduler.py`）：

- 一个超长请求被选出为 `chunked_req`（`scheduler.py:1178` 初始化，全局单例，同一时刻只有一个请求处于被切分状态）；
- 每轮调度放行一段（长度由 `chunked_prefill_size` 控制，`scheduler.py:1160` 起初始化），请求的 `extend_range` 记录已算进度；
- 这一段算完，请求被 `stash_chunked_request`（`scheduler.py:2925`）移出 batch，把位置让给 decode 和其他 prefill；`extend_range.end > len(prefix_indices)`（`:3051`）判断本段是否产生了新 KV，避免重复缓存；
- 下一轮从 stash 取回继续。

对 KDA 请求，SGLang 额外做了一件关键的事：**强制调度切块对齐模型 chunk 边界**（`server_args.py` 的 `mamba_cache_chunk_size` 属性）：

```python
chunk_size = getattr(hf_config, "mamba_chunk_size", FLA_CHUNK_SIZE)  # 默认 64
self._mamba_cache_chunk_size = max(chunk_size, page_size)
assert max(chunk_size, page_size) % min(chunk_size, page_size) == 0
```

检查点粒度（模型 chunk）与页大小必须整除，取两者的较大值作为缓存检查点位置。这等于承认：KDA 请求的调度切块不能随意落在任意 token 上，必须落在状态检查点能对齐的位置。

另一个细节：`req.init_next_round_input(self.tree_cache, cow_mamba=False)`（`scheduler.py:2695`）。KV 可以做 radix tree 的 copy-on-write，mamba 状态续算时显式关闭 COW，状态不能像 KV 那样被多请求共享写。

### 2.2 vLLM：budget 轮转天然插 decode

vLLM V1 没有显式的「单一切块请求」概念，靠 token budget 自然工作：每轮调度在 `running`（decode）与 `waiting`（prefill）之间分配 `max_num_scheduled_tokens` 个 token，prefill 超 budget 的部分切成 `is_prefill_chunk` 留下轮（约束解码篇见过这个字段），decode 永远优先占 budget。长 prefill 因此被摊到很多轮里，每轮之间 decode 自由插队。

对 KDA 请求，vLLM 的模型实现把整段（或调度放行的段）交给 mamba/linear kernel，kernel 内部处理模型级串行；状态经 `MambaSpec` 纳入 KV 管理，检查点模式由 `mamba_cache_mode` 控制（见[总览 §2.2](linear-attention-serving.md)）。

### 2.3 小结：能插 decode，但插的不是模型串行步

两引擎都证明了「让 decode 插进长 prefill 的轮间」是成熟能力。但这解决的是调度级切块的间隙，KDA 的模型级串行步（64 token 一步）在 kernel 内部是连续执行的，decode 插不进去。**「chunk 间隙插 decode」如果指轮间，已解决；如果指 64-token 步间，属于 kernel 内部流水线问题，调度器管不到。**

## 三、KDA 让调度变形的三点

把模型级串行叠进调度模型后，三个新问题浮现：

**1. 切块点被检查点粒度约束**。普通 attention 的 KV 前缀命中是任意粒度的，调度器切块只要页对齐；KDA 的续算必须从检查点出发，切块点最好落在检查点位置，否则一段输入内部要从上一个检查点重放。SGLang 用 max(chunk, page) 且要求整除来强制对齐，代价是调度自由度变小：切块大小不能自由选，得是 64 的倍数且适配页大小。

**2. TTFT 下限不随调度改善**。一个 1M 输入的 KDA 请求，模型级串行 15,625 步，每步延迟由 kernel 决定。调度器把这段输入切成 100 轮放行，decode 是舒服了，但这个请求的 TTFT 反而更长了（还要加上轮间等待）。调度器面临一个普通 attention 没有的取舍：**长 prefill 优先完成（占 batch、拖 decode）vs 平均分配（decode 舒服、TTFT 拖到不可接受）**。

**3. 状态占位无法共享**。KV 前缀命中后多请求共享同一份 KV；KDA 的状态检查点理论上可共享（同前缀分叉点），但活跃请求续算时的状态是独占的（SGLang 显式关掉 COW）。调度器看到的 KDA 请求不是「吃多少 KV」，而是「占一份不可共享的状态 + 一串必须串行完成的步」。

## 四、设计空间：调度器需要什么

基于源码现状，三个值得推进的方向：

**1. 串行感知的预算分配**。现有预算按 token 数计（vLLM）或按 chunk 数计（SGLang），都不反映 KDA 请求的真实耗时：一个 64-token 的 KDA chunk 要串行走完内部步，和 64 个 decode token 的耗时不在一个量级。调度器需要给 KDA prefill 一个「串行权重」，否则等 token 计数的预算公平性失真。

**2. 长 prefill 的完成时限**。KDA 长请求的 TTFT 由「轮数 × 每轮放行量」决定，调度器若永远优先 decode，1M 请求可能永远算不完。需要区分「可抢占的轮间等待」与「完成时限承诺」，这是对[总览 §五 缺口 2](linear-attention-serving.md)（状态感知准入）的调度侧补充。

**3. 检查点感知的切块**。切块点对齐检查点粒度（SGLang 已做），更进一步是切块点感知「已有哪些检查点」：请求续算时若某段输入的状态检查点已在缓存里，这一段可以直接跳过（前缀命中在状态域的对应物）。这块落到[总览 §2.2](linear-attention-serving.md) 的状态 radix cache 与检查点池，调度器要能消费这些命中的段。

## 五、待实测

挂 InferenceX 排期（同机）：1M 输入的 KDA 请求在「每轮放行量」梯度下的 TTFT 曲线与 decode 干扰曲线，验证第 2 点的取舍是否如预期陡峭，以及检查点对齐切块相对任意切块的实际收益。

---

## 结语

回到标题的问题：两个「串行」中，调度级切块和它的 decode 插入是两引擎早已解决的工程问题（SGLang 的 chunked_req、vLLM 的 budget 轮转），KDA 真正引入的是模型级串行叠加在其上。调度器为普通 attention 设计的预算、优先级、切块自由度，遇到「不可拆分的 64-token 步 + 不可共享的状态」时各自失真。上一篇文章说线性注意力的调度要「理解串行」，这篇把「理解」拆成了三件具体的事：给串行一个权重、给长 prefill 一个时限、给检查点一个对齐。

## 源文件索引

| 文件                                      | 关键符号                                                                                           | 引用点                          |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------- |
| `python/sglang/srt/managers/scheduler.py` | `chunked_req`、`stash_chunked_request`、`extend_range`、`init_next_round_input`（cow_mamba=False） | :1178、:2925、:3041-3052、:2695 |
| `python/sglang/srt/server_args.py`        | `mamba_cache_chunk_size`（max(chunk,page) + 整除断言）                                             | :9092-9110                      |
| `vllm/v1/core/sched/scheduler.py`         | `token_budget`、prefill chunk 轮转                                                                 | :460、:915-922                  |
| `vllm/config/scheduler.py`                | `enable_chunked_prefill`                                                                           | :74                             |

_版本：vLLM `43d691ec6b`（2026-08-07）、SGLang `f7101b0ae6`（2026-08-18）。_
