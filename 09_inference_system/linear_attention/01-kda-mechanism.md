# KDA 机制拆解：从 DeltaNet 到通道级门控

> [总览](linear-attention-serving.md) 把 delta-rule 线性注意力当黑盒讲系统挑战，本文把这个黑盒打开：KDA（Kimi Delta Attention）相对它的两个前身（DeltaNet、Gated DeltaNet）到底改了什么，为什么这些改动让 K3 级别的模型愿意把 3/4 的注意力层交给它，以及推理引擎侧（以 SGLang 实现为对照）如何体现这些机制。
>
> 2026-09 | 公式与章节引自 Kimi Linear（[arXiv:2510.26692](https://arxiv.org/abs/2510.26692)，v2）；实现对照 SGLang（`f7101b0ae6`）`python/sglang/srt/models/kimi_linear.py`

---

## 一、三行递推的演进

线性注意力的历史可以压缩成三行状态递推：

```text
Linear Attention（无门）:  S_t = S_{t-1} + k_t v_t^⊤
DeltaNet（删旧）:          S_t = (I − β_t k_t k_t^⊤) S_{t-1} + β_t k_t v_t^⊤
Gated DeltaNet（遗忘）:    S_t = α_t (I − β_t k_t k_t^⊤) S_{t-1} + β_t k_t v_t^⊤
```

三行分别对应三次思想跳跃（论文 §2.2）：

1. **线性注意力**把状态 S 当成「快速权重」，每步往里累积键值对 k v^⊤。累积式的问题很直接：旧信息永远不被删除，状态被无关历史污染。
2. **DeltaNet** 换了个解释：状态更新等价于对重建损失 L(S) = ½‖S^⊤k_t − v_t‖² 做一步在线梯度下降。由此得到 Householder 型的删除项 (I − β k k^⊤)：新键进来时，先把状态里与它方向重叠的部分投影掉，再写入新值。β_t 是学习率，控制写入强度。这比「无条件累积」多了一个删旧的动作。
3. **Gated DeltaNet**（GDN）加入标量遗忘门 α_t，给整个状态矩阵一个逐 token 的权重衰减。

KDA 的递推（论文 Eq. 1）是第四行：

$$S_t = (I - \beta_t k_t k_t^\top)\,\text{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top$$

表面只差一个 Diag()：遗忘门从标量变成对角矩阵。这一改牵动的机制比记号看起来大得多。

## 二、通道级门控：细到什么程度

GDN（以及 Mamba-2 系）的遗忘门是**逐头标量**：一个头的整条状态通道共享同一个遗忘率。论文的用词是 "coarse head-wise forget gate"。KDA 把它换成**每个特征维度独立遗忘率**（论文 §3 前言）：α_t ∈ [0,1]^{d_k}，S 的每一行按自己的速率衰减，衰减发生在 Householder 投影之前。

SGLang 的实现直接体现了这个形状（`kimi_linear.py:349` `forward_qkvbfg` 起）：遗忘门由两段低秩投影生成，`f_a_proj`（ReplicatedLinear，hidden → head_dim）再接 `f_b_proj` 得到 forget_gate，在 prefill 路径上 unflatten 成 `[T, H, head_dim]`（`kimi_linear.py` forward 内 `forget_gate.unflatten(-1, (-1, self.head_dim))`）。head_dim = 128，所以每个头每步产生 128 个独立遗忘率，经 `fused_fg_b_proj` 的 batch matmul 一次算完（`:364` fused 路径）。β_t 则是标量，sigmoid 后使用（`:386` 附近，prefill 路径 `beta.float().sigmoid()`，decode 路径由 kernel 内部激活）。

两个动机（论文 §3、§6.1）：

**记忆的精细调控**。有限大小的循环状态是稀缺记忆，逐头标量遗忘太粗：一个头里既有要长期保留的关键信息，也有该丢弃的噪声，却被迫共享同一遗忘率。通道级门控让模型按特征维度选择性遗忘。合成任务（Palindrome、MQAR、Stack，长度 256→2048）上 KDA 的准确率与收敛速度都好于 GDN（论文 Fig. 4），没有 delta rule 的纯乘性衰减（Mamba-2 类）在这些任务上全部失败。

**频率多样性**。这间接补了位置编码的缺（见第四节）：RoPE 的核心是把不同频率分给不同维度对，而逐头标量衰减丢掉了这种逐维多样性。通道级门控给了循环转移矩阵沿特征维度的变化频率，表达能力向 RoPE 看齐。

## 三、DPLR 特化：把「算得快」写进约束里

chunkwise 并行（[总览](linear-attention-serving.md) §2.1 讲过它的存在）需要在数学上把一段序列的状态转移压缩成紧凑形式。这里 KDA 有个关键设计：**把转移矩阵约束成 DPLR（对角加低秩）的特化形态**（论文 §6.2）。

一般 DPLR 转移写成 S_t = (D − a_t b_t^⊤) S_{t−1} + k_t v_t^⊤。KDA 令 D = Diag(α_t)、a_t = β_t k_t、b_t = k_t ⊙ α_t，于是 (D − a b^⊤) 恰好等于 Eq. 1 的 (I − β k k^⊤) Diag(α)。两个约束收益：

1. **数值稳定**：a、b 都绑在 k 上，chunk 并行公式里的 1/Γ 除法（一般 DPLR 分块递推的常见不稳定源）被消掉；
2. **算子更省**：第二级 chunk 的矩阵计算从 4 个减到 2 个，再省 3 次矩阵乘。论文 §3.2 称算子效率相对一般 DPLR 提升约 100%，Fig. 2 实测 KDA 在至 64k 长度上接近 DPLR 的 2 倍速度。

chunkwise 主体（论文 §3.1，Eq. 2-9）分四步：部分展开递推、WY 表示把一串 rank-1 更新压成紧凑乘积、UT 变换用下三角求逆一次性打包（把非 matmul 计算降到最少，喂饱 Tensor Core）、最后 chunk 间只传状态、chunk 内全并行。总复杂度（论文 Eq. 13，C=64）为 6T·d_h² + 3TC·d_h + TC²，对比全注意力的 2T²·d_h。

## 四、衰减门把 RoPE 换掉了

位置编码的替代是 KDA 机制里最反直觉的一环。RoPE 的相对位置编码可以写成 q_t^⊤ (∏R_j) k_i 的旋转矩阵累积；而带门控 delta rule 的输出同样能写成 q_t^⊤ (∏A_j (I − β_j k_j k_j^⊤)) k_j v_j 的转移矩阵累积（论文 Eq. 12，§6.1）。差别在于：RoPE 的 R_j 是固定的正交旋转，KDA 的 A_j 是**数据依赖的衰减门**。论文的结论是：遗忘门相当于一种可学习、数据依赖的乘性位置编码，还「放松了 RoPE 的正交约束」。

这带来一个架构后果：既然 KDA 层承担了近因与位置信息，**MLA 全注意力层就不需要 RoPE 了**（论文 §4，"we apply NoPE to all full attention (MLA) layers"）。附带三个工程红利：MLA 推理时可转纯 MQA（KV 头数进一步压缩）；长上下文训练不需要调 RoPE base 或 YaRN；K3 的技术报告也沿用了同一设计。消融数据支持这个选择（论文 Table 5）：NoPE 版 RULER 84.3 vs RoPE 版 78.8。

## 五、机制换来的账：质量与速度

- **KV 与 decode**：3:1 混合让 3/4 的层不再产生随长度增长的 KV。1M 上下文、利用省下的显存加大 batch，TPOT 1.84 ms vs 全注意力 11.48 ms（约 6.3×，论文 Fig. 1b）；batch=1 时约 2.3×（§6.3），随序列变长趋近 3:1 混合的理论上限。
- **Prefill**：512k 输入约 2.3×、1M 约 2.9×（Fig. 7a）；4k–16k 短序列与全注意力相当。值得注意：GDN-H 与 KDA 的 prefill 曲线几乎重合，说明细粒度门控在 prefill 侧几乎零额外开销（代价都付在更慢的收敛与记忆精度上，不付在吞吐上）。
- **质量**：MMLU-Pro 51.0 vs MLA 47.2，RULER(128k) 84.3 vs 81.3（同 token 预算，Fig. 1a）。论文想证明的正是「又快又好」：省显存不该拿质量换。

## 六、机制到实现的对照

| 机制         | 论文表述                        | SGLang 实现（`kimi_linear.py`）                                                                                                                                                     |
| ------------ | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 通道级遗忘门 | Diag(α_t)，每特征维独立速率     | forget_gate 经低秩双投影生成，unflatten 为 [T, H, head_dim=128]（`forward_qkvbfg` :349、forward :386）                                                                              |
| 标量写入门 β | Sigmoid(W_β x)，∈[0,1]          | beta 由 b_proj 生成，prefill 路径显式 sigmoid（forward :386 附近）                                                                                                                  |
| 融合投影     | qkv+β+f_a+g_a 一次算            | `fused_qkvbfg_a_proj`（MergedColumnParallelRepeatedLinear），两段低秩经 `fused_fg_b_proj` batch matmul（:364）                                                                      |
| 门激活位置   | 论文未指定                      | decode/target-verify kernel 内部激活，prefill chunk kernel 收预激活 β（forward 注释）                                                                                               |
| 短卷积       | ShortConv + Swish 预处理        | `conv_size = config.linear_attn_config["short_conv_kernel_size"]`（:212）                                                                                                           |
| 状态         | 每头 d_k×d_v 固定，不随长度增长 | 状态池与检查点管理见 [总览 §2.2](linear-attention-serving.md)                                                                                                                   |
| Kernel       | chunkwise + recurrent 双模式    | decode 走 recurrent kernel，extend 走 chunk kernel（forward 内 forward_mode 分支）；kernel 实现见 `layers/attention/linear/kernels/`（kda_triton/kda_ptx/kda_flashkda 等 7 个后端） |

---

## 结语

把三行递推连起来看，KDA 的全部改动可以归纳为一句话：**把「忘什么」的决定权从整头细化到每个特征维度，再把转移矩阵约束成算得快的形态**。通道级门控回答的是记忆精度，DPLR 特化回答的是硬件效率，NoPE 的架构后果回答的是与全注意力层如何分工。理解这三件事，再看 [总览](linear-attention-serving.md) 里的系统挑战（串行 prefill、状态检查点、双账本），以及本系列接下来的两篇——[02 · chunkwise 串行 prefill 的调度](02-chunkwise-scheduling.md)与 [03 · 状态语义深水区](03-state-semantics.md)——机制层面的地基就算打完了。

## 源文件索引

| 文件                                                 | 关键符号                                                                                     | 引用点                 |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------- |
| `python/sglang/srt/models/kimi_linear.py`            | `KimiDeltaAttention`、`forward_qkvbfg`、`forward_qkvbfg_fused`、`forward`                    | :185、:349、:364、:386 |
| `python/sglang/srt/layers/attention/linear/kernels/` | kda_triton / kda_ptx / kda_flashkda / kda_cutedsl / kda_helion / kda_nvidia / kda_flashinfer | 7 个后端               |

## 参考资料

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692), arXiv:2510.26692, 2025——本文公式与 benchmark 全部出处（§2.2、§3、§3.1、§3.2、§4、§6.1-6.3、Table 5、Fig. 1/2/4/7）
- [总览：没有 KV Cache 的模型](linear-attention-serving.md)——谱系与系统侧挑战
- Kimi Team, [Kimi K3 技术报告](https://arxiv.org/abs/2607.24653), 2026——KDA 的 2.8T 级落地与 scaled-sigmoid 下界改动
