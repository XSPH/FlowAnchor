# FlowAnchor 实现说明:代码与论文对照(详细版)

> 论文: **FlowAnchor: Stabilizing the Editing Signal for Inversion-Free Video Editing**
> (arXiv 2604.22586, Chen et al., 基于 Wan2.1-T2V-1.3B / Wan-Edit / FlowEdit)
>
> 本文档把每段代码标注到论文的具体位置(章节 / 公式 / 算法行),并用通俗语言解释
> 它在做什么、为什么这么做。引用格式:`文件名:行号` ↔ 论文 Eq./Sec./Alg.,
> 行号在 IDE 中可直接点击跳转。文中所有具体数值(sigma 网格、手算示例、γ_F 表)
> 均为实际运行本仓库代码得到,可复算核对。

**目录**

1. [背景:这篇论文在做什么](#1-背景这篇论文在做什么)
2. [文件结构总览](#2-文件结构总览)
3. [采样主循环 ↔ Algorithm 1(逐行对照)](#3-采样主循环--algorithm-1)
4. [SAR:空间感知注意力精炼 ↔ Sec. 3.3 / B.1](#4-sar空间感知注意力精炼)
5. [AMM:自适应幅度调制 ↔ Sec. 3.4 / B.2](#5-amm自适应幅度调制)
6. [mask 的加载与下采样](#6-mask-的加载与下采样)
7. [目标词 → token 索引 J_tar](#7-目标词--token-索引-j_tar)
8. [超参数速查表](#8-超参数速查表)
9. [论文未明说之处与本实现的选择](#9-论文未明说之处与本实现的选择)
10. [验证:每个测试对应哪条公式](#10-验证每个测试对应哪条公式)
11. [常见问题 FAQ](#11-常见问题-faq)

---

## 1. 背景:这篇论文在做什么

### 1.1 任务:文本驱动的视频编辑

输入三样东西:一段视频、一句描述原视频的话(源提示词 P)、一句描述想要结果的话
(目标提示词 P*)。比如:

> 视频:一辆红色 SUV 在山路上行驶
> P  = "A red SUV is navigating a mountain road ..."
> P* = "A blue SUV is navigating a mountain road ..."

要求输出"蓝车版"的视频,且**不训练任何模型**——只靠一个现成的文生视频模型
(Wan2.1-T2V-1.3B)在推理时做文章。

### 1.2 从 Rectified Flow 说起(论文 Sec. 3.1, Eqs. 1-2)

Wan2.1 这类模型的生成原理是"整流流"(rectified flow):把生成看成一条从纯噪声
(t=1)走到干净视频(t=0)的直线路径。任意时刻的中间状态是噪声和干净数据的线性
插值(Eq. 2):

```
Z_t = (1-t)·X₀ + t·X₁        X₀=干净数据, X₁=噪声, t∈[0,1]
```

模型学到的是一个**速度场** V(Z, t, 文本):告诉你"位于 Z、时刻 t、想生成'文本'
描述的内容时,该往哪个方向走"。生成 = 沿速度场做数值积分(本仓库用 Euler 法)。

### 1.3 FlowEdit 的免反演编辑(Sec. 3.1, Eqs. 3-5)

传统编辑要先做 **inversion**:把原视频"倒放"回噪声,再从那个噪声出发用新提示词
生成。又慢又容易积累误差(视频上尤其明显,论文 Sec. 1)。

FlowEdit 的思路:根本不用回到噪声,直接从原视频出发,每一步问模型两个问题:

1. 给**加噪的原视频**,以源提示词 P 提问 → 得到源速度 V_src;
2. 给**加噪的编辑中间态**,以目标提示词 P* 提问 → 得到目标速度 V_tar。

两个速度的差 **ΔV = V_tar − V_src** 称为**编辑信号**(Eq. 4)。直觉:两次提问
唯一的区别是"red vs blue",所以其它一切共同因素(构图、运动、光照该怎么走)在
相减时抵消,只剩下"从红到蓝"这个纯语义方向。沿 ΔV 积分,就把原视频平移到了
目标视频(Eq. 5),背景结构天然保留。

FiVE-Bench 把这套方法原样搬到 Wan2.1 上,称为 **Wan-Edit**(本仓库
`FiVE-Bench/models/wan-edit/wan/text2video.py` 的 `edit()`,是我们的出发点)。

### 1.4 FlowAnchor 发现的问题(Sec. 3.2, Fig. 3)

Wan-Edit 在多物体、长视频上频繁失败。论文把根因定位在 **编辑信号 ΔV 本身失稳**,
表现为两个互补的症状:

| 症状 | 现象 | 原因(论文分析) |
|---|---|---|
| **定位不准** (Fig. 3a) | ΔV 的能量跑到错误区域:想改右边的鸟,信号漂到左边;或糊成一片 | cross-attention 图在多物体场景中语义对齐差,ΔV 跟着 CA 图走偏;统计上 ΔV 与真值 mask 的 IoU 越低,Local CLIP-T 越低 |
| **幅度衰减** (Fig. 3b) | 帧数越多,ΔV 幅值单调下降,编辑"没劲了" | 帧数增大后,时空注意力聚合的**稠密源上下文**盖过了目标词提供的**稀疏编辑语义**,V_tar 越来越接近 V_src,差值趋零 |

FlowAnchor 的两个对策正好一人管一个症状:

- **SAR**(Spatial-aware Attention Refinement):修 CA 图 → 管"**改哪里**";
- **AMM**(Adaptive Magnitude Modulation):放大 ΔV → 管"**改多强**"。

两者都是推理时的即插即用操作,不训练、不加网络。

---

## 2. 文件结构总览

| 文件 | 职责 | 对应论文 |
|---|---|---|
| `flowanchor.py` | SAR / AMM 的数学核心 + mask / token 工具(纯函数,可独立测试) | Sec. 3.3, 3.4, 附录 B |
| `edit_flowanchor.py` | 完整编辑管线:加载 → 编码 → Alg.1 主循环 → 解码,以及 CLI | Algorithm 1, Sec. 4.1 |
| `eval_five.py` | FiVE-Bench 批量评测(自动读 bmasks、自动推导目标词) | Sec. 4.2 |
| `run.sh` | 单视频快速运行(已按论文默认超参配置) | Sec. 4.1 |
| `run_ablation.sh` | 一键跑论文 Table 2(模块消融)/ Table 3(超参敏感性)全部配置 | Sec. 4.3, 附录 E |
| `make_mask.py` | 手工框选生成 mask(bbox 足够,见 Table 4) | Sec. 4.5, 附录 F |
| `test_sanity.py` | 37 项公式级对拍(不需要 GPU/权重) | 附录 B 全部公式 |
| `test_pipeline.py` | 9 项管线冒烟测试(mock 模型驱动完整循环) | Algorithm 1 |
| `FiVE-Bench/`(子模块) | Wan-Edit 参考实现 + 评测基准 | — |

依赖关系:`edit_flowanchor.py` → `flowanchor.py` + `wan`(来自子模块,经
PYTHONPATH 引入);`eval_five.py` → `edit_flowanchor.py`。

---

## 3. 采样主循环 ↔ Algorithm 1

入口:`edit_flowanchor.py:119` `flowanchor_edit()`。
下面先给一步之内的数据流,再逐行对照论文伪代码。

### 3.1 默认配置下的张量形状(832×480, 41 帧)

理解代码时对照这张表会轻松很多:

| 量 | 形状 | 说明 |
|---|---|---|
| 输入视频 | `[1, 3, 41, 480, 832]` | RGB, 归一化到 [-1,1] |
| VAE latent `x_src` | `[16, 11, 60, 104]` | 时间 4× 压缩((41-1)/4+1=11), 空间 8× |
| DiT 视频 token | `16380 = 11×30×52` 个 | patchify 再把空间 2× 分块, (F,H,W) 行优先展平 |
| 文本 context | `[L, 4096]`, pad 到 `[512, 4096]` | umT5-xxl 编码, L 为真实 token 数 |
| CA 打分表(单层) | `[1, 12, 16380, 512]` | 12 个头, 每头独立一张"视频 token × 文本 token"表 |
| 编辑信号 ΔV | `[16, 11, 60, 104]` | 与 latent 同形 |

### 3.2 时间网格(Sec. B "follow FlowEdit ... T=25")

`edit_flowanchor.py:214-221`:用 Wan 官方 `FlowUniPCMultistepScheduler` 生成
shift=5 的 sigma 网格(σ 即归一化时刻 t,σ=1 纯噪声,σ=0 干净)。25 步的实际数值:

```
i:      0     1     2     3     4     5   ...   10    ...   19    20    21    22    23    24
σ_i:  1.000 0.992 0.983 0.973 0.963 0.952 ... 0.882 ... 0.612 0.555 0.488 0.405 0.303 0.172
       └─跳过─┘  └───────────── SAR 激活区 (σ ≥ 0.6, 即 i=0..19) ─────────────┘
```

三个关键刻度:
- **i=0,1 被跳过**(`:234`,`skip_timesteps=2`):对应论文 n_max = T−2 = 23
  ("skip the first two steps ... preserve the source layout")。噪声最大的两步
  什么都不做,Z_edit 保持为原视频,保住粗布局;
- **σ ≥ 0.6 时 SAR 开**(`:249`):即前 20 步(扣掉跳过的 2 步,实际 18 个编辑步);
- **σ < 0.6 后 SAR 关**:最后 5 步模型专心画细节(论文 E.2:"subsequent denoising
  steps are better left to preserve appearance and structural details")。

注意 shift=5 把网格强烈地"挤"向高噪端——前 20 步只把 σ 从 1.0 走到 0.6,
最后 5 步却要从 0.55 冲到 0。这是 Wan 官方的调度,编辑的"语义决策"集中在前段。

### 3.3 逐行对照 Algorithm 1(附录 A)

| 论文 Alg. 1 | 代码 | 通俗解释 |
|---|---|---|
| line 1: `Z_edit ← X_src` | `edit_flowanchor.py:231` | 编辑轨迹从原视频 latent 出发,不是从噪声 |
| line 2: `τ ← 0.6T; γ_F = γ·logF/logF₀` | `:249` 与 `flowanchor.py:53` | 预先确定 SAR 生效窗口与 AMM 放大系数 |
| line 3: `for i = T,...,1` | `:233` | 沿 §3.2 的网格从高噪走向低噪 |
| line 4: `N ~ N(0,I)` | `:241-243` | 每步**独立重采**一份噪声(n_avg=1, Sec. B),由 seed 控制可复现 |
| line 5: `Z_src ← (1−t_i)X_src + t_i·N` | `:244` | 给原视频按**当前** t_i 加噪。⚠ Wan-Edit 旧代码用的是上一步的 t_prev,论文和 FlowEdit 都用当前 t_i,本实现已改正 |
| line 6: `Z_tar ← Z_edit + Z_src − X_src` | `:245` | 关键技巧:目标态与源态**共享同一份噪声**,两者之差恰好是到目前为止累积的编辑量 Z_edit−X_src。这样下一步两次前向的差才是纯语义差 |
| line 7: `V_tar ← V_SAR(Z_tar,...)` 若 t≥τ | `:248-252` | 算目标速度。**只有这一次前向**开 SAR(见 §4.2);uncond 前向和源前向都不开 |
| line 8: `V_src ← V(Z_src, t, P)` | `:253-258` | 算源速度。实践中 V 都带 CFG:`uncond + scale·(cond − uncond)`,源用 scale 5.0、目标用 10.0(沿用 Wan-Edit,论文未另行说明);因此每步共 **4 次**模型前向,25 步编辑 23 步共 92 次 |
| line 9: `ΔV ← V_tar − V_src` | `:260` | 编辑信号 |
| line 10: AMM | `:262` | 每步都放大(见 §5),与 mask 无关 |
| line 11: `Z_edit += (t_{i−1} − t_i)·ΔV` | `:265` | **显式 Euler 更新**(Eq. 13)。(t_{i−1}−t_i) 为负,方向指向低噪端。⚠ Wan-Edit 旧代码把 ΔV 塞进 UniPC 多步求解器,与论文/FlowEdit 不符,本实现改为 Euler;`--sample_solver unipc` 保留旧行为仅供对比 |
| line 13: 返回 Z_edit(t=0) | `:277-281` | VAE 解码回像素 |

### 3.4 为什么 SAR 不作用在源分支和 uncond 分支?

Alg. 1 中只有 V_tar 写成 V_SAR(...)。直觉:SAR 的作用是"让**目标词**贴住目标区域",
而源提示词里没有目标词(它还是 red),负向提示词更没有;若对源分支也调制,
相减时增强量会被部分抵消。代码用 `sar_controller.active` 开关在四次前向之间
精确切换(`:249, :252`)。

---

## 4. SAR:空间感知注意力精炼

**一句话:** 把 cross-attention 的"注意力打分表"改一改,让目标词死死盯住
mask 圈出的区域,并且整个时间轴上都盯住。

### 4.1 打分表长什么样

DiT 的每个 cross-attention 层里,每个注意力头都有一张打分表(logits)
A ∈ R^(16380×512):第 i 行是第 i 个视频小块,第 j 列是提示词的第 j 个 token,
A_{i,j} = "小块 i 有多在意词 j"(softmax 之前的原始分数)。

**论文 B.1 明确:SAR 改的是 softmax 之前的 logits**(Eq. 14 先定义 softmax
归一化,Eq. 15/17 在 logits 上调制,改完再 softmax)。好处是数学性质干净:
调制是"向已有的最大/最小值做凸插值"(Eqs. 19-20),所以**永远不会造出超出原
表范围的值**(Eqs. 21-22),softmax 之后仍是合法的概率分布——只重塑相对
对比度,不引入数值风险。

### 4.2 两步调制 — `flowanchor.py:88` `sar_modulate_logits()`

**Step 1: Text-Token Modulation(Eq. 15, 代码 `:133-136`)——按行改**

对 mask 内的每个视频小块(M_i=1):

```
目标词 j∈J_tar:  A'_{i,j} = A_{i,j} + β₁·(A_i^max − A_{i,j})     往该行最大值抬
其他词 j∉J_tar:  A'_{i,j} = A_{i,j} − β₁·(A_{i,j} − A_i^min)     往该行最小值压
mask 外(M_i=0):  不动
```

A_i^max/A_i^min 是**该行**(该小块对所有真实文本 token)的最大/最小分数
(Eq. 16, 代码 `:133-134`)。通俗:告诉车身区域的每个小块——"你们只需要在意
blue 这个词,别的词先放一放"。

**Step 2: Spatio-Temporal Modulation(Eq. 17, 代码 `:139-142`)——按列改**

只看目标词那几**列**,统计每列在**全部 16380 个视频 token 上**(即跨越所有帧!)
的最大/最小值(Eq. 18, 代码 `:139-140`),然后:

```
mask 内(M_i=1):  A''_{i,j} = A'_{i,j} + β₂·(A'^max_j − A'_{i,j})   往整列最大值抬
mask 外(M_i=0):  A''_{i,j} = A'_{i,j} − β₂·(A'_{i,j} − A'^min_j)   往整列最小值压
```

通俗:反过来告诉 blue 这个词——"你只许作用在车身上,不许碰背景"。因为列统计
横跨所有帧,这个约束在时间上是一致的:不会出现第 3 帧 blue 盯着车、第 4 帧
blue 飘到马路上的情况,从而抑制帧间闪烁(论文:"enforce spatio-temporal
consistency ... unstable attention distributions ... manifesting as flickering")。

**手算示例**(可与 `sar_modulate_logits` 输出逐位核对,β₁=β₂=0.3):

设 1 个头、3 个视频小块(前两个在 mask 内)、3 个词,目标词是第 0 列 "blue":

```
原始 logits A          Step 1 之后 A'         Step 2 之后 A''
      blue  car  road       blue  car  road        blue  car  road
i0(内) 1.0  2.0  0.0   →    1.3   1.4  0.0    →    1.51  1.4  0.0
i1(内) 0.5  1.5  1.0   →    0.8   1.2  0.85   →    1.16  1.2  0.85
i2(外) 2.0  1.0  0.5   →    2.0   1.0  0.5    →    1.64  1.0  0.5
```

看 blue 列的变化:原始表里**mask 外的 i2 反而是 blue 的最大响应**(2.0)——这正是
"定位不准"的缩影。Step 1 把 mask 内小块的 blue 分数抬起来、car 分数压下去;
Step 2 继续抬 mask 内(1.3→1.51)、压 mask 外(2.0→1.64),最终 mask 内外的相对
关系被翻转过来。βs 越大翻转越狠(β=1 时直接取到极值),0.3 是论文在"定位够准"
和"不破坏保真"之间选的平衡点(Table 3, 附录 E.1)。

**实现细节**(容易踩坑处):

- 文本轴统计只在**真实文本长度** L 内做:umT5 的 context 会 pad 到 512,pad 位
  的 logits 保持原样、也不参与 max/min(`:114` 截取,`:145` 写回);
- 视频轴统计(Eq. 18)同理只在真实视频 token 数内做(`:139-140` 的 `:n_vid` 切片);
- 每个注意力头**独立**调制(max/min 都是每头自己的,论文未提 head,见 §9)。

### 4.3 怎么塞进 Wan 模型 — `flowanchor.py:271` `SARController`

Wan 的注意力用 flash-attention / SDPA 一步到位,**中途拿不到打分表**。所以在编辑
开始前,把全部 30 个 DiT block 的 `cross_attn.forward` 替换成自己的版本
(`patch()`, `:295`;结束后 `unpatch()`, `:304` 恢复):

`_sar_attention()`(`:316`)显式走一遍注意力:

```
q,k,v 投影(复用原模块权重) → logits = qkᵀ/√d      [1, 12, 16380, 512]
→ sar_modulate_logits(...)                          (上面的两步调制)
→ softmax(float32, 数值稳定)  → 权重 × v  → 输出投影 o
```

`active` 开关(`:311-313`)让补丁"平时隐身":False 时直接调用原 forward,
零开销;只有目标条件前向 + σ≥0.6 时才走显式路径。显式注意力每层临时多占
约 0.4 GB(bf16 的 logits + softmax 副本),用完即释放,不叠加。

正确性验证:`test_sanity.py` 用 wan 官方 `WanT2VCrossAttention` 模块对拍——
β=0 时 SAR 路径与官方 flash/SDPA 路径输出一致(bf16 容差),patch/unpatch
前后行为逐位相同。

---

## 5. AMM:自适应幅度调制

**一句话:** 编辑信号哪里强就把哪里再放大一点(背景近零处不动),视频越长放得
越大,正好抵消长视频的信号衰减。

代码:`flowanchor.py:59` `adaptive_magnitude_modulation()`,
每个采样步都调用(`edit_flowanchor.py:262`),**完全不依赖 mask**。

### 5.1 逐条公式对照

| 论文 | 代码 | 做什么 |
|---|---|---|
| Eq. 23: ΔV ∈ R^(B×C×F×H×W) | `:70-71` | 输入即编辑信号(兼容无 batch 维的 `[C,F,H,W]`) |
| Eq. 24: V̄ = (1/C)Σ_c ΔV^(c) | `:78` | 16 个 latent 通道求平均 → 每个时空位置一个标量,代表"这里语义变化有多大" |
| Eq. 25: C = (V̄−min)/(max−min+ε) | `:79-81` | **逐样本**在整个 F×H×W 上 min-max 归一到 [0,1](ε=1e-7):强变化处≈1,背景≈0。得到一张"软重要性图" |
| Eq. 26: γ_F = γ·logF/logF₀ | `:53` `gamma_f()` | 帧数越多衰减越狠 → 放大系数按 log 递增补偿 |
| Eq. 27: ΔV ← (1+γ_F·C)⊙ΔV | `:82` | 逐元素放大,C 沿通道维广播 |
| Eq. 28: F₀ = (81−1)/4+1 = 21 | `--f0` 默认 21 | 基准取 Wan2.1 原生 81 帧对应的 **latent 帧数**(模型"舒适区"的时间长度) |
| Eq. 29: 1 ≤ 放大倍率 ≤ 1+γ_F | 公式自然保证 | 只放大不缩小、有上界——不会把噪声无限吹大 |

### 5.2 γ_F 具体数值(γ=1.0)

| 像素帧数 | latent 帧数 F | γ_F | 最大放大倍率 |
|---|---|---|---|
| 1(单图) | 1 | **0.000** | 1.00×(完全不放大) |
| 17 | 5 | 0.529 | 1.53× |
| 41(FiVE 默认) | 11 | 0.788 | 1.79× |
| 81(Wan 原生) | 21 | 1.000 | 2.00× |

两个自检推论:
- **F=1 时 AMM 恒等**——与"FlowEdit 在图像域本来就好,不需要补偿"的观察一致
  (论文特意强调这一设计性质);
- 帧数是从 ΔV 的形状**自动读出**的(`:73`),无需手动传。

### 5.3 与"直接乘 mask"的本质区别(论文附录 G 的观点)

FlowDirector 等方法用 CA 图当硬 mask 去闸门编辑流,注意力一错立刻污染轨迹。
AMM 不做闸门:对比度图 C 来自**编辑信号自己**,只是把"已经在变化的地方"推得
更用力,背景处 C≈0 保持 1× 不动。所以就算 mask 不给(SAR 关闭),AMM 依然
成立——这也是为什么代码里 AMM 无条件执行。

---

## 6. mask 的加载与下采样

### 6.1 像素空间加载 — `edit_flowanchor.py:72` `load_mask()`

三种输入格式,统一输出 `[1, 1, F, 480, 832]` 的 0/1 张量(白 >127 = 编辑区):

| `--mask_path` 形式 | 处理 |
|---|---|
| 目录(`bmasks/0001_bus/`) | 逐帧 png/jpg,不足时重复最后一帧 |
| `.mp4` | 逐帧灰度化后二值化 |
| 单张 `.png/.jpg` | 静态 mask 复制到所有帧(适合目标不怎么动;`make_mask.py` 生成的就是这种) |

### 6.2 下采样到 token 网格 — `flowanchor.py:152` `downsample_mask_to_latent()`

SAR 的 M_i 定义在**视频 token** 上(16380 个),必须把像素 mask 精确映射过去,
错一位整个调制就贴错地方。三个映射步骤:

**时间(`:177-184`)** — Wan VAE 是**因果**压缩,首帧单独成一个 latent 帧:

```
latent 帧 0 ← 像素帧 0
latent 帧 j ← 像素帧 4j-3 .. 4j    (j ≥ 1)
例(41帧): latent 2 ← 像素帧 5,6,7,8;  latent 10 ← 像素帧 37..40
```

每组取 max(组内任一帧被标记即算标记)。

**空间(`:186-188`)** — 一个 token 对应 16×16 像素(VAE 8× × patchify 2×),
用 `adaptive_max_pool2d`:**格子里任何一个像素在 mask 内,整个 token 就算在内**。
选 max 而不是均值/阈值,是为了细长物体(链条、腿)不在下采样中断掉;代价是
mask 边缘外扩最多 15 像素——由于 SAR 是软调制(§5.3 同理),外扩无害。

**展平顺序(`:189`)** — 按 (F, H, W) 行优先展平:
`token_idx = f·(30·52) + h·52 + w`。这与 DiT patchify 的顺序
(`wan/modules/model.py:523-526`: `patch_embedding → flatten(2).transpose`)
严格一致——这是旧版代码出错的地方(直接把像素分辨率的 mask 展平截断,
与 token 完全对不上号)。

---

## 7. 目标词 → token 索引 J_tar

SAR 需要知道"目标词是打分表的哪几列"。**列编号是 umT5 分词器的 token 位置,
不是空格分词的单词位置**——旧版代码用 `prompt.split()` 数位置,和 CA 的文本轴
完全对不上,是必须修的 bug。两级流程:

**第一步:确定目标词 — `flowanchor.py:195` `derive_target_words()`**

不指定 `--target_words` 时,对源/目标提示词做 word-level diff
(`difflib.SequenceMatcher`),取出被替换/新增的词:

```
"A woman in a pink sweater ..." vs "A woman in a lemon sweater ..." → ['lemon']
"A man is performing parkour"   vs "A Spiderman is performing parkour" → ['Spiderman']
```

这对应论文对 J_tar 的定义:"the index set of target text tokens **driving the
edit**"。也可以手动指定(比如编辑涉及多个词时更稳)。

**第二步:词 → token 位置 — `flowanchor.py:211` `find_target_token_indices()`**

用管线自带的 umT5 tokenizer(`text_encoder.tokenizer`)+ fast tokenizer 的
offset mapping(字符区间 ↔ token 区间求交):

```
"A pink flamingo in the middle ..." 分词为 ▁A / ▁pink / ▁flam / ing / o / ...
target_words = ['pink', 'flamingo']  →  J_tar = [1, 2, 3, 4]
```

注意 flamingo 被切成 3 个 piece,**必须全部选中**,否则 SAR 只增强了半个词。
细节处理:
- 匹配前先做与文本编码完全相同的 whitespace 清洗(`:234`),保证字符偏移一致;
- 优先词边界匹配(`red` 不会命中 `covered`),失配再放宽为子串(`:237-243`);
- 慢速 tokenizer 时回退为 token-id 子序列匹配(`:257-268`);
- **一个词都映射不到时禁用 SAR 并打 warning**(`edit_flowanchor.py:198-200`)——
  旧版回退成"全部 token 都是目标",等于把整句话都增强,必须移除。

---

## 8. 超参数速查表

| CLI 参数 | 默认 | 论文出处 | 调大/调小的效果(附录 E) |
|---|---|---|---|
| `--sample_steps` | 25 | Sec. 4.1 "T=25" | — |
| `--skip_timesteps` | 2 | Sec. 4.1 / B (n_max=23) | 调大→更保原视频但编辑变弱 |
| `--sample_solver` | euler | Eq. 13 / Alg.1 line 11 | `unipc` = Wan-Edit 旧行为,仅对比用 |
| `--sample_shift` | 5.0 | Wan2.1 官方默认 | — |
| `--sample_guide_scale` | 5.0 | 论文未写明,沿用 Wan-Edit | 源分支 CFG |
| `--tgt_guide_scale` | 10.0 | 同上 | 调大→编辑更强也更容易过 |
| `--beta1` | 0.3 | Sec. 4.1 / Table 3 / E.1 | 0.1 定位不足;0.5 过强微损保真 |
| `--beta2` | 0.3 | 同上 | 同上,另控帧间稳定 |
| `--gamma` | 1.0 | Sec. 4.1 / Table 3 / Fig. 13b | 0.5 编辑不足;1.5 过编辑结构变形;0 = 关闭 AMM |
| `--f0` | 21 | Eq. 28 | 一般不动 |
| `--sar_tau` | 0.6 | Sec. 4.1 "τ=0.6T" / Fig. 14 / E.2 | 0.8 语义锚定不足;0.4 干扰细节生成 |
| `--no_sar` | 关 | Table 2 消融 | 开 = w/o SAR |
| `--target_words` | 自动 diff | — | 多词编辑建议手动给全 |

---

## 9. 论文未明说之处与本实现的选择

精准复现时,以下四点论文没有给出实现细节。当前默认选择与替代方案如下,
服务器消融结果可用来校准(对应关系见 `run_ablation.sh` 的反馈流程):

| # | 歧义点 | 当前选择 | 依据 / 替代方案 |
|---|---|---|---|
| 1 | CFG 引导强度 | src 5.0 / tgt 10.0 | Alg.1 只写抽象的 V(·);FlowAnchor 基于 Wan-Edit,沿用其默认。若编辑普遍偏弱可试 tgt 12~15 |
| 2 | "τ=0.6T" 的解释 | 时间值 σ≥0.6(前 20/25 步) | Alg.1 的 `if t_i ≥ τ` 比较的是时间值。另一解释"步数索引前 40%"(前 10 步)等价于 `--sar_tau 0.88` |
| 3 | 多头处理 | 每个 attention head 独立调制 | 论文写 A^(l)∈R^(N×L) 未提 head。替代:头间均值统一调制 |
| 4 | mask 下采样 | max-pool(任一像素命中) | 论文只说 "at the corresponding latent resolution"。替代:面积插值 + 0.5 阈值 |

---

## 10. 验证:每个测试对应哪条公式

两个套件均**不需要 GPU 和模型权重**,可在任何机器上运行:

```bash
python test_sanity.py      # 37 项
python test_pipeline.py    # 9 项
```

**test_sanity.py — 公式级对拍**(方法:把论文公式用朴素 for 循环逐字翻译成
参考实现,与 `flowanchor.py` 的向量化实现比对):

| 测试组 | 验证内容 ↔ 论文 |
|---|---|
| 1. SAR | 三种形状下与 Eqs. 15-18 参考实现逐位一致;β=0 恒等;pad 区不动;调制不越界(Eqs. 21-22) |
| 2. 注意力路径 | 与 wan 官方 `WanT2VCrossAttention` 对拍:β=0 时两条路径输出一致;patch/unpatch 可逆 |
| 3. AMM | 与 Eqs. 24-27 参考实现一致;F=1 恒等;γ_F 公式(Eq. 26/28);倍率界(Eq. 29);极端值稳定 |
| 4. mask 下采样 | 形状/空间对位/因果时间分块(帧0→latent0, 帧5→latent2)/全 1 往返 |
| 5. 词工具 | diff 推导('pink→lemon');fallback 定位;未命中返回空 |
| 6. 采样网格 | σ 网格 26 点终点 0、起点≈1、单调;n_max=23;τ=0.6 激活 20 步 |

**test_pipeline.py — 管线冒烟**(方法:mock VAE/文本编码器/DiT,但 mock DiT
内部真实调用 wan 的 CrossAttention,SAR 的 monkey-patch 路径被完整执行):

- 前向次数恰为 4×23=92(CFG×4、跳 2 步);
- SAR 改变轨迹、τ=2.0(永不激活)时与无 SAR 逐位一致(门控与透传正确);
- γ=0 与 γ=1 结果不同(AMM 生效);同 seed 可复现;unipc 旧路径可运行;
- target_words 缺省时自动推导可用。

模型级验证(真实编辑质量)需 Wan2.1-T2V-1.3B 权重,见 `run.sh` 与
`run_ablation.sh`(后者一键复现 Table 2/3 全部配置)。

---

## 11. 常见问题 FAQ

**Q: 不给 mask 会怎样?**
SAR 自动关闭(日志有 warning),AMM 照常。相当于"增强版 Wan-Edit",简单的
单物体编辑往往也能成;多物体/小目标场景强烈建议给 mask。

**Q: mask 需要多精确?**
不需要精确。论文 Table 4:bounding box 的 L.CLIP-T 21.31 vs 精确 mask 21.59,
几乎无差。用 `make_mask.py` 框个矩形即可;目标会移动就把框画大罩住运动范围。

**Q: 帧数有什么要求?**
建议 4n+1(1, 5, ..., 41, ..., 81),与 Wan VAE 的时间压缩对齐;`--frame_num`
默认 41(FiVE 协议)。不足时按实际读到的帧数走。

**Q: 一次编辑要多少次模型前向? 显存多少?**
(25−2)×4 = 92 次 DiT 前向 + 1 次 VAE 编码 + 1 次解码。SAR 激活的 18 步里,
目标条件前向走显式注意力,每层临时 +0.4 GB。论文报告(Fig. 8)整体峰值约
20 GB(A800);消费级卡配合 `--offload_model True --t5_cpu` 可显著降低。

**Q: 为什么改成 Euler 而不用 Wan 默认的 UniPC?**
论文 Eq. 13 / Alg.1 line 11 就是 Euler,且 FlowEdit 官方实现也是手写 Euler。
UniPC 是多步求解器,内部维护"模型输出"的历史做高阶外推——但我们喂给它的
ΔV 并不是 Z_edit 真正的速度场,高阶校正在数学上不成立。`--sample_solver unipc`
仅为与旧 Wan-Edit 对比保留。

**Q: 随机性来自哪里? 怎么保证可复现?**
唯一随机源是每步的加噪 N(Alg.1 line 4),由 `--base_seed` 控制的 Generator
产生(`edit_flowanchor.py:241-243`)。同 seed 同配置结果逐位一致
(test_pipeline 第 7 项验证)。
