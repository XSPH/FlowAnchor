<h1 align="center">FlowAnchor</h1>

<p align="center"><strong>稳定编辑信号，实现无反演视频编辑</strong></p>

<p align="center">
  中文 | <a href="README.md">English</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.22586">论文</a> | <a href="https://cuc-mipg.github.io/FlowAnchor.github.io/">项目主页</a>
</p>

---

复现论文 **FlowAnchor: Stabilizing the Editing Signal for Inversion-Free Video Editing** (arXiv:2604.22586)，基于 [Wan-Edit](https://github.com/MinghanLi/FiVE-Bench/tree/main/models/wan-edit) 代码库实现。

FlowAnchor 是一种**无需训练、无需反演**的视频编辑方法。它直接在采样轨迹上施加稳定的编辑信号，速度快且结构保持好，适用于多种复杂场景。

---

## 快速开始

### Windows

```cmd
:: 安装
setup.bat

:: 下载模型 (~5.5GB)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir checkpoints\Wan-AI\Wan2.1-T2V-1.3B

:: 编辑视频
run.bat data\car.mp4 "a red car" "a blue car"
```

### Linux

```bash
# 安装
bash setup.sh

# 下载模型 (~5.5GB)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir checkpoints/Wan-AI/Wan2.1-T2V-1.3B

# 编辑视频
bash run.sh data/car.mp4 "a red car" "a blue car"
```

---

## 系统要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA，显存 ≥ 8GB |
| 系统 | Windows / Linux |
| Python | 3.8+ |
| PyTorch | 2.0+ |
| CUDA | 11.8+ |

**已测试硬件：** RTX 4060 Laptop 8GB

---

## 核心机制

FlowAnchor 解决了无反演视频编辑中的两个关键问题：

### SAR（空间感知注意力精炼）

通过在文本 token 和时空调制两个层面调制交叉注意力图，防止多物体场景中的编辑信号泄漏。

```
A''_i,j = A'_i,j + β₂(A'^max_j - A'_i,j)   if M_i=1, j∈J_tar
```

### AMM（自适应幅度调制）

使用归一化对比度图和帧感知缩放，补偿帧数增加导致的信号衰减。

```
ΔV_flowanchor = ΔV + γ_F × (C ⊙ ΔV)
```

---

## 使用方法

### 单视频编辑

**Windows:**
```cmd
run.bat data\car.mp4 "a red car" "a blue car"
run.bat data\car.mp4 "a red car" "a blue car" masks\car_mask.mp4
```

**Linux:**
```bash
bash run.sh data/car.mp4 "a red car" "a blue car"
bash run.sh data/car.mp4 "a red car" "a blue car" masks/car_mask.mp4
```

### 直接运行 Python

```bash
python edit_flowanchor.py \
    --task t2v-1.3B \
    --ckpt_dir checkpoints/Wan-AI/Wan2.1-T2V-1.3B \
    --video_path data/car.mp4 \
    --prompt "a red car on the road" \
    --tgt_prompt "a blue car on the road" \
    --save_dir outputs
```

### FiVE-Bench 批量评估

```bash
python eval_five.py \
    --ckpt_dir checkpoints/Wan-AI/Wan2.1-T2V-1.3B \
    --FiVE_dataset_json data_FiVE/edit_prompt/edit1_FiVE.json \
    --data_dir data \
    --save_dir outputs_flowanchor
```

---

## 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--prompt` | (必填) | 源视频描述文本 |
| `--tgt_prompt` | (必填) | 目标编辑描述文本 |
| `--video_path` | (必填) | 输入视频路径 (.mp4) |
| `--mask_path` | 无 | 遮罩路径 (.mp4 / .png / 文件夹) |
| `--target_words` | 无 | 目标关键词，用于 SAR 注意力调制 |

### FlowAnchor 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--beta1` | 0.5 | SAR 文本 token 调制强度 |
| `--beta2` | 0.5 | SAR 时空调制强度 |
| `--gamma_scale` | 1.0 | AMM 信号放大系数 |

### 采样参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sample_steps` | 50 | 去噪步数 |
| `--sample_shift` | 5.0 | Flow matching 偏移参数 |
| `--sample_guide_scale` | 5.0 | 源提示词 CFG 引导强度 |
| `--tgt_guide_scale` | 10.0 | 目标提示词 CFG 引导强度 |
| `--skip_timesteps` | 16 | 跳过的初始步数 |
| `--base_seed` | -1 | 随机种子，-1 为随机 |

---

## 项目结构

```
FlowAnchor/
├── flowanchor.py        # SAR + AMM 核心算法
├── edit_flowanchor.py   # 单视频编辑主脚本
├── eval_five.py         # FiVE-Bench 批量评估
├── run.bat              # Windows 启动脚本
├── run.sh               # Linux 启动脚本
├── setup.bat            # Windows 安装脚本
├── setup.sh             # Linux 安装脚本
├── test_sanity.py       # 代码验证脚本
└── README.md
```

---

## 模型选择

| 模型 | 显存需求 | 质量 | 下载命令 |
|------|---------|------|---------|
| Wan2.1-T2V-1.3B | ≥ 8GB | 良好 | `huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B` |
| Wan2.1-T2V-14B | ≥ 24GB | 最佳 | `huggingface-cli download Wan-AI/Wan2.1-T2V-14B` |

---

## 常见问题

| 问题 | 解答 |
|------|------|
| 没有 GPU 能用吗？ | 可以但非常慢，41 帧视频可能需要几小时 |
| 遮罩必须提供吗？ | 不必须。有遮罩时 SAR+AMM 都启用，效果最好 |
| 如何生成遮罩？ | 用 SAM / Grounded-SAM 等分割工具，或手动用画图工具 |
| `No module named 'wan'` | 设置 PYTHONPATH 或使用 run.bat / run.sh |
| 显存不够？ | `--offload_model True`（默认开启）自动加载/卸载模型 |

---

## 引用

```bibtex
@article{chen2026flowanchor,
  title={FlowAnchor: Stabilizing the Editing Signal for Inversion-Free Video Editing},
  author={Chen, Ze and Chen, Lan and Li, Yuanhang and Mao, Qi},
  journal={arXiv preprint arXiv:2604.22586},
  year={2026}
}
```

---

## 许可证

本项目仅供研究用途。
