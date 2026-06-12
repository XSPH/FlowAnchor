# FlowAnchor - 视频编辑复现

基于 Wan-Edit 复现论文 **FlowAnchor: Stabilizing the Editing Signal for Inversion-Free Video Editing** (arXiv:2604.22586)。

## 这是什么？

FlowAnchor 是一种**无需训练、无需反演**的视频编辑方法。相比传统的反演方法，它直接在采样轨迹上施加编辑信号，速度快且结构保持好。

核心解决两个问题：
1. **SAR (空间感知注意力精炼)** — 防止编辑信号泄漏到错误区域（多物体场景）
2. **AMM (自适应幅度调制)** — 补偿帧数增加导致的信号衰减

## 系统要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA，显存 ≥ 8GB |
| 系统 | Windows / Linux / macOS (仅 CPU 推理) |
| Python | 3.8+ |
| PyTorch | 2.0+ |
| CUDA | 11.8+ |

**已测试硬件：** RTX 4060 Laptop 8GB

## 安装步骤

### Windows

```cmd
:: 1. 进入项目目录
cd FlowAnchor

:: 2. 运行安装脚本
setup.bat

:: 3. 下载模型 (约5.5GB)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir checkpoints\Wan-AI\Wan2.1-T2V-1.3B
```

### Linux

```bash
cd FlowAnchor
bash setup.sh

# 下载模型
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir checkpoints/Wan-AI/Wan2.1-T2V-1.3B
```

### 模型选择

| 模型 | 显存需求 | 质量 | 下载命令 |
|------|---------|------|---------|
| Wan2.1-T2V-1.3B | ≥ 8GB | 良好 | `huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B` |
| Wan2.1-T2V-14B | ≥ 24GB | 最佳 | `huggingface-cli download Wan-AI/Wan2.1-T2V-14B` |

## 使用方法

### 方法一：单视频编辑

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

### 方法二：直接运行 Python

```cmd
:: 设置 Python 路径
set PYTHONPATH=FiVE-Bench\models\wan-edit;%PYTHONPATH%

:: 编辑视频
python edit_flowanchor.py ^
    --task t2v-1.3B ^
    --ckpt_dir checkpoints\Wan-AI\Wan2.1-T2V-1.3B ^
    --video_path data\car.mp4 ^
    --prompt "a red car on the road" ^
    --tgt_prompt "a blue car on the road" ^
    --save_dir outputs
```

### 方法三：FiVE-Bench 批量评估

```cmd
python eval_five.py ^
    --ckpt_dir checkpoints\Wan-AI\Wan2.1-T2V-1.3B ^
    --FiVE_dataset_json data_FiVE\edit_prompt\edit1_FiVE.json ^
    --data_dir data ^
    --save_dir outputs_flowanchor
```

## 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--prompt` | (必填) | 源视频的描述文本 |
| `--tgt_prompt` | (必填) | 编辑后的目标描述文本 |
| `--video_path` | (必填) | 输入视频路径 (.mp4) |
| `--mask_path` | 无 | 遮罩路径，支持 .mp4 / .png / 文件夹 |
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
| `--sample_steps` | 50 | 去噪步数，越多质量越好但越慢 |
| `--sample_shift` | 5.0 | Flow matching 偏移参数 |
| `--sample_guide_scale` | 5.0 | 源提示词 CFG 引导强度 |
| `--tgt_guide_scale` | 10.0 | 目标提示词 CFG 引导强度 |
| `--skip_timesteps` | 16 | 跳过的初始步数 |
| `--base_seed` | -1 | 随机种子，-1 为随机 |

### 输出控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save_dir` | outputs | 输出目录 |
| `--save_file` | 自动生成 | 指定输出文件名 |
| `--size` | 832*480 | 输出分辨率 (宽*高) |
| `--frame_num` | 41 | 输出帧数 |

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

## 工作原理

### 基线方法：Wan-Edit

Wan-Edit 将 FlowEdit 应用于视频，计算编辑信号：

```
ΔV = V(Z_tgt, t, P_tgt) - V(Z_src, t, P_src)
```

这个信号引导去噪轨迹从源分布转向目标分布。

### FlowAnchor 的改进

**问题1：定位不精确**
- 编辑信号泄漏到多物体场景的错误区域
- SAR 通过调制交叉注意力图来强制空间对齐

**问题2：信号衰减**
- 随帧数增加，信号幅度衰减
- AMM 使用归一化对比度图自适应放大信号

### FlowAnchor 编辑信号

```
ΔV_flowanchor = ΔV + γ_F × (C ⊙ ΔV)
```

其中 C 是从信号本身导出的归一化对比度图。

## 常见问题

### Q: 没有 GPU 能用吗？
A: 可以但非常慢。代码支持 CPU 推理，但一个 41 帧视频可能需要几小时。

### Q: 遮罩必须提供吗？
A: 不必须。有遮罩时 SAR + AMM 都会启用，效果最好。没有遮罩时只用 AMM。

### Q: 如何生成遮罩？
A: 可以用 SAM、Grounded-SAM 等分割工具，或者手动用画图工具画白色区域。

### Q: Windows 下报错 `No module named 'wan'`
A: 需要设置 PYTHONPATH。运行 `set PYTHONPATH=FiVE-Bench\models\wan-edit;%PYTHONPATH%` 或使用 `run.bat`。

### Q: 显存不够怎么办？
A: 使用 `--offload_model True`（默认开启），模型会在推理时自动加载/卸载到 GPU。

## 引用

```bibtex
@article{chen2026flowanchor,
  title={FlowAnchor: Stabilizing the Editing Signal for Inversion-Free Video Editing},
  author={Chen, Ze and Chen, Lan and Li, Yuanhang and Mao, Qi},
  journal={arXiv preprint arXiv:2604.22586},
  year={2026}
}
```
