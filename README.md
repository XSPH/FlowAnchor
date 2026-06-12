<h1 align="center">FlowAnchor</h1>

<p align="center"><strong>Stabilizing the Editing Signal for Inversion-Free Video Editing</strong></p>

<p align="center">
  <a href="README.zh.md">中文</a> | English
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.22586">Paper</a> | <a href="https://cuc-mipg.github.io/FlowAnchor.github.io/">Project Page</a>
</p>

---

Reproduction of **FlowAnchor: Stabilizing the Editing Signal for Inversion-Free Video Editing** (arXiv:2604.22586) based on the [Wan-Edit](https://github.com/MinghanLi/FiVE-Bench/tree/main/models/wan-edit) codebase.

FlowAnchor is a **training-free, inversion-free** video editing method. It directly steers the sampling trajectory with a stabilized editing signal, achieving fast and structure-preserving edits across diverse scenarios.

---

## Quick Start

### Windows

```cmd
:: Install
setup.bat

:: Download model (~5.5GB)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir checkpoints\Wan-AI\Wan2.1-T2V-1.3B

:: Edit video
run.bat data\car.mp4 "a red car" "a blue car"
```

### Linux

```bash
# Install
bash setup.sh

# Download model (~5.5GB)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir checkpoints/Wan-AI/Wan2.1-T2V-1.3B

# Edit video
bash run.sh data/car.mp4 "a red car" "a blue car"
```

---

## Requirements

| Item | Requirement |
|------|-------------|
| GPU | NVIDIA, VRAM ≥ 8GB |
| OS | Windows / Linux |
| Python | 3.8+ |
| PyTorch | 2.0+ |
| CUDA | 11.8+ |

**Tested on:** RTX 4060 Laptop 8GB

---

## Core Mechanisms

FlowAnchor addresses two key failure modes in inversion-free video editing:

### SAR (Spatial-aware Attention Refinement)

Prevents editing signal leakage in multi-object scenes by modulating cross-attention maps at both text-token and spatio-temporal levels.

```
A''_i,j = A'_i,j + β₂(A'^max_j - A'_i,j)   if M_i=1, j∈J_tar
```

### AMM (Adaptive Magnitude Modulation)

Compensates for signal attenuation caused by increased frame counts using a normalized contrast map with frame-aware scaling.

```
ΔV_flowanchor = ΔV + γ_F × (C ⊙ ΔV)
```

---

## Usage

### Single Video Editing

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

### Direct Python

```bash
python edit_flowanchor.py \
    --task t2v-1.3B \
    --ckpt_dir checkpoints/Wan-AI/Wan2.1-T2V-1.3B \
    --video_path data/car.mp4 \
    --prompt "a red car on the road" \
    --tgt_prompt "a blue car on the road" \
    --save_dir outputs
```

### FiVE-Bench Batch Evaluation

```bash
python eval_five.py \
    --ckpt_dir checkpoints/Wan-AI/Wan2.1-T2V-1.3B \
    --FiVE_dataset_json data_FiVE/edit_prompt/edit1_FiVE.json \
    --data_dir data \
    --save_dir outputs_flowanchor
```

---

## Parameters

### Core

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt` | (required) | Source video description |
| `--tgt_prompt` | (required) | Target edited description |
| `--video_path` | (required) | Input video path (.mp4) |
| `--mask_path` | none | Mask path (.mp4 / .png / directory) |
| `--target_words` | none | Target words for SAR attention modulation |

### FlowAnchor Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--beta1` | 0.5 | SAR text-token modulation strength |
| `--beta2` | 0.5 | SAR spatio-temporal modulation strength |
| `--gamma_scale` | 1.0 | AMM signal amplification scale |

### Sampling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sample_steps` | 50 | Denoising steps |
| `--sample_shift` | 5.0 | Flow matching shift parameter |
| `--sample_guide_scale` | 5.0 | Source prompt CFG scale |
| `--tgt_guide_scale` | 10.0 | Target prompt CFG scale |
| `--skip_timesteps` | 16 | Steps to skip at beginning |
| `--base_seed` | -1 | Random seed (-1 for random) |

---

## Project Structure

```
FlowAnchor/
├── flowanchor.py        # SAR + AMM core algorithm
├── edit_flowanchor.py   # Single video editing pipeline
├── eval_five.py         # FiVE-Bench batch evaluation
├── run.bat              # Windows launcher
├── run.sh               # Linux launcher
├── setup.bat            # Windows setup
├── setup.sh             # Linux setup
├── test_sanity.py       # Code verification
└── README.md
```

---

## Model Selection

| Model | VRAM | Quality | Download |
|-------|------|---------|----------|
| Wan2.1-T2V-1.3B | ≥ 8GB | Good | `huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B` |
| Wan2.1-T2V-14B | ≥ 24GB | Best | `huggingface-cli download Wan-AI/Wan2.1-T2V-14B` |

---

## FAQ

| Question | Answer |
|----------|--------|
| Can I run without GPU? | Yes but very slow; 41 frames may take hours |
| Is a mask required? | No. SAR+AMM both activate with mask for best results |
| How to generate masks? | Use SAM / Grounded-SAM or draw white regions manually |
| `No module named 'wan'` | Set PYTHONPATH or use run.bat / run.sh |
| Out of VRAM? | `--offload_model True` (default) auto offloads model |

---

## Citation

```bibtex
@article{chen2026flowanchor,
  title={FlowAnchor: Stabilizing the Editing Signal for Inversion-Free Video Editing},
  author={Chen, Ze and Chen, Lan and Li, Yuanhang and Mao, Qi},
  journal={arXiv preprint arXiv:2604.22586},
  year={2026}
}
```

---

## License

This project is for research purposes only.
