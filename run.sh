#!/bin/bash
# FlowAnchor: Quick start script
# Usage: bash run.sh <video_path> <src_prompt> <tgt_prompt> [mask_path] [target_words...]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WAN_EDIT_DIR="${WAN_EDIT_DIR:-$SCRIPT_DIR/FiVE-Bench/models/wan-edit}"
CKPT_DIR="${CKPT_DIR:-$SCRIPT_DIR/checkpoints/Wan-AI/Wan2.1-T2V-1.3B}"
SIZE="${SIZE:-832*480}"   # 竖屏视频用: SIZE=480*832 bash run.sh ...

if [ $# -lt 3 ]; then
    echo "Usage: bash run.sh <video_path> <src_prompt> <tgt_prompt> [mask_path] [target_words...]"
    echo ""
    echo "Example:"
    echo "  bash run.sh data/my_video.mp4 'a red car' 'a blue car'"
    echo "  bash run.sh data/my_video.mp4 'a red car' 'a blue car' masks/car_mask.mp4 blue"
    echo ""
    echo "Environment variables:"
    echo "  WAN_EDIT_DIR  - Path to wan-edit directory (default: ./FiVE-Bench/models/wan-edit)"
    echo "  CKPT_DIR      - Path to Wan model checkpoints (default: ./checkpoints/Wan-AI/Wan2.1-T2V-1.3B)"
    exit 1
fi

VIDEO_PATH="$1"
SRC_PROMPT="$2"
TGT_PROMPT="$3"
MASK_PATH="${4:-}"
shift 3
[ $# -gt 0 ] && shift  # drop mask_path if present
TARGET_WORDS=("$@")

echo "=== FlowAnchor: Inversion-Free Video Editing ==="
echo "Source: $VIDEO_PATH"
echo "Source prompt: $SRC_PROMPT"
echo "Target prompt: $TGT_PROMPT"
if [ -n "$MASK_PATH" ]; then
    echo "Mask: $MASK_PATH"
else
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! 警告: 未提供 mask, SAR (论文核心模块 TTM+STM) 将被完全禁用 !!!"
    echo "!!! 实际运行的是 Wan-Edit + AMM, 效果会远差于论文.              !!!"
    echo "!!! 用 make_mask.py 生成 mask 后作为第 4 个参数传入.            !!!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo ""
fi
[ ${#TARGET_WORDS[@]} -gt 0 ] && echo "Target words: ${TARGET_WORDS[*]}"
echo ""

if [ ! -d "$WAN_EDIT_DIR" ]; then
    echo "Error: WAN_EDIT_DIR not found: $WAN_EDIT_DIR"
    echo "Run: git submodule update --init"
    exit 1
fi

if [ ! -d "$CKPT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CKPT_DIR"
    echo "Please download Wan2.1-T2V-1.3B model to $CKPT_DIR"
    echo "  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir $CKPT_DIR"
    exit 1
fi

EXTRA_ARGS=""
[ -n "$MASK_PATH" ] && EXTRA_ARGS="$EXTRA_ARGS --mask_path $MASK_PATH"
[ ${#TARGET_WORDS[@]} -gt 0 ] && EXTRA_ARGS="$EXTRA_ARGS --target_words ${TARGET_WORDS[*]}"
[ -n "$SEED" ] && EXTRA_ARGS="$EXTRA_ARGS --base_seed $SEED"   # SEED=42 bash run.sh ... 可复现

export PYTHONPATH="$WAN_EDIT_DIR:$PYTHONPATH"

# Paper settings (arXiv 2604.22586, Sec. 4.1 / Sec. B):
#   T=25 steps, skip first 2 (n_max=23), Euler update (Eq. 13), shift 5,
#   beta1=beta2=0.3, gamma=1.0, tau=0.6, F0=21.
python "$SCRIPT_DIR/edit_flowanchor.py" \
    --task t2v-1.3B \
    --size "$SIZE" \
    --ckpt_dir "$CKPT_DIR" \
    --video_path "$VIDEO_PATH" \
    --prompt "$SRC_PROMPT" \
    --tgt_prompt "$TGT_PROMPT" \
    --save_dir "$SCRIPT_DIR/outputs" \
    --sample_solver euler \
    --sample_steps 25 \
    --sample_shift 5.0 \
    --sample_guide_scale 5.0 \
    --tgt_guide_scale 10.0 \
    --skip_timesteps 2 \
    --beta1 0.3 \
    --beta2 0.3 \
    --gamma 1.0 \
    --sar_tau 0.6 \
    --offload_model True \
    $EXTRA_ARGS
