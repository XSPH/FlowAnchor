#!/bin/bash
# FlowAnchor: Quick start script
# Usage: bash run.sh <video_path> <src_prompt> <tgt_prompt> [mask_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WAN_EDIT_DIR="${WAN_EDIT_DIR:-$SCRIPT_DIR/../FiVE-Bench/models/wan-edit}"
CKPT_DIR="${CKPT_DIR:-$SCRIPT_DIR/checkpoints/Wan-AI/Wan2.1-T2V-1.3B}"

if [ $# -lt 3 ]; then
    echo "Usage: bash run.sh <video_path> <src_prompt> <tgt_prompt> [mask_path]"
    echo ""
    echo "Example:"
    echo "  bash run.sh data/my_video.mp4 'a red car' 'a blue car'"
    echo "  bash run.sh data/my_video.mp4 'a red car' 'a blue car' masks/car_mask.mp4"
    echo ""
    echo "Environment variables:"
    echo "  WAN_EDIT_DIR  - Path to wan-edit directory (default: ../FiVE-Bench/models/wan-edit)"
    echo "  CKPT_DIR      - Path to Wan model checkpoints (default: ./checkpoints/Wan-AI/Wan2.1-T2V-1.3B)"
    exit 1
fi

VIDEO_PATH="$1"
SRC_PROMPT="$2"
TGT_PROMPT="$3"
MASK_PATH="${4:-}"

echo "=== FlowAnchor: Inversion-Free Video Editing ==="
echo "Source: $VIDEO_PATH"
echo "Source prompt: $SRC_PROMPT"
echo "Target prompt: $TGT_PROMPT"
[ -n "$MASK_PATH" ] && echo "Mask: $MASK_PATH"
echo ""

if [ ! -d "$WAN_EDIT_DIR" ]; then
    echo "Error: WAN_EDIT_DIR not found: $WAN_EDIT_DIR"
    echo "Please set WAN_EDIT_DIR to the wan-edit directory path"
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

export PYTHONPATH="$WAN_EDIT_DIR:$PYTHONPATH"

python "$SCRIPT_DIR/edit_flowanchor.py" \
    --task t2v-1.3B \
    --ckpt_dir "$CKPT_DIR" \
    --video_path "$VIDEO_PATH" \
    --prompt "$SRC_PROMPT" \
    --tgt_prompt "$TGT_PROMPT" \
    --save_dir "$SCRIPT_DIR/outputs" \
    --sample_steps 50 \
    --sample_shift 5.0 \
    --sample_guide_scale 5.0 \
    --tgt_guide_scale 10.0 \
    --skip_timesteps 16 \
    --beta1 0.5 \
    --beta2 0.5 \
    --gamma_scale 1.0 \
    --offload_model True \
    $EXTRA_ARGS
