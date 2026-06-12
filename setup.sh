#!/bin/bash
# FlowAnchor: Environment setup script
# Clones FiVE-Bench (with wan-edit), installs dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== FlowAnchor Setup ==="

# 1. Clone FiVE-Bench if not exists
if [ ! -d "FiVE-Bench" ]; then
    echo "Cloning FiVE-Bench..."
    git clone https://github.com/MinghanLi/FiVE-Bench.git
fi

# 2. Install Python dependencies
echo "Installing dependencies..."
pip install -q opencv-python numpy pillow tqdm psutil

# 3. Install wan package from FiVE-Bench
WAN_EDIT_DIR="$SCRIPT_DIR/FiVE-Bench/models/wan-edit"
if [ -d "$WAN_EDIT_DIR/wan" ]; then
    echo "Installing wan package..."
    cd "$WAN_EDIT_DIR"
    pip install -e . 2>/dev/null || echo "wan package install skipped (may already be installed)"
    cd "$SCRIPT_DIR"
fi

# 4. Download Wan2.1-T2V-1.3B model (recommended for 8GB VRAM)
CKPT_DIR="$SCRIPT_DIR/checkpoints/Wan-AI/Wan2.1-T2V-1.3B"
if [ ! -d "$CKPT_DIR" ]; then
    echo ""
    echo "=== Model Download Required ==="
    echo "Please download the Wan2.1-T2V-1.3B model (~5.5GB):"
    echo ""
    echo "  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir $CKPT_DIR"
    echo ""
    echo "Or for better quality (requires ~24GB VRAM):"
    echo "  huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir $SCRIPT_DIR/checkpoints/Wan-AI/Wan2.1-T2V-14B"
    echo ""
else
    echo "Model found at $CKPT_DIR"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick start:"
echo "  bash run.sh <video_path> '<src_prompt>' '<tgt_prompt>'"
echo ""
echo "Example:"
echo "  bash run.sh data/car.mp4 'a red car on the road' 'a blue car on the road'"
echo ""
echo "Batch evaluation:"
export PYTHONPATH="$WAN_EDIT_DIR:$PYTHONPATH"
echo "  python eval_five.py --ckpt_dir $CKPT_DIR --FiVE_dataset_json data_FiVE/edit_prompt/edit1_FiVE.json --data_dir data"
