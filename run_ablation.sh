#!/bin/bash
# FlowAnchor 消融实验 (对应论文 Table 2 / Table 3)
# 用法: bash run_ablation.sh <video_path> <src_prompt> <tgt_prompt> <mask_path> [target_words...]
# 所有配置共用同一个 seed, 结果存到 outputs/ablation/<配置名>.mp4

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WAN_EDIT_DIR="${WAN_EDIT_DIR:-$SCRIPT_DIR/FiVE-Bench/models/wan-edit}"
CKPT_DIR="${CKPT_DIR:-$SCRIPT_DIR/checkpoints/Wan-AI/Wan2.1-T2V-1.3B}"
SEED="${SEED:-42}"
SAVE_DIR="${SAVE_DIR:-$SCRIPT_DIR/outputs/ablation}"

if [ $# -lt 4 ]; then
    echo "Usage: bash run_ablation.sh <video_path> <src_prompt> <tgt_prompt> <mask_path> [target_words...]"
    echo "消融实验需要 mask, 否则 SAR 相关配置无意义"
    exit 1
fi

VIDEO_PATH="$1"; SRC_PROMPT="$2"; TGT_PROMPT="$3"; MASK_PATH="$4"
shift 4
TARGET_WORDS=("$@")

export PYTHONPATH="$WAN_EDIT_DIR:$PYTHONPATH"
mkdir -p "$SAVE_DIR"

run_cfg() {
    local name="$1"; shift
    local out="$SAVE_DIR/${name}.mp4"
    if [ -f "$out" ]; then
        echo ">>> [$name] 已存在, 跳过"
        return
    fi
    echo ">>> [$name] $*"
    local extra=()
    [ ${#TARGET_WORDS[@]} -gt 0 ] && extra=(--target_words "${TARGET_WORDS[@]}")
    python "$SCRIPT_DIR/edit_flowanchor.py" \
        --task t2v-1.3B \
        --ckpt_dir "$CKPT_DIR" \
        --video_path "$VIDEO_PATH" \
        --prompt "$SRC_PROMPT" \
        --tgt_prompt "$TGT_PROMPT" \
        --mask_path "$MASK_PATH" \
        --save_file "$out" \
        --base_seed "$SEED" \
        --offload_model True \
        "${extra[@]}" \
        "$@"
}

# ---- 完整方法 (论文默认: beta1=beta2=0.3, gamma=1.0, tau=0.6) ----
run_cfg "full"

# ---- Table 2: 模块消融 ----
run_cfg "wo_TTM"  --beta1 0.0                 # w/o Text-Token Modulation
run_cfg "wo_STM"  --beta2 0.0                 # w/o Spatio-Temporal Modulation
run_cfg "wo_AMM"  --gamma 0.0                 # w/o AMM
run_cfg "wo_SAR"  --no_sar                    # SAR 整体关闭 (TTM+STM)
run_cfg "wanedit_baseline" --no_sar --gamma 0.0   # 两模块全关 = Wan-Edit(论文采样设置)

# ---- Table 3: 超参敏感性 ----
run_cfg "sar_0.1" --beta1 0.1 --beta2 0.1     # SAR (0.1, 0.1)
run_cfg "sar_0.5" --beta1 0.5 --beta2 0.5     # SAR (0.5, 0.5)
run_cfg "amm_0.5" --gamma 0.5                 # AMM gamma=0.5
run_cfg "amm_1.5" --gamma 1.5                 # AMM gamma=1.5
run_cfg "tau_0.8" --sar_tau 0.8               # SAR 范围 [T, 0.8T] (更短)
run_cfg "tau_0.4" --sar_tau 0.4               # SAR 范围 [T, 0.4T] (更长)

echo ""
echo "全部完成, 结果在 $SAVE_DIR/"
ls -1 "$SAVE_DIR"
