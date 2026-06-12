@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM FlowAnchor Windows 启动脚本
REM 用法: run.bat <视频路径> <源提示词> <目标提示词> [遮罩路径]

if "%~3"=="" (
    echo 用法: run.bat ^<视频路径^> ^<源提示词^> ^<目标提示词^> [遮罩路径]
    echo.
    echo 示例:
    echo   run.bat data\car.mp4 "a red car" "a blue car"
    echo   run.bat data\car.mp4 "a red car" "a blue car" masks\car_mask.mp4
    exit /b 1
)

set SCRIPT_DIR=%~dp0
set WAN_EDIT_DIR=%WAN_EDIT_DIR%
if "%WAN_EDIT_DIR%"=="" set WAN_EDIT_DIR=%SCRIPT_DIR%..\FiVE-Bench\models\wan-edit
set CKPT_DIR=%CKPT_DIR%
if "%CKPT_DIR%"=="" set CKPT_DIR=%SCRIPT_DIR%checkpoints\Wan-AI\Wan2.1-T2V-1.3B

set VIDEO_PATH=%~1
set SRC_PROMPT=%~2
set TGT_PROMPT=%~3
set MASK_PATH=%~4

echo === FlowAnchor: 视频编辑 ===
echo 源视频: %VIDEO_PATH%
echo 源提示词: %SRC_PROMPT%
echo 目标提示词: %TGT_PROMPT%
if not "%MASK_PATH%"=="" echo 遮罩: %MASK_PATH%
echo.

if not exist "%WAN_EDIT_DIR%" (
    echo 错误: 找不到 wan-edit 目录: %WAN_EDIT_DIR%
    echo 请设置 WAN_EDIT_DIR 环境变量
    exit /b 1
)

if not exist "%CKPT_DIR%" (
    echo 错误: 找不到模型目录: %CKPT_DIR%
    echo 请下载模型:
    echo   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir %CKPT_DIR%
    exit /b 1
)

set EXTRA_ARGS=
if not "%MASK_PATH%"=="" set EXTRA_ARGS=--mask_path "%MASK_PATH%"

set PYTHONPATH=%WAN_EDIT_DIR%;%PYTHONPATH%

python "%SCRIPT_DIR%edit_flowanchor.py" ^
    --task t2v-1.3B ^
    --ckpt_dir "%CKPT_DIR%" ^
    --video_path "%VIDEO_PATH%" ^
    --prompt "%SRC_PROMPT%" ^
    --tgt_prompt "%TGT_PROMPT%" ^
    --save_dir "%SCRIPT_DIR%outputs" ^
    --sample_steps 50 ^
    --sample_shift 5.0 ^
    --sample_guide_scale 5.0 ^
    --tgt_guide_scale 10.0 ^
    --skip_timesteps 16 ^
    --beta1 0.5 ^
    --beta2 0.5 ^
    --gamma_scale 1.0 ^
    --offload_model True ^
    %EXTRA_ARGS%
