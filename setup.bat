@echo off
chcp 65001 >nul

REM FlowAnchor Windows 环境安装脚本

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo === FlowAnchor 安装 ===

REM 1. 克隆 FiVE-Bench
if not exist "FiVE-Bench" (
    echo 正在克隆 FiVE-Bench...
    git clone https://github.com/MinghanLi/FiVE-Bench.git
)

REM 2. 安装依赖
echo 正在安装依赖...
pip install opencv-python numpy pillow tqdm psutil

REM 3. 安装 wan 包
set WAN_EDIT_DIR=%SCRIPT_DIR%FiVE-Bench\models\wan-edit
if exist "%WAN_EDIT_DIR%\wan" (
    echo 正在安装 wan 包...
    cd /d "%WAN_EDIT_DIR%"
    pip install -e .
    cd /d "%SCRIPT_DIR%"
)

REM 4. 检查模型
set CKPT_DIR=%SCRIPT_DIR%checkpoints\Wan-AI\Wan2.1-T2V-1.3B
if not exist "%CKPT_DIR%" (
    echo.
    echo === 需要下载模型 ===
    echo 请下载 Wan2.1-T2V-1.3B 模型 ^(~5.5GB^):
    echo.
    echo   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir %CKPT_DIR%
    echo.
) else (
    echo 模型已存在: %CKPT_DIR%
)

echo.
echo === 安装完成 ===
echo.
echo 快速开始:
echo   run.bat ^<视频路径^> "源提示词" "目标提示词"
echo.
echo 示例:
echo   run.bat data\car.mp4 "a red car on the road" "a blue car on the road"
