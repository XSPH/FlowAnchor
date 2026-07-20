"""
快速生成 FlowAnchor 可用的编辑区域 mask.

论文 Table 4 / Fig. 10: FlowAnchor 对粗糙 mask 鲁棒, bounding box 与精确
分割 mask 的指标几乎相同, 所以手工框一个矩形通常就够用.

用法:
  1) 先导出首帧, 用看图软件读出目标的像素坐标:
       python make_mask.py --video data/car.mp4 --dump_frame frame0.png
  2) 按 (x1 y1 x2 y2) 生成静态 mask (自动匹配视频分辨率):
       python make_mask.py --video data/car.mp4 --box 300 120 520 400 --out masks/car.png
  3) 编辑:
       bash run.sh data/car.mp4 "a red car ..." "a blue car ..." masks/car.png

  --out 以目录结尾(如 masks/car/)时, 会展开成逐帧 png 序列, 方便在个别帧上
  手动修正后使用.
"""
import argparse
import os

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="源视频 (用于取分辨率/帧数/首帧)")
    p.add_argument("--dump_frame", default=None, metavar="PNG",
                   help="仅导出首帧到该文件, 便于确定 box 坐标")
    p.add_argument("--box", type=int, nargs=4, default=None,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="编辑区域矩形 (原视频像素坐标)")
    p.add_argument("--out", default=None,
                   help="mask 输出: .png 为静态单帧; 以 / 结尾为逐帧序列目录")
    p.add_argument("--num_frames", type=int, default=41)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {args.video}")
    ret, frame0 = cap.read()
    if not ret:
        raise SystemExit("视频没有可读帧")
    h, w = frame0.shape[:2]
    print(f"视频分辨率: {w}x{h}")

    if args.dump_frame:
        cv2.imwrite(args.dump_frame, frame0)
        print(f"首帧已导出: {args.dump_frame}  (在看图软件里读取目标区域的像素坐标)")
        return

    if args.box is None or args.out is None:
        raise SystemExit("需要 --box x1 y1 x2 y2 和 --out (或用 --dump_frame 先看坐标)")

    x1, y1, x2, y2 = args.box
    x1, x2 = sorted((max(0, x1), min(w, x2)))
    y1, y2 = sorted((max(0, y1), min(h, y2)))
    if x2 <= x1 or y2 <= y1:
        raise SystemExit(f"无效 box: {args.box}")

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    cover = 100.0 * (x2 - x1) * (y2 - y1) / (h * w)
    print(f"box ({x1},{y1})-({x2},{y2}), 覆盖画面 {cover:.1f}%")

    if args.out.endswith('/') or os.path.isdir(args.out):
        os.makedirs(args.out, exist_ok=True)
        for i in range(args.num_frames):
            cv2.imwrite(os.path.join(args.out, f"{i + 1:05d}.png"), mask)
        print(f"已写出 {args.num_frames} 帧 mask 序列到 {args.out}")
    else:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        cv2.imwrite(args.out, mask)
        print(f"静态 mask 已写出: {args.out} (load_mask 会自动扩展到所有帧)")


if __name__ == "__main__":
    main()
