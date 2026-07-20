"""
mask_gui.py — 图形化框选编辑区域 mask (本地用, 生成后上传服务器)

用法:
    python mask_gui.py --video data/my_video3.mp4
    python mask_gui.py --video data/my_video3.mp4 --out masks/sweater.png

操作:
    鼠标左键拖拽      框一个矩形 (可框多个, 自动取并集)
    A / D             上一帧 / 下一帧 (检查运动目标是否始终在框内)
    空格              自动播放 / 暂停
    U 或 鼠标右键     撤销上一个框
    R                 清空所有框
    S                 保存 mask 并打印上传/运行命令
    Q / ESC           退出

说明:
    - mask 按原视频分辨率保存 (管线里会和视频一起 resize, 保持对齐);
    - 目标会动的话把框画大一圈, 覆盖它在所有帧扫过的范围即可
      (论文 Table 4: FlowAnchor 对粗糙 bounding box mask 鲁棒);
    - 绿色半透明区域 = mask 内 (SAR 允许编辑), 其余 = mask 外 (SAR 抑制).
"""
import argparse
import os

import cv2
import numpy as np

WINDOW = "FlowAnchor mask (S=save  Q=quit  A/D=frame  Space=play  U=undo  R=reset)"


def load_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video_path}")
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit("视频没有可读帧")
    return frames


class MaskGUI:
    def __init__(self, frames, video_path, out_path, max_w=1280, max_h=720):
        self.frames = frames
        self.video_path = video_path
        self.out_path = out_path
        self.h, self.w = frames[0].shape[:2]
        self.scale = min(1.0, max_w / self.w, max_h / self.h)
        self.boxes = []           # [(x1, y1, x2, y2)] 原始分辨率坐标
        self.drag_start = None    # 显示坐标
        self.drag_cur = None
        self.cur = 0
        self.playing = False
        self.saved = False

    # ---------- 坐标 ----------
    def to_orig(self, x, y):
        return int(round(x / self.scale)), int(round(y / self.scale))

    def mask(self):
        m = np.zeros((self.h, self.w), dtype=np.uint8)
        for x1, y1, x2, y2 in self.boxes:
            m[y1:y2, x1:x2] = 255
        return m

    # ---------- 交互 ----------
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_cur = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            ox1, oy1 = self.to_orig(*self.drag_start)
            ox2, oy2 = self.to_orig(x, y)
            x1, x2 = sorted((max(0, ox1), min(self.w, ox2)))
            y1, y2 = sorted((max(0, oy1), min(self.h, oy2)))
            if x2 - x1 >= 4 and y2 - y1 >= 4:   # 忽略误点
                self.boxes.append((x1, y1, x2, y2))
            self.drag_start = self.drag_cur = None
        elif event == cv2.EVENT_RBUTTONDOWN and self.boxes:
            self.boxes.pop()

    # ---------- 绘制 ----------
    def render(self):
        frame = self.frames[self.cur]
        if self.scale < 1.0:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale)
        vis = frame.copy()

        if self.boxes:
            m = self.mask()
            if self.scale < 1.0:
                m = cv2.resize(m, (vis.shape[1], vis.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            green = np.zeros_like(vis)
            green[:, :, 1] = 255
            sel = m > 0
            vis[sel] = cv2.addWeighted(vis, 0.55, green, 0.45, 0)[sel]
            for x1, y1, x2, y2 in self.boxes:
                p1 = (int(x1 * self.scale), int(y1 * self.scale))
                p2 = (int(x2 * self.scale), int(y2 * self.scale))
                cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)

        if self.drag_start is not None and self.drag_cur is not None:
            cv2.rectangle(vis, self.drag_start, self.drag_cur, (0, 255, 255), 2)

        cover = 100.0 * float(self.mask().mean()) / 255.0 if self.boxes else 0.0
        hud = (f"frame {self.cur + 1}/{len(self.frames)}  "
               f"boxes {len(self.boxes)}  cover {cover:.1f}%"
               + ("  [PLAYING]" if self.playing else ""))
        cv2.putText(vis, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    # ---------- 保存 ----------
    def save(self):
        if not self.boxes:
            print("还没有框任何区域, 未保存.")
            return
        os.makedirs(os.path.dirname(self.out_path) or '.', exist_ok=True)
        cv2.imwrite(self.out_path, self.mask())
        cover = 100.0 * float(self.mask().mean()) / 255.0
        self.saved = True
        rel = os.path.relpath(self.out_path)
        if rel.startswith('..'):
            rel = os.path.abspath(self.out_path)
        print(f"\nmask 已保存: {self.out_path}  ({self.w}x{self.h}, 覆盖 {cover:.1f}%)")
        print("\n--- 上传到服务器 (按需替换用户名/地址) ---")
        print(f"  scp {rel} root@<服务器IP>:~/FlowAnchor/{rel}")
        print("\n--- 服务器上运行 ---")
        print(f"  SEED=42 bash run.sh {self.video_path} '<源prompt>' '<目标prompt>' \\")
        print(f"      {rel} <target词...>")
        print("  (日志需出现 'Loaded mask' 和 'SAR target token indices' 才说明 SAR 生效)\n")

    # ---------- 主循环 ----------
    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        n = len(self.frames)
        while True:
            cv2.imshow(WINDOW, self.render())
            key = cv2.waitKey(66 if self.playing else 30) & 0xFF
            if self.playing:
                self.cur = (self.cur + 1) % n
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                self.save()
            elif key in (ord('a'), 81):     # 81 = 左方向键(部分平台)
                self.playing = False
                self.cur = (self.cur - 1) % n
            elif key in (ord('d'), 83):     # 83 = 右方向键
                self.playing = False
                self.cur = (self.cur + 1) % n
            elif key == ord(' '):
                self.playing = not self.playing
            elif key == ord('u') and self.boxes:
                self.boxes.pop()
            elif key == ord('r'):
                self.boxes = []
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
        if self.boxes and not self.saved:
            ans = input("已画框但未保存, 现在保存? [Y/n] ").strip().lower()
            if ans in ('', 'y', 'yes'):
                self.save()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="源视频路径")
    p.add_argument("--out", default=None,
                   help="mask 输出 png (默认 masks/<视频名>_mask.png)")
    p.add_argument("--num_frames", type=int, default=41,
                   help="预览的帧数 (与管线一致, 默认 41)")
    args = p.parse_args()

    out = args.out
    if out is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        out = os.path.join("masks", f"{stem}_mask.png")

    frames = load_frames(args.video, args.num_frames)
    print(f"视频分辨率: {frames[0].shape[1]}x{frames[0].shape[0]}, "
          f"预览 {len(frames)} 帧")
    if frames[0].shape[0] > frames[0].shape[1]:
        print("注意: 这是竖屏视频, 服务器上要用 SIZE=480*832 bash run.sh ...")

    gui = MaskGUI(frames, args.video, out)
    try:
        gui.run()
    except cv2.error as e:
        raise SystemExit(
            f"OpenCV 窗口创建失败 (无显示环境?): {e}\n"
            "无图形界面时可改用: python make_mask.py --video ... --box x1 y1 x2 y2 --out ...")


if __name__ == "__main__":
    main()
