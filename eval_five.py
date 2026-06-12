"""
FlowAnchor: Batch evaluation on FiVE-Bench dataset
"""
import argparse
import json
import os
import sys
import time
import logging
import warnings

warnings.filterwarnings('ignore')

import torch
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FiVE-Bench', 'models', 'wan-edit'))
from wan.utils.utils import str2bool
from edit_flowanchor import (
    load_frames, load_frames_path, load_mask, flowanchor_edit, _init_logging
)
import wan
from wan.configs import WAN_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(description="FlowAnchor batch evaluation")
    parser.add_argument("--task", type=str, default="t2v-1.3B")
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--frame_num", type=int, default=41)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--offload_model", type=str2bool, default=True)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="outputs_flowanchor")
    parser.add_argument("--FiVE_dataset_json", type=str, required=True)
    parser.add_argument("--sample_solver", type=str, default='unipc')
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--tgt_guide_scale", type=float, default=10.0)
    parser.add_argument("--skip_timesteps", type=int, default=16)
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.5)
    parser.add_argument("--gamma_scale", type=float, default=1.0)
    parser.add_argument("--eval_gpu_time", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = world_size <= 1

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        import torch.distributed as dist
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size)

    cfg = WAN_CONFIGS[args.task]
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_cpu=args.t5_cpu,
        use_usp=False,
    )

    with open(args.FiVE_dataset_json, 'r') as f:
        data = json.load(f)

    if args.eval_gpu_time:
        data = data[:1]

    process = psutil.Process(os.getpid())
    start_time = time.time()
    failed_videos = []
    num_videos = len(data)
    target_size = tuple(int(x) for x in args.size.split('*'))

    for vid, entry in enumerate(data):
        video_name = entry["video_name"]
        src_prompt = entry["source_prompt"]
        tgt_prompt = entry["target_prompt"]
        video_path = os.path.join(args.data_dir, video_name + '.mp4')

        type_idx = os.path.basename(args.FiVE_dataset_json).split('_')[0].replace("edit", "")
        save_file = os.path.join(args.save_dir, video_name,
                                 f"{type_idx}_{tgt_prompt[:20]}.mp4")

        if os.path.exists(save_file):
            print(f"[{vid}/{num_videos}] Skip: {save_file}")
            continue

        print(f"[{vid}/{num_videos}] {video_name}: {src_prompt} -> {tgt_prompt}")

        try:
            if video_path.endswith('.mp4'):
                video = load_frames(video_path, num_frames=args.frame_num, target_size=target_size)
            elif os.path.isdir(video_path):
                video = load_frames_path(video_path, num_frames=args.frame_num, target_size=target_size)
            else:
                video = load_frames(video_path, num_frames=args.frame_num, target_size=target_size)

            actual_frames = video.shape[2]
            seed = args.base_seed if args.base_seed >= 0 else torch.randint(0, 2**31, (1,)).item()

            video_out = flowanchor_edit(
                wan_pipeline=wan_t2v,
                video=video,
                src_prompt=src_prompt,
                tgt_prompt=tgt_prompt,
                mask=None,
                target_words=None,
                size=target_size,
                frame_num=actual_frames,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                tgt_guide_scale=args.tgt_guide_scale,
                skip_timesteps=args.skip_timesteps,
                seed=seed,
                offload_model=args.offload_model,
                beta1=args.beta1,
                beta2=args.beta2,
                gamma_scale=args.gamma_scale,
                device=torch.device(f"cuda:{device}"),
            )

            if rank == 0 and video_out is not None:
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                from wan.utils.utils import cache_video
                cache_video(tensor=video_out[None], save_file=save_file,
                            fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
                print(f"  Saved: {save_file}")

        except Exception as e:
            print(f"  Error: {e}")
            failed_videos.append(vid)

    running_time = time.time() - start_time
    peak_gpu = torch.cuda.max_memory_allocated(device="cuda") / (1024**2) if torch.cuda.is_available() else 0

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "memory_stats.txt"), "a") as f:
        f.write(f"FlowAnchor: GPU={peak_gpu:.1f}MB Time={running_time:.1f}s Failed={failed_videos}\n")

    print(f"\nDone. {running_time:.1f}s, GPU={peak_gpu:.1f}MB, Failed={len(failed_videos)}")


if __name__ == "__main__":
    main()
