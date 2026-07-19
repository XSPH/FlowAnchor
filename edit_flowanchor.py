"""
FlowAnchor: Full editing pipeline on top of Wan-Edit (arXiv 2604.22586, Alg. 1)

Follows the paper's implementation details (supplementary Sec. B):
  * T = 25 inference steps, first 2 skipped (n_max = 23), n_avg = 1;
  * explicit Euler trajectory update Z_edit += (t_{i-1} - t_i) * dV (Eq. 13),
    on the shift-5 rectified-flow sigma grid;
  * source/target latents built with the CURRENT t_i (Alg. 1 line 5);
  * SAR modulates CA logits of all layers, only inside the target-conditional
    velocity forward, while t_i >= tau (tau = 0.6);
  * AMM applied to the editing signal at every step (mask-free).
"""
import argparse
import gc
import logging
import math
import os
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Tuple

warnings.filterwarnings('ignore')

import cv2
import numpy as np
import torch
import torch.cuda.amp as amp

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.utils.utils import cache_video, str2bool
from flowanchor import FlowAnchorEditor, SARController


def load_frames(video_path=None, num_frames=41, target_size=(832, 480)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, target_size)
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        tensor_frame = torch.tensor(resized_frame).permute(2, 0, 1).float() / 255.0
        tensor_frame = 2 * tensor_frame - 1
        frames.append(tensor_frame)
    cap.release()
    if not frames:
        raise ValueError("Video does not have enough frames")
    return torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0)


def load_frames_path(video_path=None, num_frames=41, target_size=(832, 480)):
    frame_files = sorted([f for f in os.listdir(video_path)
                          if f.endswith(('.jpg', '.png'))])
    frames = []
    for i in range(min(num_frames, len(frame_files))):
        frame = cv2.imread(os.path.join(video_path, frame_files[i]))
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frame = 2 * frame.astype(np.float32) / 255.0 - 1
        frames.append(np.transpose(frame, (2, 0, 1)))
    return torch.tensor(np.array(frames)).float().permute(1, 0, 2, 3).unsqueeze(0)


def load_mask(mask_path: str, num_frames: int, target_size: Tuple[int, int]) -> Optional[torch.Tensor]:
    """Load a pixel-space binary mask as [1, 1, F, H, W] (values 0/1)."""
    if mask_path is None:
        return None

    if os.path.isdir(mask_path):
        mask_files = sorted([f for f in os.listdir(mask_path)
                             if f.endswith(('.png', '.jpg'))])
        masks = []
        for i in range(min(num_frames, len(mask_files))):
            m = cv2.imread(os.path.join(mask_path, mask_files[i]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = cv2.resize(m, target_size)
            masks.append((m > 127).astype(np.float32))
        if masks:
            while len(masks) < num_frames:  # repeat last frame if mask is short
                masks.append(masks[-1])
            return torch.tensor(np.array(masks)).float()[None, None]

    elif mask_path.endswith('.mp4'):
        cap = cv2.VideoCapture(mask_path)
        masks = []
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            masks.append((gray > 127).astype(np.float32))
        cap.release()
        if masks:
            while len(masks) < num_frames:
                masks.append(masks[-1])
            return torch.tensor(np.array(masks)).float()[None, None]

    elif mask_path.endswith(('.png', '.jpg')):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            m = cv2.resize(m, target_size)
            m = (m > 127).astype(np.float32)
            mask_tensor = torch.tensor(m).float()[None, None, None]
            return mask_tensor.expand(1, 1, num_frames, -1, -1).contiguous()

    return None


def flowanchor_edit(
    wan_pipeline,
    video: torch.Tensor,
    src_prompt: str,
    tgt_prompt: str,
    mask: Optional[torch.Tensor],
    target_words: Optional[List[str]],
    size: Tuple[int, int],
    frame_num: int,
    shift: float = 5.0,
    sample_solver: str = 'euler',
    sampling_steps: int = 25,
    guide_scale: float = 5.0,
    tgt_guide_scale: float = 10.0,
    skip_timesteps: int = 2,
    seed: int = -1,
    offload_model: bool = True,
    beta1: float = 0.3,
    beta2: float = 0.3,
    gamma: float = 1.0,
    f0: int = 21,
    sar_tau: float = 0.6,
    no_sar: bool = False,
    device: torch.device = torch.device('cuda'),
) -> torch.Tensor:
    F = frame_num
    video = video.to(device)
    latents = wan_pipeline.vae.encode(video)
    start_latents = latents if isinstance(latents, list) else [latents]
    x_src = start_latents[0].float()                    # [C, F', H/8, W/8]

    lat_c, lat_f, lat_h, lat_w = x_src.shape
    patch_f, patch_h, patch_w = wan_pipeline.patch_size
    tok_f = lat_f // patch_f
    tok_h = lat_h // patch_h
    tok_w = lat_w // patch_w
    num_video_tokens = tok_f * tok_h * tok_w
    seq_len = math.ceil(
        (lat_h * lat_w) / (patch_h * patch_w) * lat_f / wan_pipeline.sp_size
    ) * wan_pipeline.sp_size

    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    if not wan_pipeline.t5_cpu:
        wan_pipeline.text_encoder.model.to(device)
        context_src = wan_pipeline.text_encoder([src_prompt], device)
        context_tgt = wan_pipeline.text_encoder([tgt_prompt], device)
        context_null = wan_pipeline.text_encoder([wan_pipeline.sample_neg_prompt], device)
        if offload_model:
            wan_pipeline.text_encoder.model.cpu()
    else:
        context_src = [t.to(device) for t in wan_pipeline.text_encoder([src_prompt], torch.device('cpu'))]
        context_tgt = [t.to(device) for t in wan_pipeline.text_encoder([tgt_prompt], torch.device('cpu'))]
        context_null = [t.to(device) for t in wan_pipeline.text_encoder([wan_pipeline.sample_neg_prompt], torch.device('cpu'))]

    editor = FlowAnchorEditor(device=device, beta1=beta1, beta2=beta2,
                              gamma=gamma, f0=f0, sar_tau=sar_tau)

    # ---------------- SAR setup ---------------- #
    sar_controller = None
    if not no_sar and mask is not None:
        if not target_words:
            target_words = editor.derive_target_words(src_prompt, tgt_prompt)
            logging.info(f"SAR target words (auto-derived): {target_words}")
        tokenizer = wan_pipeline.text_encoder.tokenizer
        target_token_indices = editor.find_target_token_indices(
            tgt_prompt, target_words, tokenizer)
        if target_token_indices:
            logging.info(f"SAR target token indices (umT5): {target_token_indices}")
            mask_flat = editor.downsample_mask_to_latent(
                mask.to(device), tok_f, tok_h, tok_w)
            sar_controller = SARController(
                editor=editor,
                mask_flat=mask_flat,
                target_indices=target_token_indices,
                text_len=context_tgt[0].shape[0],
                num_video_tokens=num_video_tokens,
            )
        else:
            logging.warning(
                "SAR disabled: could not map target words to token indices.")
    elif not no_sar:
        logging.warning("SAR disabled: no mask provided.")

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(wan_pipeline.model, 'no_sync', noop_no_sync)

    with amp.autocast(dtype=wan_pipeline.param_dtype), torch.no_grad(), no_sync():
        # Rectified-flow sigma grid (shift-5, T steps + terminal 0), shared by
        # both solvers.  Following FlowEdit / Alg. 1, the trajectory update
        # itself is explicit Euler; 'unipc' is kept only as the legacy
        # Wan-Edit-style update for comparison.
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=wan_pipeline.num_train_timesteps,
            shift=1, use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
        timesteps = sample_scheduler.timesteps
        sigmas = sample_scheduler.sigmas          # length T+1, ends at 0

        arg_src = {'context': context_src, 'seq_len': seq_len}
        arg_tgt = {'context': context_tgt, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        wan_pipeline.model.to(device)
        if sar_controller is not None:
            sar_controller.patch(wan_pipeline.model)

        zt_edit = x_src.clone()                              # Alg. 1 line 1
        try:
            for i, t in enumerate(timesteps):
                if i < skip_timesteps:                       # n_max = T - skip
                    continue

                t_i = float(sigmas[i])
                t_im1 = float(sigmas[i + 1])
                timestep = torch.stack([t])

                noise = torch.randn(
                    x_src.shape, generator=seed_g,
                    dtype=torch.float32, device=device)
                zt_src = (1.0 - t_i) * x_src + t_i * noise   # Alg. 1 line 5
                zt_tar = zt_edit + zt_src - x_src            # Alg. 1 line 6

                # V_tar: SAR active only here, and only while t_i >= tau
                if sar_controller is not None:
                    sar_controller.active = t_i >= sar_tau
                v_cond_tgt = wan_pipeline.model([zt_tar], t=timestep, **arg_tgt)[0]
                if sar_controller is not None:
                    sar_controller.active = False

                v_uncond_tgt = wan_pipeline.model([zt_tar], t=timestep, **arg_null)[0]
                v_cond_src = wan_pipeline.model([zt_src], t=timestep, **arg_src)[0]
                v_uncond_src = wan_pipeline.model([zt_src], t=timestep, **arg_null)[0]

                v_src = v_uncond_src + guide_scale * (v_cond_src - v_uncond_src)
                v_tar = v_uncond_tgt + tgt_guide_scale * (v_cond_tgt - v_uncond_tgt)
                delta_v = (v_tar - v_src).float()            # Alg. 1 line 9

                delta_v = editor.adaptive_magnitude_modulation(delta_v)  # line 10

                if sample_solver == 'euler':
                    zt_edit = zt_edit + (t_im1 - t_i) * delta_v          # Eq. 13
                elif sample_solver == 'unipc':
                    zt_edit = sample_scheduler.step(
                        delta_v.unsqueeze(0), t, zt_edit.unsqueeze(0),
                        return_dict=False, generator=seed_g)[0].squeeze(0)
                else:
                    raise NotImplementedError(
                        f"Unsupported solver: {sample_solver}")
        finally:
            if sar_controller is not None:
                sar_controller.unpatch()

        x0 = [zt_edit]
        if offload_model:
            wan_pipeline.model.cpu()
        if wan_pipeline.rank == 0:
            videos = wan_pipeline.vae.decode(x0)

    del latents
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()

    return videos[0] if wan_pipeline.rank == 0 else None


def _parse_args():
    parser = argparse.ArgumentParser(description="FlowAnchor: Stable Inversion-Free Video Editing")
    parser.add_argument("--task", type=str, default="t2v-1.3B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--frame_num", type=int, default=41)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--offload_model", type=str2bool, default=True)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--video_dir", type=str, default="data")
    parser.add_argument("--video_name", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--tgt_prompt", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--target_words", type=str, nargs='+', default=None,
                        help="Words in the target prompt driving the edit (J_tar); "
                             "auto-derived from the prompt diff if omitted.")
    # Sampling (paper Sec. B: T=25, skip first 2 steps, Euler update, shift 5)
    parser.add_argument("--sample_solver", type=str, default='euler',
                        choices=['euler', 'unipc'],
                        help="'euler' = paper Alg. 1 / FlowEdit update (default); "
                             "'unipc' = legacy Wan-Edit-style scheduler step.")
    parser.add_argument("--sample_steps", type=int, default=25)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--tgt_guide_scale", type=float, default=10.0)
    parser.add_argument("--skip_timesteps", type=int, default=2,
                        help="Skipped leading steps = T - n_max (paper: 25 - 23 = 2).")
    parser.add_argument("--base_seed", type=int, default=-1)
    # FlowAnchor hyperparameters (paper: beta1=beta2=0.3, gamma=1.0, tau=0.6T, F0=21)
    parser.add_argument("--beta1", type=float, default=0.3)
    parser.add_argument("--beta2", type=float, default=0.3)
    parser.add_argument("--gamma", "--gamma_scale", dest="gamma", type=float, default=1.0,
                        help="AMM base amplification strength (0 disables AMM).")
    parser.add_argument("--f0", type=int, default=21,
                        help="Reference latent temporal length F0 (Eq. 28).")
    parser.add_argument("--sar_tau", type=float, default=0.6,
                        help="SAR active while normalized time t >= tau.")
    parser.add_argument("--no_sar", action="store_true", default=False)
    return parser.parse_args()


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def main():
    args = _parse_args()
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.video_path is not None:
        video_path = args.video_path
    elif args.video_name is not None:
        video_path = os.path.join(args.video_dir, args.video_name)
    else:
        raise ValueError("Must specify --video_path or --video_name")

    target_size = tuple(int(x) for x in args.size.split('*'))
    if os.path.isdir(video_path):
        video = load_frames_path(video_path, num_frames=args.frame_num, target_size=target_size)
    else:
        video = load_frames(video_path, num_frames=args.frame_num, target_size=target_size)

    actual_frames = video.shape[2]
    logging.info(f"Loaded video with {actual_frames} frames")

    mask = None
    if args.mask_path:
        mask = load_mask(args.mask_path, actual_frames, target_size)
        if mask is not None:
            logging.info(f"Loaded mask: {tuple(mask.shape)}")
        else:
            logging.warning(f"Failed to load mask from {args.mask_path}")

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

    seed = args.base_seed if args.base_seed >= 0 else torch.randint(0, 2**31, (1,)).item()
    logging.info(f"Seed: {seed}")

    video = flowanchor_edit(
        wan_pipeline=wan_t2v,
        video=video,
        src_prompt=args.prompt,
        tgt_prompt=args.tgt_prompt,
        mask=mask,
        target_words=args.target_words,
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
        gamma=args.gamma,
        f0=args.f0,
        sar_tau=args.sar_tau,
        no_sar=args.no_sar,
        device=torch.device(f"cuda:{device}"),
    )

    if rank == 0:
        if args.save_file is None:
            os.makedirs(args.save_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fp = args.tgt_prompt.replace(" ", "_").replace("/", "_")[:50]
            args.save_file = os.path.join(args.save_dir, f"flowanchor_{args.size}_{fp}_{ts}.mp4")
        os.makedirs(os.path.dirname(args.save_file) or '.', exist_ok=True)
        logging.info(f"Saving to {args.save_file}")
        cache_video(tensor=video[None], save_file=args.save_file,
                    fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))

    logging.info("Done.")


if __name__ == "__main__":
    main()
