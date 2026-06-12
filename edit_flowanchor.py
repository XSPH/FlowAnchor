"""
FlowAnchor: Full editing pipeline on top of Wan-Edit
Integrates SAR and AMM into the Wan T2V editing workflow
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
from flowanchor import FlowAnchorEditor


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
            return torch.tensor(np.array(masks)).float().permute(1, 0, 2, 3).unsqueeze(0)

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
            return torch.tensor(np.array(masks)).float().permute(1, 0, 2, 3).unsqueeze(0)

    elif mask_path.endswith(('.png', '.jpg')):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            m = cv2.resize(m, target_size)
            m = (m > 127).astype(np.float32)
            mask_tensor = torch.tensor(m).float().unsqueeze(0).unsqueeze(0)
            return mask_tensor.expand(1, 1, num_frames, -1, -1)

    return None


class SARCrossAttentionHook:
    """Hook that applies SAR refinement to cross-attention during forward pass."""

    def __init__(self, editor: FlowAnchorEditor, mask: torch.Tensor,
                 target_indices: List[int]):
        self.editor = editor
        self.mask = mask
        self.target_indices = target_indices
        self.handles = []

    def register(self, model):
        for block in model.blocks:
            if hasattr(block, 'cross_attn'):
                h = block.cross_attn.register_forward_hook(self._hook)
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _hook(self, module, inp, output):
        if not (hasattr(module, 'q') and hasattr(module, 'k')):
            return output
        q_raw = module.norm_q(module.q(inp[0]))
        k_raw = module.norm_k(module.k(inp[1]))
        B, L_v, n, d = q_raw.shape
        L_t = k_raw.shape[1]
        q = q_raw.reshape(B, L_v, n, d)
        k = k_raw.reshape(B, L_t, n, d)

        ca_weights = torch.einsum('bind,bjnd->bijn', q, k) / (d ** 0.5)
        ca_weights = torch.softmax(ca_weights, dim=-1)

        refined_ca = self.editor.spatial_aware_attention_refinement(
            ca_weights, self.mask, self.target_indices)

        diff = refined_ca - ca_weights
        modulated_k = k + torch.einsum('bijn,bjnd->bjnd', diff, k)

        from wan.modules.attention import flash_attention
        v = module.v(inp[1])
        x = flash_attention(q, modulated_k.reshape(B, L_t, n, d),
                            v.reshape(B, L_t, n, d), k_lens=None)
        x = x.flatten(2)
        x = module.o(x)
        return x


def flowanchor_edit(
    wan_pipeline,
    video: torch.Tensor,
    src_prompt: str,
    tgt_prompt: str,
    mask: Optional[torch.Tensor],
    target_words: Optional[List[str]],
    size: Tuple[int, int],
    frame_num: int,
    shift: float,
    sample_solver: str,
    sampling_steps: int,
    guide_scale: float,
    tgt_guide_scale: float,
    skip_timesteps: int,
    seed: int,
    offload_model: bool,
    beta1: float,
    beta2: float,
    gamma_scale: float,
    device: torch.device,
) -> torch.Tensor:
    F = frame_num
    video = video.to(device)
    latents = wan_pipeline.vae.encode(video)
    target_shape = (
        wan_pipeline.vae.model.z_dim,
        (F - 1) // wan_pipeline.vae_stride[0] + 1,
        size[1] // wan_pipeline.vae_stride[1],
        size[0] // wan_pipeline.vae_stride[2],
    )
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3])
        / (wan_pipeline.patch_size[1] * wan_pipeline.patch_size[2])
        * target_shape[1] / wan_pipeline.sp_size
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

    editor = FlowAnchorEditor(device=device, beta1=beta1, beta2=beta2, gamma_scale=gamma_scale)

    target_token_indices = None
    if target_words:
        target_token_indices = editor.find_target_token_indices(tgt_prompt, target_words)
        logging.info(f"SAR target token indices: {target_token_indices}")

    mask_on_device = mask.to(device) if mask is not None else None

    use_sar = (target_token_indices is not None and mask_on_device is not None)

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(wan_pipeline.model, 'no_sync', noop_no_sync)

    with amp.autocast(dtype=wan_pipeline.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == 'unipc':
            from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=wan_pipeline.num_train_timesteps,
                shift=1, use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                              get_sampling_sigmas, retrieve_timesteps)
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=wan_pipeline.num_train_timesteps,
                shift=1, use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        arg_src = {'context': context_src, 'seq_len': seq_len}
        arg_tgt = {'context': context_tgt, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        start_latents = latents
        mv_latent = latents.clone()

        for i, t in enumerate(timesteps):
            noise = [torch.randn(target_shape[0], target_shape[1],
                                 target_shape[2], target_shape[3],
                                 dtype=torch.float32, device=device)]

            if i < skip_timesteps:
                continue

            t_prev = 1000 if i == 0 else timesteps[i - 1]
            src_latent = [t_prev / 1000.0 * noise[0] + (1000 - t_prev) / 1000.0 * start_latents[0]]
            tgt_latent = [mv_latent[0] + src_latent[0] - start_latents[0]]

            timestep = torch.stack([t])
            wan_pipeline.model.to(device)

            sar_hook = None
            if use_sar:
                sar_hook = SARCrossAttentionHook(editor, mask_on_device, target_token_indices)
                sar_hook.register(wan_pipeline.model)

            noise_pred_cond_src = wan_pipeline.model(src_latent, t=timestep, **arg_src)[0]
            noise_pred_cond_tgt = wan_pipeline.model(tgt_latent, t=timestep, **arg_tgt)[0]

            if sar_hook is not None:
                sar_hook.remove()

            noise_pred_uncond_src = wan_pipeline.model(src_latent, t=timestep, **arg_null)[0]
            noise_pred_uncond_tgt = wan_pipeline.model(tgt_latent, t=timestep, **arg_null)[0]

            noise_pred_src = noise_pred_uncond_src + guide_scale * (
                noise_pred_cond_src - noise_pred_uncond_src)
            noise_pred_tgt = noise_pred_uncond_tgt + tgt_guide_scale * (
                noise_pred_cond_tgt - noise_pred_uncond_tgt)

            delta_v = noise_pred_tgt - noise_pred_src

            if mask_on_device is not None:
                delta_v = editor.adaptive_magnitude_modulation(
                    delta_v, mask_on_device, target_shape[1])

            temp_x0 = sample_scheduler.step(
                delta_v.unsqueeze(0), t,
                mv_latent[0].unsqueeze(0),
                return_dict=False, generator=seed_g)[0]
            mv_latent = [temp_x0.squeeze(0)]

        x0 = mv_latent
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
    parser.add_argument("--target_words", type=str, nargs='+', default=None)
    parser.add_argument("--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'])
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--tgt_guide_scale", type=float, default=10.0)
    parser.add_argument("--skip_timesteps", type=int, default=16)
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.5)
    parser.add_argument("--gamma_scale", type=float, default=1.0)
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
    if args.video_path is not None and args.video_path.endswith('.mp4'):
        video = load_frames(args.video_path, num_frames=args.frame_num, target_size=target_size)
    elif os.path.isdir(video_path):
        video = load_frames_path(video_path, num_frames=args.frame_num, target_size=target_size)
    else:
        video = load_frames(video_path, num_frames=args.frame_num, target_size=target_size)

    actual_frames = video.shape[2]
    logging.info(f"Loaded video with {actual_frames} frames")

    mask = None
    if args.mask_path:
        mask = load_mask(args.mask_path, actual_frames, target_size)
        if mask is not None:
            logging.info(f"Loaded mask: {mask.shape}")

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
        gamma_scale=args.gamma_scale,
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
