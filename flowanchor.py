"""
FlowAnchor: Stabilizing the Editing Signal for Inversion-Free Video Editing
Core implementation of SAR (Spatial-aware Attention Refinement) and AMM (Adaptive Magnitude Modulation)
"""
import torch
from typing import Optional, List


class FlowAnchorEditor:
    """
    FlowAnchor editing framework that stabilizes the editing signal
    via SAR and AMM on top of Wan-Edit (FlowEdit for videos).
    """

    def __init__(
        self,
        device: torch.device,
        beta1: float = 0.5,
        beta2: float = 0.5,
        gamma_scale: float = 1.0,
    ):
        self.device = device
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma_scale = gamma_scale

    @staticmethod
    def find_target_token_indices(prompt: str, target_words: List[str]) -> List[int]:
        """Find token indices for target words in the prompt."""
        tokens = prompt.lower().split()
        if not tokens:
            return []
        indices = []
        for word in target_words:
            word_lower = word.lower()
            for i, t in enumerate(tokens):
                if word_lower in t and i not in indices:
                    indices.append(i)
        return indices if indices else list(range(len(tokens)))

    def spatial_aware_attention_refinement(
        self,
        ca_map: torch.Tensor,
        mask: torch.Tensor,
        target_indices: List[int],
    ) -> torch.Tensor:
        """
        SAR: Spatial-aware Attention Refinement

        Modulates cross-attention maps to enforce alignment between target text
        tokens and spatial regions specified by the mask.

        Args:
            ca_map: Cross-attention map [B, L_video, L_text]
            mask: Binary mask [B, 1, F, H, W] or [B, L_video] indicating edit region
            target_indices: Indices of target text tokens driving the edit

        Returns:
            Refined attention map [B, L_video, L_text]
        """
        B, L_v, L_t = ca_map.shape

        mask = mask.to(ca_map.device)
        if mask.dim() == 5:
            mask_flat = mask.reshape(B, -1)
        elif mask.dim() == 2:
            mask_flat = mask
        else:
            mask_flat = mask.reshape(B, -1)

        if mask_flat.shape[1] < L_v:
            pad = torch.zeros(B, L_v - mask_flat.shape[1],
                              device=ca_map.device, dtype=mask_flat.dtype)
            mask_flat = torch.cat([mask_flat, pad], dim=1)
        else:
            mask_flat = mask_flat[:, :L_v]

        mask_bool = mask_flat > 0.5
        target_idx = torch.tensor(target_indices, device=ca_map.device, dtype=torch.long)

        A = ca_map.clone()
        A_max = A.max(dim=-1, keepdim=True).values
        A_min = A.min(dim=-1, keepdim=True).values

        mask_expanded = mask_bool.unsqueeze(-1)

        is_target = torch.zeros(B, L_v, L_t, device=ca_map.device, dtype=torch.bool)
        idx = target_idx.unsqueeze(0).unsqueeze(0).expand(B, L_v, -1)
        is_target.scatter_(2, idx, True)

        step1 = A.clone()
        boost = self.beta1 * (A_max - A)
        suppress = self.beta1 * (A - A_min)

        target_in_mask = mask_expanded & is_target
        non_target_in_mask = mask_expanded & ~is_target

        step1 = torch.where(target_in_mask, A + boost, step1)
        step1 = torch.where(non_target_in_mask, A - suppress, step1)

        A_prime_max = step1.max(dim=1, keepdim=True).values
        A_prime_min = step1.min(dim=1, keepdim=True).values

        step2 = step1.clone()
        in_mask_target = mask_expanded & is_target
        out_mask_target = ~mask_expanded & is_target

        boost2 = self.beta2 * (A_prime_max - step1)
        suppress2 = self.beta2 * (step1 - A_prime_min)

        step2 = torch.where(in_mask_target, step1 + boost2, step2)
        step2 = torch.where(out_mask_target, step1 - suppress2, step2)

        return step2

    def adaptive_magnitude_modulation(
        self,
        delta_v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_frames: int = 1,
    ) -> torch.Tensor:
        """
        AMM: Adaptive Magnitude Modulation

        Amplifies the editing signal using a normalized contrast map derived
        from the signal itself, with frame-aware scaling to compensate for
        length-induced magnitude attenuation.

        Args:
            delta_v: Editing signal [B, C, F, H, W]
            mask: Optional binary mask [B, 1, F, H, W]
            num_frames: Number of video frames for frame-aware scaling

        Returns:
            Modulated editing signal [B, C, F, H, W]
        """
        if mask is not None:
            masked_signal = delta_v * mask
        else:
            masked_signal = delta_v

        abs_signal = torch.abs(masked_signal)
        spatial_mean = abs_signal.mean(dim=(-3, -2, -1), keepdim=True)
        spatial_std = abs_signal.std(dim=(-3, -2, -1), keepdim=True) + 1e-8
        contrast_map = (abs_signal - spatial_mean) / spatial_std
        contrast_map = torch.clamp(contrast_map, min=0)
        contrast_sum = contrast_map.sum(dim=(-3, -2, -1), keepdim=True) + 1e-8
        contrast_map = contrast_map / contrast_sum
        if mask is not None:
            contrast_map = contrast_map * mask

        frame_scale = self.gamma_scale * (num_frames / 81.0) ** 0.5

        amplified = delta_v + frame_scale * (contrast_map * delta_v)

        return amplified
