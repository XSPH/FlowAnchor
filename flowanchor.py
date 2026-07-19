"""
FlowAnchor: Stabilizing the Editing Signal for Inversion-Free Video Editing
(arXiv 2604.22586)

Core implementation of SAR (Spatial-aware Attention Refinement, paper Sec. 3.3 /
Eqs. 14-18) and AMM (Adaptive Magnitude Modulation, paper Sec. 3.4 / Eqs. 23-28),
following the supplementary implementation details (Sec. B) exactly:

  * SAR modulates the cross-attention LOGITS (before softmax) of all CA layers,
    during the early denoising stage t >= tau (tau = 0.6 in normalized time),
    with beta1 = beta2 = 0.3.
  * AMM is applied at every step: the contrast map is the channel-mean of the
    editing signal, min-max normalized per sample over the flattened F x H x W
    latent positions (eps = 1e-7), scaled by gamma_F = gamma * log F / log F0
    with F0 = 21 (Wan2.1's native latent temporal length, Eq. 28) and F the
    latent temporal length of the current video.  AMM does not use the mask.
"""
import difflib
import logging
import math
import re
import string
from typing import List, Optional

import torch
import torch.nn.functional as nn_f


class FlowAnchorEditor:
    """Holds FlowAnchor hyperparameters and implements SAR / AMM math."""

    def __init__(
        self,
        device: torch.device,
        beta1: float = 0.3,          # SAR text-token modulation strength (Eq. 15)
        beta2: float = 0.3,          # SAR spatio-temporal modulation strength (Eq. 17)
        gamma: float = 1.0,          # AMM base amplification strength (Eq. 26)
        f0: int = 21,                # reference latent temporal length (Eq. 28)
        sar_tau: float = 0.6,        # SAR active while normalized time t >= tau
        eps: float = 1e-7,           # Eq. 25
    ):
        self.device = device
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.f0 = f0
        self.sar_tau = sar_tau
        self.eps = eps

    # ------------------------------------------------------------------ #
    # AMM                                                                #
    # ------------------------------------------------------------------ #
    def gamma_f(self, latent_frames: int) -> float:
        """Frame-adaptive amplification factor gamma_F = gamma * log F / log F0 (Eq. 26)."""
        if latent_frames <= 1:
            return 0.0
        return self.gamma * math.log(latent_frames) / math.log(self.f0)

    def adaptive_magnitude_modulation(self, delta_v: torch.Tensor) -> torch.Tensor:
        """
        AMM (Eqs. 23-27): amplify the editing signal with a min-max normalized
        contrast map derived from its own channel mean.

        Args:
            delta_v: Editing signal, [C, F, H, W] or [B, C, F, H, W] (latent space).

        Returns:
            Modulated editing signal, same shape and dtype as input.
        """
        squeeze = delta_v.dim() == 4
        dv = delta_v.unsqueeze(0) if squeeze else delta_v

        latent_frames = dv.shape[2]
        gamma_f = self.gamma_f(latent_frames)
        if gamma_f == 0.0:
            return delta_v

        v_bar = dv.float().mean(dim=1, keepdim=True)                    # Eq. 24
        v_min = v_bar.amin(dim=(2, 3, 4), keepdim=True)
        v_max = v_bar.amax(dim=(2, 3, 4), keepdim=True)
        c_map = (v_bar - v_min) / (v_max - v_min + self.eps)            # Eq. 25
        out = ((1.0 + gamma_f * c_map) * dv.float()).to(dv.dtype)       # Eq. 27
        return out.squeeze(0) if squeeze else out

    # ------------------------------------------------------------------ #
    # SAR                                                                #
    # ------------------------------------------------------------------ #
    def sar_modulate_logits(
        self,
        logits: torch.Tensor,
        mask_flat: torch.Tensor,
        target_indices: List[int],
        text_len: int,
        num_video_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        SAR on cross-attention logits (Eqs. 15-18), applied per attention head.

        Args:
            logits: CA logits [B, num_heads, L_video, L_text_padded].
            mask_flat: Binary latent-space mask [B, L_video] (bool or 0/1),
                flattened in the model's (F, H, W) row-major token order.
            target_indices: Text-token indices J_tar of the target words.
            text_len: True (unpadded) text length L; only the first `text_len`
                token logits are modulated and used for max/min statistics.
            num_video_tokens: True (unpadded) video token count N_l; video-token
                max/min statistics (Eq. 18) are restricted to this range.

        Returns:
            Modulated logits, same shape as input.
        """
        b, n, l_v, l_t_pad = logits.shape
        n_vid = min(num_video_tokens or l_v, l_v)
        l_t = min(text_len, l_t_pad)

        mask_b = (mask_flat > 0.5).to(logits.device)
        if mask_b.shape[1] != l_v:
            raise ValueError(
                f"mask length {mask_b.shape[1]} != video token count {l_v}")

        is_tgt = torch.zeros(l_t, dtype=torch.bool, device=logits.device)
        idx = [i for i in target_indices if 0 <= i < l_t]
        if not idx:
            return logits
        is_tgt[idx] = True

        a = logits[..., :l_t].float()                           # [B, n, L_v, L]
        m = mask_b[:, None, :, None]                            # [B, 1, L_v, 1]
        j_t = is_tgt[None, None, None, :]                       # [1, 1, 1, L]

        # Step 1: text-token modulation (Eqs. 15-16); max/min per video token
        # over the true text tokens (the text axis is already sliced to L).
        a_max = a.max(dim=-1, keepdim=True).values
        a_min = a.min(dim=-1, keepdim=True).values
        a1 = torch.where(m & j_t, a + self.beta1 * (a_max - a), a)
        a1 = torch.where(m & ~j_t, a - self.beta1 * (a - a_min), a1)

        # Step 2: spatio-temporal modulation for target tokens (Eqs. 17-18)
        a1_max = a1[:, :, :n_vid].max(dim=2, keepdim=True).values  # per text token
        a1_min = a1[:, :, :n_vid].min(dim=2, keepdim=True).values
        a2 = torch.where(m & j_t, a1 + self.beta2 * (a1_max - a1), a1)
        a2 = torch.where(~m & j_t, a1 - self.beta2 * (a1 - a1_min), a2)

        out = logits.clone()
        out[..., :l_t] = a2.to(logits.dtype)
        return out

    # ------------------------------------------------------------------ #
    # Mask handling                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def downsample_mask_to_latent(
        mask: torch.Tensor,
        lat_f: int,
        lat_h: int,
        lat_w: int,
    ) -> torch.Tensor:
        """
        Downsample a pixel-space binary mask to the latent token grid used by
        the DiT (any pixel coverage -> token in mask, i.e. max pooling).

        Temporal mapping follows the causal Wan VAE: latent frame 0 <- pixel
        frame 0; latent frame j (j >= 1) <- pixel frames 4j-3 .. 4j.

        Args:
            mask: [B, 1, F, H, W] pixel-space mask (0/1).
            lat_f, lat_h, lat_w: latent token grid (F', H/16, W/16 for the
                default VAE stride 8 x patch size 2).

        Returns:
            Bool tensor [B, lat_f * lat_h * lat_w], (F, H, W) row-major —
            matching the DiT patchify token order.
        """
        b, _, f, h, w = mask.shape
        m = mask.float()

        chunks = [m[:, :, 0:1].amax(dim=2)]
        for j in range(1, lat_f):
            s, e = 4 * (j - 1) + 1, min(4 * j + 1, f)
            if s >= f:
                chunks.append(chunks[-1])
            else:
                chunks.append(m[:, :, s:e].amax(dim=2))
        mt = torch.stack(chunks, dim=2)                          # [B, 1, F', H, W]

        ms = nn_f.adaptive_max_pool2d(
            mt.reshape(b * lat_f, 1, h, w), (lat_h, lat_w))
        ms = ms.reshape(b, lat_f, lat_h, lat_w)
        return (ms > 0.5).reshape(b, -1)

    # ------------------------------------------------------------------ #
    # Target-word / token utilities                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def derive_target_words(src_prompt: str, tgt_prompt: str) -> List[str]:
        """Words appearing in the target prompt but not aligned with the source
        prompt (word-level diff) — the default J_tar source when no explicit
        --target_words are given."""
        def norm(words):
            return [w.strip(string.punctuation).lower() for w in words]

        src_w, tgt_w = src_prompt.split(), tgt_prompt.split()
        sm = difflib.SequenceMatcher(None, norm(src_w), norm(tgt_w))
        out = []
        for tag, _, _, j1, j2 in sm.get_opcodes():
            if tag in ('replace', 'insert'):
                out.extend(w.strip(string.punctuation) for w in tgt_w[j1:j2])
        return [w for w in out if w]

    @staticmethod
    def find_target_token_indices(
        prompt: str,
        target_words: List[str],
        tokenizer=None,
    ) -> List[int]:
        """
        Map target words to token indices J_tar in the text encoder's token
        sequence for `prompt`.

        `tokenizer` is wan's HuggingfaceTokenizer wrapper (pipeline
        `text_encoder.tokenizer`); when given, indices refer to actual umT5
        token positions (matching the CA text axis).  Without it, falls back
        to whitespace word positions (only meaningful for tests).
        """
        if not target_words:
            return []

        if tokenizer is None:
            tokens = prompt.lower().split()
            wanted = {w.lower().strip(string.punctuation) for w in target_words}
            return [i for i, t in enumerate(tokens)
                    if t.strip(string.punctuation) in wanted]

        text = tokenizer._clean(prompt) if getattr(tokenizer, 'clean', None) else prompt
        hf = tokenizer.tokenizer

        def word_spans(word):
            w = re.escape(word.strip())
            spans = [mt.span() for mt in
                     re.finditer(r'(?<!\w)' + w + r'(?!\w)', text, re.IGNORECASE)]
            if not spans:  # relax to substring match
                spans = [mt.span() for mt in re.finditer(w, text, re.IGNORECASE)]
            return spans

        if getattr(hf, 'is_fast', False):
            enc = hf(text, return_offsets_mapping=True, add_special_tokens=True,
                     truncation=True, max_length=getattr(tokenizer, 'seq_len', None))
            offsets = enc['offset_mapping']
            found = set()
            for word in target_words:
                for s, e in word_spans(word):
                    for ti, (ts, te) in enumerate(offsets):
                        if te > ts and ts < e and te > s:
                            found.add(ti)
            return sorted(found)

        # Slow tokenizer: match token-id subsequences.
        pids = hf(text, add_special_tokens=False)['input_ids']
        found = set()
        for word in target_words:
            for variant in (word, ' ' + word):
                wids = hf(variant, add_special_tokens=False)['input_ids']
                if not wids:
                    continue
                for s in range(len(pids) - len(wids) + 1):
                    if pids[s:s + len(wids)] == wids:
                        found.update(range(s, s + len(wids)))
        return sorted(found)


class SARController:
    """
    Patches the WanModel cross-attention layers so that, while `active` is
    True, attention is computed with SAR-modulated logits (softmax after
    modulation, Sec. B.1).  Patch once per edit; toggle `active` so that SAR
    only affects the target-conditional velocity V_tar (Algorithm 1).
    """

    def __init__(
        self,
        editor: FlowAnchorEditor,
        mask_flat: torch.Tensor,           # [B, L_video] bool
        target_indices: List[int],
        text_len: int,
        num_video_tokens: int,
    ):
        self.editor = editor
        self.mask_flat = mask_flat
        self.target_indices = target_indices
        self.text_len = text_len
        self.num_video_tokens = num_video_tokens
        self.active = False
        self._patched = []

    def patch(self, model):
        for block in model.blocks:
            attn = getattr(block, 'cross_attn', None)
            if attn is None:
                continue
            orig = attn.forward
            attn.forward = self._make_forward(attn, orig)
            self._patched.append((attn, orig))

    def unpatch(self):
        for attn, orig in self._patched:
            attn.forward = orig
        self._patched.clear()

    def _make_forward(self, attn, orig_forward):
        def forward(x, context, context_lens):
            if not self.active:
                return orig_forward(x, context, context_lens)
            return self._sar_attention(attn, x, context)
        return forward

    def _sar_attention(self, attn, x, context):
        b, n, d = x.size(0), attn.num_heads, attn.head_dim
        q = attn.norm_q(attn.q(x)).view(b, -1, n, d)
        k = attn.norm_k(attn.k(context)).view(b, -1, n, d)
        v = attn.v(context).view(b, -1, n, d)

        logits = torch.einsum('bind,bjnd->bnij', q, k) / math.sqrt(d)
        logits = self.editor.sar_modulate_logits(
            logits, self.mask_flat, self.target_indices,
            self.text_len, self.num_video_tokens)
        weights = logits.float().softmax(dim=-1).to(v.dtype)
        out = torch.einsum('bnij,bjnd->bind', weights, v)

        return attn.o(out.flatten(2))
