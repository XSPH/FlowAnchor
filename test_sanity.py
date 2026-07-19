"""FlowAnchor 数学验证:逐条对照论文 (arXiv 2604.22586) 附录 B 的公式.

参考实现全部用朴素循环直接翻译论文公式 (Eqs. 15-18, 24-27),
与 flowanchor.py 的向量化实现对拍; SAR 注意力路径与 wan 官方
CrossAttention 模块对拍.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'FiVE-Bench', 'models', 'wan-edit'))

import torch

from flowanchor import FlowAnchorEditor, SARController

torch.manual_seed(0)
passed, total = 0, 0


def check(name, cond):
    global passed, total
    total += 1
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}')
    passed += bool(cond)


# ---------------------------------------------------------------- #
# 参考实现: 论文公式的朴素循环翻译                                    #
# ---------------------------------------------------------------- #
def sar_reference(logits, mask_flat, j_tar, text_len, beta1, beta2):
    """Eqs. 15-18, 按 (b, head, i, j) 逐元素循环."""
    out = logits.clone().double()
    b_sz, n_heads, l_v, _ = logits.shape
    for b in range(b_sz):
        for h in range(n_heads):
            a = logits[b, h, :, :text_len].double().clone()
            # Step 1 (Eq. 15): per-video-token max/min over text tokens (Eq. 16)
            a1 = a.clone()
            for i in range(l_v):
                a_max, a_min = a[i].max(), a[i].min()
                for j in range(text_len):
                    if mask_flat[b, i] > 0.5:
                        if j in j_tar:
                            a1[i, j] = a[i, j] + beta1 * (a_max - a[i, j])
                        else:
                            a1[i, j] = a[i, j] - beta1 * (a[i, j] - a_min)
            # Step 2 (Eq. 17): per-target-token max/min over video tokens (Eq. 18)
            a2 = a1.clone()
            for j in j_tar:
                aj_max, aj_min = a1[:, j].max(), a1[:, j].min()
                for i in range(l_v):
                    if mask_flat[b, i] > 0.5:
                        a2[i, j] = a1[i, j] + beta2 * (aj_max - a1[i, j])
                    else:
                        a2[i, j] = a1[i, j] - beta2 * (a1[i, j] - aj_min)
            out[b, h, :, :text_len] = a2
    return out.to(logits.dtype)


def amm_reference(delta_v, gamma, f0, eps=1e-7):
    """Eqs. 24-27, 按 batch 逐样本循环."""
    b_sz, c, f, h, w = delta_v.shape
    out = torch.empty_like(delta_v)
    gamma_f = 0.0 if f <= 1 else gamma * math.log(f) / math.log(f0)
    for b in range(b_sz):
        v_bar = delta_v[b].mean(dim=0)                       # Eq. 24: 1/C sum_c
        v_min, v_max = v_bar.min(), v_bar.max()
        c_map = (v_bar - v_min) / (v_max - v_min + eps)      # Eq. 25
        out[b] = (1.0 + gamma_f * c_map)[None] * delta_v[b]  # Eq. 27 (广播到通道)
    return out


# ---------------------------------------------------------------- #
print('=== 1. SAR: 向量化实现 vs 论文公式参考实现 ===')
editor = FlowAnchorEditor(torch.device('cpu'), beta1=0.3, beta2=0.3, gamma=1.0)
for (b_sz, n_h, l_v, l_t, text_len) in [(1, 2, 12, 8, 6), (2, 3, 20, 10, 10), (1, 1, 6, 5, 5)]:
    logits = torch.randn(b_sz, n_h, l_v, l_t)
    mask = (torch.rand(b_sz, l_v) > 0.5).float()
    j_tar = [1, 3]
    got = editor.sar_modulate_logits(logits, mask, j_tar, text_len)
    ref = sar_reference(logits, mask, j_tar, text_len, 0.3, 0.3)
    check(f'Eqs.15-18 一致 (shape {b_sz}x{n_h}x{l_v}x{l_t}, L={text_len})',
          torch.allclose(got, ref, atol=1e-5))

logits = torch.randn(1, 2, 12, 8)
mask = (torch.rand(1, 12) > 0.5).float()
ed0 = FlowAnchorEditor(torch.device('cpu'), beta1=0.0, beta2=0.0)
check('beta=0 时为恒等变换', torch.allclose(ed0.sar_modulate_logits(logits, mask, [1], 6), logits))

got = editor.sar_modulate_logits(logits, mask, [1, 3], 6)
check('padding 文本位置 (j>=L) 不被修改', torch.equal(got[..., 6:], logits[..., 6:]))
non_tgt = [j for j in range(6) if j not in (1, 3)]
out_mask = mask[0] < 0.5
check('mask 外的非目标 token 不被修改',
      torch.allclose(got[0, :, out_mask][..., non_tgt], logits[0, :, out_mask][..., non_tgt]))
a_slice, g_slice = logits[..., :6], got[..., :6]
check('数值稳定性 (Eqs.21-22): 调制后不超出原 logit 范围',
      (g_slice.amax() <= a_slice.amax() + 1e-5) and (g_slice.amin() >= a_slice.amin() - 1e-5))

print('\n=== 2. SAR 注意力路径 vs wan 官方 CrossAttention ===')
try:
    from wan.modules.model import WanT2VCrossAttention
    # GPU 上模块跑在 bf16 autocast 下; CPU 无 autocast, 直接用 bf16 模块对拍
    attn = WanT2VCrossAttention(dim=64, num_heads=4).eval().to(torch.bfloat16)
    x = torch.randn(1, 24, 64).to(torch.bfloat16)
    ctx = torch.randn(1, 12, 64).to(torch.bfloat16)
    with torch.no_grad():
        ref_out = attn(x, ctx, None)
        ctrl = SARController(
            FlowAnchorEditor(torch.device('cpu'), beta1=0.0, beta2=0.0),
            mask_flat=torch.ones(1, 24), target_indices=[2],
            text_len=12, num_video_tokens=24)

        class Blk:
            cross_attn = attn
        class Mdl:
            blocks = [Blk()]

        ctrl.patch(Mdl())
        pass_out = attn(x, ctx, None)          # active=False -> 原始路径
        ctrl.active = True
        sar_out = attn(x, ctx, None)           # beta=0 -> 数学上等于原注意力
        ctrl.unpatch()
        restored = attn(x, ctx, None)
    check('active=False 时走原始 forward', torch.equal(pass_out, ref_out))
    # wan 的 SDPA fallback 用 bf16, SAR 路径 fp32 softmax, 允许 bf16 量级误差
    check('beta=0 的 SAR 路径 ≈ 官方注意力 (bf16 容差)',
          torch.allclose(sar_out, ref_out, atol=3e-2, rtol=3e-2))
    check('unpatch 恢复原始 forward', torch.equal(restored, ref_out))
except Exception as e:
    check(f'wan CrossAttention 对拍 (异常: {e})', False)

print('\n=== 3. AMM: 向量化实现 vs 论文公式参考实现 ===')
editor = FlowAnchorEditor(torch.device('cpu'), gamma=1.0, f0=21)
for (c, f, h, w) in [(16, 11, 8, 8), (4, 21, 6, 6), (16, 3, 4, 4)]:
    dv = torch.randn(2, c, f, h, w)
    got = editor.adaptive_magnitude_modulation(dv)
    ref = amm_reference(dv, 1.0, 21)
    check(f'Eqs.24-27 一致 (C={c}, F={f})', torch.allclose(got, ref, atol=1e-5))

dv = torch.randn(16, 11, 8, 8)
check('无 batch 维输入 [C,F,H,W]',
      torch.allclose(editor.adaptive_magnitude_modulation(dv),
                     amm_reference(dv.unsqueeze(0), 1.0, 21).squeeze(0), atol=1e-5))
dv1 = torch.randn(1, 16, 1, 8, 8)
check('F=1 时 gamma_F=0, AMM 为恒等 (单图不放大)',
      torch.equal(editor.adaptive_magnitude_modulation(dv1), dv1))
check('gamma_F(21) == gamma', abs(editor.gamma_f(21) - 1.0) < 1e-9)
check('gamma_F(11) == log11/log21',
      abs(editor.gamma_f(11) - math.log(11) / math.log(21)) < 1e-9)
dv = torch.randn(1, 8, 11, 4, 4)
ratio = editor.adaptive_magnitude_modulation(dv) / dv
gf = editor.gamma_f(11)
check('放大系数在 [1, 1+gamma_F] 内 (Eq.29)',
      (ratio.min() >= 1.0 - 1e-4) and (ratio.max() <= 1.0 + gf + 1e-4))
for val in [0.0, 1e-10, 1e10]:
    dv = torch.full((1, 4, 11, 4, 4), val)
    r = editor.adaptive_magnitude_modulation(dv)
    check(f'极端值 {val} 无 NaN/Inf', torch.isfinite(r).all().item())

print('\n=== 4. Mask 下采样到 latent token 网格 ===')
mask = torch.zeros(1, 1, 41, 64, 64)
mask[:, :, :, :32, :32] = 1.0                     # 左上角 1/4
flat = FlowAnchorEditor.downsample_mask_to_latent(mask, 11, 4, 4)
check('形状 [B, F*H*W]', flat.shape == (1, 11 * 4 * 4))
grid = flat.reshape(11, 4, 4)
check('空间位置正确 (左上 2x2 为 True)',
      grid[:, :2, :2].all().item() and not grid[:, 2:, :].any().item()
      and not grid[:, :, 2:].any().item())
mask = torch.zeros(1, 1, 41, 16, 16)
mask[:, :, 0] = 1.0                               # 仅像素帧 0
flat = FlowAnchorEditor.downsample_mask_to_latent(mask, 11, 2, 2)
grid = flat.reshape(11, 2, 2)
check('时间因果分块: 帧0 -> latent 0', grid[0].all().item() and not grid[1:].any().item())
mask = torch.zeros(1, 1, 41, 16, 16)
mask[:, :, 5] = 1.0                               # 像素帧 5 -> latent 帧 2 (帧 5..8)
grid = FlowAnchorEditor.downsample_mask_to_latent(mask, 11, 2, 2).reshape(11, 2, 2)
check('时间因果分块: 帧5 -> latent 2', grid[2].all().item() and not grid[[0, 1, 3]].any().item())
check('全 1 mask -> 全 True', FlowAnchorEditor.downsample_mask_to_latent(
    torch.ones(1, 1, 41, 16, 16), 11, 2, 2).all().item())

print('\n=== 5. 目标词推导与 token 定位 (无 tokenizer fallback) ===')
words = FlowAnchorEditor.derive_target_words(
    'A woman in a pink sweater walks forward', 'A woman in a lemon sweater walks forward')
check(f'替换词 diff: {words}', words == ['lemon'])
words = FlowAnchorEditor.derive_target_words(
    'A man is performing parkour', 'A Spiderman is performing parkour')
check(f'替换词 diff: {words}', words == ['Spiderman'])
idx = FlowAnchorEditor.find_target_token_indices('a red car', ['red'])
check('whitespace fallback 定位', idx == [1])
check('未命中时返回空 (不再回退到全部 token)',
      FlowAnchorEditor.find_target_token_indices('hello world', ['xyz']) == [])

print('\n=== 6. 采样网格 (T=25, shift=5, 论文 Sec. B) ===')
try:
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1,
                                      use_dynamic_shifting=False)
    sch.set_timesteps(25, shift=5.0)
    sig = sch.sigmas
    check('sigma 网格长度 T+1 且终点为 0', len(sig) == 26 and sig[-1].item() == 0)
    check('sigma[0] ≈ 1 (纯噪声端, wan sigma_max=0.999)', sig[0].item() > 0.99)
    check('sigma 单调递减', bool((sig[:-1] > sig[1:]).all()))
    n_sar = int((sig[:-1] >= 0.6).sum())
    n_edit = 25 - 2
    print(f'  [INFO] 跳过前 2 步后编辑步数 n_max = {n_edit} (论文: 23)')
    print(f'  [INFO] tau=0.6 时 SAR 激活步数 = {n_sar}/25 (跳过后实际 {n_sar - 2} 步)')
except Exception as e:
    check(f'sigma 网格 (异常: {e})', False)

print('\n=== 7. 语法检查 ===')
import py_compile
for fname in ['flowanchor.py', 'edit_flowanchor.py', 'eval_five.py', 'test_sanity.py']:
    try:
        py_compile.compile(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname),
                           doraise=True)
        check(fname, True)
    except py_compile.PyCompileError:
        check(fname, False)

print(f'\n{"=" * 40}')
print(f'结果: {passed}/{total} 通过')
if passed != total:
    print('存在失败项!')
    sys.exit(1)
print('全部验证通过!')
