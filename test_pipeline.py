"""FlowAnchor 管线冒烟测试 (CPU, 无需模型权重).

用 mock 的 VAE / 文本编码器 / DiT 驱动完整的 flowanchor_edit 循环, 验证
Algorithm 1 的接线: 跳步、SAR 门控 (仅目标条件分支、仅 t>=tau)、AMM、
Euler 轨迹更新、每步 4 次模型前向.  mock DiT 内部真实调用 wan 的
CrossAttention 模块, 因此 SAR 的 monkey-patch 路径被完整执行.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'FiVE-Bench', 'models', 'wan-edit'))

import torch
import torch.nn.functional as nn_f

from edit_flowanchor import flowanchor_edit
from wan.modules.model import WanT2VCrossAttention

torch.manual_seed(0)
DIM = 64
passed, total = 0, 0


def check(name, cond):
    global passed, total
    total += 1
    print(f'  [{"PASS" if cond else "FAIL"}] {name}')
    passed += bool(cond)


class MockVAE:
    class _M:
        z_dim = 16
    model = _M()

    def encode(self, videos):
        out = []
        for v in videos:                       # [3, F, H, W]
            c, f, h, w = v.shape
            lat_f = (f - 1) // 4 + 1
            lat = nn_f.adaptive_avg_pool3d(v.unsqueeze(0), (lat_f, h // 8, w // 8))[0]
            out.append(lat.repeat(6, 1, 1, 1)[:16].float())
        return out

    def decode(self, latents):
        return [nn_f.adaptive_avg_pool3d(z.unsqueeze(0), (z.shape[1], 32, 32))[0][:3]
                for z in latents]


class MockTokenizerHF:
    is_fast = False

    def __call__(self, text, add_special_tokens=False, **kw):
        return {'input_ids': [abs(hash(w)) % 30000 for w in text.strip().split()]}


class MockTokenizer:
    tokenizer = MockTokenizerHF()
    clean = None
    seq_len = 512


class MockTextEncoder:
    tokenizer = MockTokenizer()

    def __call__(self, texts, device):
        g = torch.Generator().manual_seed(sum(ord(c) for c in texts[0]) % 9973)
        n = max(len(texts[0].split()), 1)
        return [torch.randn(n, DIM, generator=g)]


class Block:
    def __init__(self, attn):
        self.cross_attn = attn


class MockDiT:
    """v = -0.5 z + (cross-attention 输出的逐 token 均值, 上采样广播).

    输出依赖 CrossAttention 的结果, 因此 SAR 调制会改变速度场."""

    def __init__(self):
        g = torch.Generator().manual_seed(7)
        self.attn = WanT2VCrossAttention(dim=DIM, num_heads=4).eval().to(torch.bfloat16)
        for p in self.attn.parameters():
            p.data = torch.randn(p.shape, generator=g).to(torch.bfloat16) * 0.2
        self.blocks = [Block(self.attn)]
        self.lift = torch.randn(1, DIM, generator=g) * 0.3
        self.calls = 0

    def to(self, *a, **k):
        return self

    def cpu(self):
        return self

    def __call__(self, x_list, t=None, context=None, seq_len=None):
        self.calls += 1
        outs = []
        for z in x_list:                        # [16, F', H8, W8]
            c, f, h8, w8 = z.shape
            th, tw = h8 // 2, w8 // 2
            tok = nn_f.adaptive_avg_pool3d(
                z.mean(0, keepdim=True).unsqueeze(0), (f, th, tw))
            tok = tok.reshape(1, f * th * tw, 1) @ self.lift          # [1, L_v, DIM]
            ctx = context[0].unsqueeze(0)
            with torch.no_grad():
                a = self.blocks[0].cross_attn(
                    tok.to(torch.bfloat16), ctx.to(torch.bfloat16), None)
            a = a.float().mean(-1).reshape(f, th, tw)                 # [F', th, tw]
            a = a.repeat_interleave(2, -2).repeat_interleave(2, -1)   # -> [F', H8, W8]
            outs.append(-0.5 * z + a.unsqueeze(0).expand(c, -1, -1, -1))
        return outs


class MockPipeline:
    vae = MockVAE()
    text_encoder = MockTextEncoder()
    patch_size = (1, 2, 2)
    sp_size = 1
    t5_cpu = True
    sample_neg_prompt = 'bad quality'
    param_dtype = torch.float32
    num_train_timesteps = 1000
    rank = 0

    def __init__(self):
        self.model = MockDiT()


def run(mask=None, **kw):
    pipe = MockPipeline()
    video = torch.zeros(1, 3, 17, 64, 64)
    video[:, 0, :, 16:32, 16:32] = 0.8
    args = dict(
        wan_pipeline=pipe, video=video,
        src_prompt='a red car driving', tgt_prompt='a blue car driving',
        mask=mask, target_words=['blue'], size=(64, 64), frame_num=17,
        sampling_steps=25, skip_timesteps=2, seed=42,
        offload_model=False, device=torch.device('cpu'))
    args.update(kw)
    out = flowanchor_edit(**args)
    return out, pipe.model.calls


mask = torch.zeros(1, 1, 17, 64, 64)
mask[:, :, :, 16:32, 16:32] = 1.0

print('=== 管线冒烟测试 (mock 模型, T=25, skip 2) ===')
out_nosar, calls = run(mask=None)
check('无 SAR 运行成功且输出有限', out_nosar is not None and torch.isfinite(out_nosar).all())
check(f'模型前向次数 = 4 x (25-2) = 92 (实际 {calls})', calls == 4 * 23)

out_sar, _ = run(mask=mask)
check('SAR 运行成功且输出有限', torch.isfinite(out_sar).all())
check('SAR 改变了编辑轨迹', not torch.allclose(out_sar, out_nosar))

out_tau_never, _ = run(mask=mask, sar_tau=2.0)
check('tau=2.0 (SAR 永不激活) == 无 SAR (门控/patch 透传正确)',
      torch.allclose(out_tau_never, out_nosar, atol=1e-6))

out_gamma0, _ = run(mask=None, gamma=0.0)
check('gamma=0 (无 AMM) 与 gamma=1 结果不同', not torch.allclose(out_gamma0, out_nosar))

out_seed, _ = run(mask=None)
check('相同 seed 结果可复现', torch.allclose(out_seed, out_nosar))

out_unipc, _ = run(mask=None, sample_solver='unipc')
check('legacy unipc 求解器可运行且输出有限', torch.isfinite(out_unipc).all())

out_nomask_words, _ = run(mask=mask, target_words=None)
check('target_words 缺省时自动 diff 推导并运行', torch.isfinite(out_nomask_words).all())

print(f'\n结果: {passed}/{total} 通过')
if passed != total:
    sys.exit(1)
print('管线冒烟测试全部通过!')
