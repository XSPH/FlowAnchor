"""FlowAnchor 最终验证脚本"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from flowanchor import FlowAnchorEditor

print('=== FlowAnchor 完整验证 ===\n')

editor = FlowAnchorEditor(torch.device('cpu'), beta1=0.5, beta2=0.5, gamma_scale=1.0)
passed = 0
total = 0

def check(name, cond):
    global passed, total
    total += 1
    if cond:
        passed += 1
        print(f'  [PASS] {name}')
    else:
        print(f'  [FAIL] {name}')

# --- SAR ---
print('1. SAR 空间感知注意力精炼')
for B,Lv,Lt in [(1,64,10),(1,256,50),(2,128,20)]:
    ca = torch.randn(B,Lv,Lt).softmax(dim=-1)
    mask = torch.ones(B,1,4,4,4); mask[:,:,2:,:,:] = 0
    r = editor.spatial_aware_attention_refinement(ca, mask, [0,1])
    check(f'形状 {ca.shape}', r.shape == ca.shape)
ca = torch.randn(1,64,10).softmax(dim=-1)
mask = torch.ones(1,1,4,4,4)
r1 = editor.spatial_aware_attention_refinement(ca, mask, [2,3])
r2 = editor.spatial_aware_attention_refinement(ca, mask, [2,3])
check('确定性', torch.allclose(r1, r2))

# --- AMM ---
print('\n2. AMM 自适应幅度调制')
for val in [0.0, 1e-10, 1e10]:
    dv = torch.full((1,16,4,8,8), val)
    r = editor.adaptive_magnitude_modulation(dv, torch.ones(1,1,4,8,8), 4)
    check(f'极端值 {val}', not torch.isnan(r).any() and not torch.isinf(r).any())
dv = torch.zeros(1,16,4,8,8)
r = editor.adaptive_magnitude_modulation(dv, torch.ones(1,1,4,8,8), 4)
check('零信号', (r == 0).all())
for nf in [1,8,41,81]:
    dv = torch.randn(1,16,4,8,8)
    r = editor.adaptive_magnitude_modulation(dv, torch.ones(1,1,4,8,8), nf)
    check(f'帧数 {nf}', r.shape == dv.shape and not torch.isnan(r).any())

# --- Token ---
print('\n3. Token 查找')
check('basic', editor.find_target_token_indices('a red car', ['red']) == [1])
check('case insensitive', editor.find_target_token_indices('A RED', ['red']) == [1])
check('fallback', len(editor.find_target_token_indices('hello', ['xyz'])) > 0)
check('empty', editor.find_target_token_indices('', ['test']) == [])

# --- 组合 ---
print('\n4. 组合流程')
ca = torch.randn(1,64,10).softmax(dim=-1)
mask_5d = torch.ones(1,1,4,4,4); mask_5d[:,:,:2,:,:] = 0
r_sar = editor.spatial_aware_attention_refinement(ca, mask_5d, [2,3])
dv = torch.randn(1,16,4,8,8)
mask_4d = torch.ones(1,1,4,8,8); mask_4d[:,:,:2,:,:] = 0
r_amm = editor.adaptive_magnitude_modulation(dv, mask_4d, 4)
check('SAR+AMM', r_sar.shape == ca.shape and r_amm.shape == dv.shape)

# --- GPU ---
print('\n5. GPU')
if torch.cuda.is_available():
    eg = FlowAnchorEditor(torch.device('cuda'), 0.5, 0.5, 1.0)
    ca_g = torch.randn(1,64,10).softmax(dim=-1).cuda()
    mask_g = torch.ones(1,1,4,4,4).cuda()
    r_g = eg.spatial_aware_attention_refinement(ca_g, mask_g, [2,3])
    check('GPU SAR', r_g.device.type == 'cuda')
    dv_g = torch.randn(1,16,4,8,8).cuda()
    r_a = eg.adaptive_magnitude_modulation(dv_g, torch.ones(1,1,4,8,8).cuda(), 4)
    check('GPU AMM', r_a.device.type == 'cuda')
else:
    check('GPU 不存在 (跳过)', True)

# --- 语法 ---
print('\n6. 语法检查')
import py_compile
for f in ['flowanchor.py', 'edit_flowanchor.py', 'eval_five.py', 'test_sanity.py']:
    try:
        py_compile.compile(f, doraise=True)
        check(f, True)
    except py_compile.PyCompileError:
        check(f, False)

print(f'\n{"="*40}')
print(f'结果: {passed}/{total} 通过')
if passed == total:
    print('全部验证通过!')
else:
    print('存在失败项!')
    sys.exit(1)
