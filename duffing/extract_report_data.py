# -*- coding: utf-8 -*-
"""提取例会报告所需量化数据(只读, 不重算Duffing)"""
import json, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = r'C:\Users\29094\Documents\ChatGPT\QY pulsar\duffing_oscilator_0902\coherent_duffing_singlepulse'
import os
os.chdir(D)

# ---- 1) mark20 (34s, seg0846=巨脉冲) ----
j = json.load(open('out_duffing_mark_20ms.json'))
print('== mark20 34s ==')
print('f_eff=%.6f Hz  phi_main_abs=%.4f  phi_ip_abs=%.4f' % (
    j['f_eff'], j['phi_main_abs'], j['phi_ip_abs']))
print('snr_thresh=%.1f  seg_ms=%.1f  t_start=%.1f  n_seg=%d  total_hits=%d' % (
    j['snr_thresh'], j['seg_ms'], j['t_start'], j['n_seg'], j['total_hits']))
print('cfg:', j['cfg'])
for i, s in enumerate(j['segments']):
    if s['n_hits'] > 0:
        print('命中窗口 idx=%d t0=%.3f events=%s' % (i, s['seg_t0'], json.dumps(s['events'])))

# ---- 2) t10 mark (seg0002) ----
t = json.load(open('out_t10_mark_20ms.json'))
print('\n== t10 mark ==')
print('f_eff=%.6f phi_main_fold=%.4f phi_ip_fold=%.4f delta=%.4f p_ms=%.6f p_source=%s seg_ms=%s t_start=%s' % (
    t['f_eff'], t['phi_main_fold'], t['phi_ip_fold'], t.get('delta', -1),
    t.get('p_ms', -1), t.get('p_source'), t.get('seg_ms'), t.get('t_start')))
seg2 = t['segments'][2]
print('seg[2]:', json.dumps({k: seg2[k] for k in seg2 if k not in ('events',)}, ensure_ascii=False))
# 各窗锚定统计
snrs = [s.get('anchor', {}).get('snr', -1) if s.get('anchor') else -1 for s in t['segments']]
snrs = [x for x in snrs if x > 0]
print('100窗锚定SNR: min=%.2f med=%.2f max=%.2f' % (min(snrs), float(np.median(snrs)), max(snrs)))
print('n_grid seg2 =', seg2.get('n_grid'))

# ---- 3) 巨脉冲折叠图数据 ----
g = np.load('out_fold2d_giant.npz')
rel, img = g['rel'], g['img']
f_eff = float(g['f_eff']); nb = rel.shape[1]
xs = np.arange(nb) / nb
ph_exp = (0.28177778818462684) % 1.0  # mark20 巨脉冲锚定相位
pk_ph = xs[np.argmax(rel, axis=1)]
tol = 0.011  # ~17μs
nclose = int(np.sum(np.minimum(np.abs(pk_ph - ph_exp), 1 - np.abs(pk_ph - ph_exp)) < tol))
print('\n== fold2d_giant ==')
print('exp 主相位=%.4f; %d/32 行峰相位偏差<%.3f' % (ph_exp, nclose, tol))
print('行峰相位: min=%.4f max=%.4f (偏差 %.4f..%.4f)' % (
    pk_ph.min(), pk_ph.max(),
    np.minimum(np.abs(pk_ph - ph_exp), 1 - np.abs(pk_ph - ph_exp)).min(),
    np.minimum(np.abs(pk_ph - ph_exp), 1 - np.abs(pk_ph - ph_exp)).max()))
# 剖面结构: 主相位附近 z
prof = img.sum(axis=0) / img.shape[0]
pm = np.median(prof); ps = 1.4826 * np.median(np.abs(prof - pm))
pz = (prof - pm) / ps
k0 = int(np.argmin(np.abs(xs - ph_exp)))
print('剖面 @主相位: z=%.1f; ±0.03内 z>2 的bin数=%d; ±0.03内局部峰数=%d' % (
    pz[k0], int(np.sum(pz[(xs >= ph_exp - 0.03) & (xs <= ph_exp + 0.03)] > 2)),
    int(np.sum(np.diff(np.sign(np.diff(pz[(xs >= ph_exp - 0.03) & (xs <= ph_exp + 0.03)]))) < 0))))
print('主相位bin宽(剖面 z>峰z*0.5 的连续bin): ', end='')
w = np.where(pz > 0.5 * pz[k0])[0]
print('%d bin = %.3f phase' % (w[-1] - w[0] + 1 if w.size else 1, (w[-1] - w[0] + 1) / nb if w.size else 1 / nb))
# 2D: 主相位附近(±0.03)最大行z与覆盖行数
mw = (xs >= ph_exp - 0.03) & (xs <= ph_exp + 0.03)
colmax = rel[:, mw].max(axis=1)
print('2D 主窗±0.03: 各行最大z 中位=%.1f max=%.1f; 32行中 z>5 行数=%d' % (
    np.median(colmax), colmax.max(), int(np.sum(colmax > 5))))
print('2D 全图最大z=%.1f @ row=%d phase=%.4f' % (
    rel.max(), int(np.unravel_index(np.argmax(rel), rel.shape)[0]),
    xs[np.unravel_index(np.argmax(rel), rel.shape)[1]]))

# ---- 4) seg0846 / seg1334 输入窗口强度 (out_coh_full.npy) ----
meta = json.load(open('out_coh_full.json'))
print('\n== out_coh_full meta ==', meta)
a = np.load('out_coh_full.npy', mmap_mode='r')
print('dtype=%s shape=%s' % (a.dtype, a.shape))
fs = 2e6
for idx, name in ((846, 'seg0846(巨脉冲)'), (1334, 'seg1334(强输入)')):
    i0 = int(idx * 0.02 * fs); i1 = i0 + int(0.02 * fs)
    seg = np.asarray(a[i0:i1], dtype=np.float64)
    seg_n = (seg - seg.mean()) / (seg.std() + 1e-30)
    k = int(np.argmax(seg_n))
    print('%s t=%.3f-%.3fs: 峰z=%.1f @%+.2fms  谷z=%.1f  |z|>3样本数=%d' % (
        name, idx * 0.02, idx * 0.02 + 0.02, seg_n[k], k / fs * 1e3,
        seg_n.min(), int(np.sum(np.abs(seg_n) > 3))))
# 巨脉冲精确时刻在2MHz流内的z及宽
i0 = int(16.9302305 * fs)
seg = np.asarray(a[i0 - 2000:i0 + 2000], dtype=np.float64)
seg_n = (seg - seg.mean()) / (seg.std() + 1e-30)
k = int(np.argmax(seg_n))
print('巨脉冲 @16.9302305±1ms: 峰z=%.1f (相对样本起点 %+.2f μs)' % (seg_n[k], (k - 2000) / 2e6 * 1e6))
