# -*- coding: utf-8 -*-
"""扫描探测2: 在原始功率序列(不z-score)上测 SNR = Σp(on窗口)/Σp(前沿等宽基线)"""
import numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SER = 'out_coh_full.npy'
meta = json.load(open('out_coh_full.json', encoding='utf-8'))
fs = float(meta['f_out'])
s = np.load(SER, mmap_mode='r')
print('series', SER, s.size, 'fs=%.1fMHz' % (fs/1e6))

# 每窗口统计原始功率分布
def win_stats(t0):
    i0 = int(t0*fs); seg = np.asarray(s[i0:i0+int(0.02*fs)], dtype=np.float64)
    return seg

def snr_raw(seg, p, W):
    half = W//2
    a = p - half
    if a - W < 0 or a + W + W > len(seg):
        return np.nan
    on = seg[a:a+W].sum()
    base = seg[a-W:a].sum()
    return on/(base+1e-30)

for t0 in (0.0, 10.0, 20.0, 30.0):
    seg = win_stats(t0)
    med = np.median(seg); mean = seg.mean()
    mx = seg.max()
    print('\n[t=%.0fs] raw power: mean=%.4g median=%.4g max=%.4g max/med=%.1f' % (t0, mean, med, mx, mx/med))
    for W in (10, 20, 40, 80):
        cand = [i for i in range(1, len(seg)-1) if seg[i] >= seg[i-1] and seg[i] > seg[i+1] and seg[i] > 4*med]
        vals = [(snr_raw(seg, p, W), p, seg[p]) for p in cand]
        vals = [v for v in vals if np.isfinite(v[0])]
        vals.sort(reverse=True)
        over5 = [v for v in vals if v[0] > 5]
        print('  W=%3d: cand=%4d SNR>5: %3d | top: %s' % (W, len(cand), len(over5),
              ' '.join('%.2f@%d(p%.4g)' % (v[0], v[1], v[2]) for v in vals[:6])))
