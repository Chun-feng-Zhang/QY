# -*- coding: utf-8 -*-
"""诊断9: 原始功率域 自适应on-region SNR 最终标定"""
import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
fs = 2.0e6
f_eff = 641.886226994
s = np.load('out_coh_full.npy', mmap_mode='r')
N = int(0.02*fs)  # 40000/窗

def region_snr(seg, p, f):
    A = seg[p]
    m = np.median(seg)
    if A <= m: return None
    thr = m + f*(A-m)
    lo = p
    while lo-1 >= 0 and seg[lo-1] >= thr: lo -= 1
    hi = p
    while hi+1 < len(seg) and seg[hi+1] >= thr: hi += 1
    W = hi-lo+1
    a = lo-W
    if a < 0: return None
    on = seg[lo:hi+1].sum()
    base = seg[a:lo].sum()
    return (on/(base+1e-30), W, lo, hi, A/m)

# 1) 巨脉冲
i0g = int(16.93*fs)
seg = np.asarray(s[i0g:i0g+N], dtype=np.float64)
pg = int(np.argmax(seg))
print('=== giant window t=16.93: peak raw=%.4g (%.1f x med) ===' % (seg[pg], seg[pg]/np.median(seg)))
for f in (0.1, 0.15, 0.2, 0.3):
    r = region_snr(seg, pg, f)
    print(' f=%.2f: snr=%.1f W=%d peak/med=%.1f' % (f, r[0], r[1], r[4]) if r else ' f=%.2f: None' % f)

# 2) 全程统计 (f=0.15)
evs = []
for j in range(int(s.size // N)):
    i0 = j*N
    seg = np.asarray(s[i0:i0+N], dtype=np.float64)
    m = np.median(seg)
    # 局部极大 且 > 3m
    cand = [i for i in range(1, N-1) if seg[i] >= seg[i-1] and seg[i] > seg[i+1] and seg[i] > 3*m]
    for p in cand:
        r = region_snr(seg, p, 0.15)
        if r and r[0] > 5:
            t = (i0+p)/fs
            evs.append((t, (t*f_eff) % 1.0, r[0], r[1], r[4]))
print('\nwindows:', s.size//N, ' SNR>5 events:', len(evs))
evs.sort(key=lambda e: -e[2])
print('top 40 (t, phase, snr, W, pk/m):')
for e in evs[:40]:
    print('t=%9.5fs ph=%.3f snr=%6.1f W=%3d pk/m=%.1f' % e)
# 相位直方图
h = np.zeros(20)
for e in evs: h[int(e[1]*20) % 20] += 1
print('phase hist (0.05 bins):', ' '.join('%d' % x for x in h))
