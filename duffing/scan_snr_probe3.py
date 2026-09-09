# -*- coding: utf-8 -*-
"""诊断3: (a)全程扫描每个20ms窗口最强候选的SNR(z域) (b)折叠验证脉冲真实幅度"""
import numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
fs = 2.0e6
s = np.load('out_coh_full.npy', mmap_mode='r')
N = s.size
nwin = int(N // int(0.02*fs))
print('windows:', nwin)

def ratio_at(z, p, W):
    half = W//2; a = p-half
    if a-W < 0 or a+W+W > len(z): return -1.0
    return np.abs(z[a:a+W]).sum() / (np.abs(z[a-W:a]).sum()+1e-30)

# (a) 全程扫描: 每窗口记录最大z峰 及 W=10/20 的ratio
best = {}
win_ratios = {10:[], 20:[], 40:[]}
maxz_win = []
for j in range(0, nwin, 10):  # 每10个窗采样1个 = 200ms 步长, 快速全程
    i0 = j*int(0.02*fs)
    seg = np.asarray(s[i0:i0+int(0.02*fs)], dtype=np.float64)
    z = (seg-seg.mean())/(seg.std()+1e-30)
    az = np.abs(z)
    pks = [i for i in range(1,len(z)-1) if az[i]>=az[i-1] and az[i]>az[i+1] and z[i]>2.0]
    if not pks: continue
    maxz_win.append((z[pks[np.argmax(z[pks])]], j))
    for W in win_ratios:
        r = max(ratio_at(z,p,W) for p in pks)
        win_ratios[W].append((r, j))
for W in win_ratios:
    arr = sorted(win_ratios[W], reverse=True)[:8]
    print('W=%d: top ratios: %s' % (W, ' '.join('%.2f@win%d(t=%.2fs)' % (r, j, j*0.02) for r,j in arr)))
mx = sorted(maxz_win, reverse=True)[:8]
print('max z per window top:', ' '.join('z=%.1f@t=%.2fs' % (z, j*0.02) for z,j in mx))
