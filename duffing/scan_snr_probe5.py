# -*- coding: utf-8 -*-
"""诊断5: 全程强事件相位聚类 — 判断强尖峰是真脉冲(聚类于主/间相位)还是噪声(均匀)"""
import numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
fs = 2.0e6
f_eff = 641.886226994
s = np.load('out_coh_full.npy', mmap_mode='r')
N = s.size
# 分块 z-score (整段做mean/std即可, 逐窗更稳)
chunk = int(0.1*fs)  # 100ms块
phases = {6.0: [], 7.0: [], 8.0: []}
evts = []
nch = 0
for i0 in range(0, N, chunk):
    seg = np.asarray(s[i0:i0+chunk], dtype=np.float64)
    z = (seg-seg.mean())/(seg.std()+1e-30)
    for thr in phases:
        idx = np.where(z > thr)[0]
        # 只取局部极大
        keep = []
        for i in idx:
            if 0 < i < len(z)-1 and z[i] >= z[i-1] and z[i] >= z[i+1]:
                keep.append(i)
        for i in keep:
            t = (i0+i)/fs
            ph = (t*f_eff) % 1.0
            phases[thr].append(ph)
            if thr == 7.0:
                evts.append((t, z[i], ph))
    nch += 1
print('chunks:', nch)
nb = 50
for thr in phases:
    arr = np.array(phases[thr])
    h, _ = np.histogram(arr, bins=nb, range=(0,1))
    print('\nthr=%.1f  n=%5d' % (thr, len(arr)))
    print('hist:', ' '.join('%d' % x for x in h))
    # 最高两簇
    order = np.argsort(h)[::-1][:4]
    print('top bins:', [(o, o/nb, h[o]) for o in sorted(order)])
# 打印最强事件的相位
evts.sort(key=lambda e: -e[1])
print('\ntop 25 z>7 events (t, z, phase):')
for t, z, ph in evts[:25]:
    print('t=%8.4fs z=%5.2f phase=%.3f' % (t, z, ph))
