# -*- coding: utf-8 -*-
"""诊断7: 标定自适应SNR参数f; 验证巨脉冲SNR>5, 普通脉冲<5, 并检查误检率"""
import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
fs = 2.0e6
f_eff = 641.886226994
s = np.load('out_coh_full.npy', mmap_mode='r')

def zwin(i0, n=int(0.02*fs)):
    seg = np.asarray(s[i0:i0+n], dtype=np.float64)
    return (seg-seg.mean())/(seg.std()+1e-30)

def adapt_snr(z, p, f):
    A = z[p]
    b = np.median(np.abs(z))
    thr = b + f*(A-b)
    if thr <= b: return None
    lo = p
    while lo-1 >= 0 and z[lo-1] >= thr: lo -= 1
    hi = p
    while hi+1 < len(z) and z[hi+1] >= thr: hi += 1
    W = hi-lo+1
    a = lo-W
    if a < 0: return None
    return (np.abs(z[lo:hi+1]).sum()/(np.abs(z[a:lo]).sum()+1e-30), W)

# 1) 巨脉冲窗口 (t=16.93)
i0g = int(16.93*fs)
zg = zwin(i0g)
pg = int(np.argmax(zg))
print('=== giant window t=16.93, peak z=%.1f at %d ===' % (zg[pg], pg))
for f in (0.15, 0.25, 0.35, 0.5):
    r = adapt_snr(zg, pg, f)
    print(' f=%.2f: snr=%s' % (f, ('%.1f W=%d' % r) if r else None))

# 2) 全34s: 对每窗局部极大 z>3 的候选算SNR(f=0.25), 统计 >5 事件数+相位聚类
ph_ev = []
tot_win = 0
for j in range(int(s.size // int(0.02*fs))):
    i0 = j*int(0.02*fs)
    z = zwin(i0)
    az = np.abs(z)
    pk = np.argmax(z)
    if z[pk] > 3.0:
        r = adapt_snr(z, pk, 0.25)
        if r and r[0] > 5:
            t = (i0+pk)/fs
            ph_ev.append(((t*f_eff) % 1.0, z[pk], r, j))
    tot_win += 1
print('\ntotal windows:', tot_win, ' windows with maxSNR>5:', len(ph_ev))
for ph, zz, r, j in sorted(ph_ev, key=lambda e: -e[1][1])[:30]:
    print('win%4d t=%7.3fs z=%6.1f snr=%6.1f W=%3d phase=%.3f' % (j, j*0.02, zz, r[0], r[1], ph))
