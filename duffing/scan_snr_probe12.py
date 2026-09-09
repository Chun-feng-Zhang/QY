# -*- coding: utf-8 -*-
"""诊断12: t10_2s 数据 SNR分布/阈值可行性/间距测量"""
import numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
fs = 2.0e6
s = np.load('out_coh_t10_2s.npy', mmap_mode='r')
N = int(0.02*fs)
f_eff = 641.886226994
P_par = 1.0/f_eff
print('series', s.size, 'windows:', s.size//N, 'P_par=%.6f ms' % (P_par*1e3))

def pulse_snr(seg, p, m, sigma, gap_tol=2):
    A = float(seg[p])
    if A <= m: return None
    thr = m + max(0.05*(A-m), 5.0*sigma)
    lo = p
    while lo-1 >= 0 and seg[lo-1] >= thr: lo -= 1
    hi = p
    while hi+1 < len(seg) and seg[hi+1] >= thr: hi += 1
    W = hi-lo+1
    if lo-W < 0: return None
    on = float(seg[lo:hi+1].sum()); base = float(seg[lo-W:lo].sum())
    snr = on/(base+1e-30)
    if not (np.isfinite(snr) and snr > 0): return None
    return dict(snr=snr, W=W, lo=lo, hi=hi, A=A, m=m, pk_over_m=A/m)

all_ev = []
for j in range(s.size//N):
    i0 = j*N
    seg = np.asarray(s[i0:i0+N], dtype=np.float64)
    m = float(np.median(seg)); sigma = float(1.4826*np.median(np.abs(seg-m)))
    mid = seg[1:-1]
    cand = np.where((mid >= seg[:-2]) & (mid > seg[2:]) & (mid > 5*m))[0] + 1
    for p in cand.tolist():
        r = pulse_snr(seg, p, m, sigma)
        if r and r['snr'] > 5:
            t = (i0+p)/fs + 10.0
            all_ev.append((j, p, t, r['snr'], r))
print('SNR>5 events in t10_2s:', len(all_ev))
ev20 = [e for e in all_ev if e[3] > 20]
print('SNR>20 events:', len(ev20))
for e in ev20:
    print('  win%d t=%.6f snr=%.1f W=%d pk/m=%.1f' % (e[0], e[2], e[3], e[4]['W'], e[4]['pk_over_m']))
# 相邻SNR>20间距
if len(ev20) >= 2:
    ev20.sort(key=lambda e: e[2])
    for a, b in zip(ev20, ev20[1:]):
        dt = b[2]-a[2]
        print('  gap between SNR>20: %.6f ms (%.3f x P_par)' % (dt*1e3, dt/P_par))
# 各窗口最大SNR
maxsnr_by_win = {}
for e in all_ev:
    if e[3] > maxsnr_by_win.get(e[0], 0): maxsnr_by_win[e[0]] = e[3]
import collections
print('windows with SNR>5 max: %d of %d' % (len(maxsnr_by_win), s.size//N))
top = sorted(maxsnr_by_win.items(), key=lambda kv: -kv[1])[:10]
print('top window SNRs:', [(w, round(v,1)) for w, v in top])
