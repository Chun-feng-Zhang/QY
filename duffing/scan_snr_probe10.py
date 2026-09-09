# -*- coding: utf-8 -*-
"""诊断10: Duffing输出在预测主/间脉冲位置的SNR(区域和定义), 看普通脉冲增强后能否>5"""
import numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from numba import njit
fs = 2.0e6
f_eff = 641.886226994
P = 1.0/f_eff
s = np.load('out_coh_full.npy', mmap_mode='r')

@njit
def rk4_coupled(signal, dt, F, omega, k, xi, x1, y1, x2, y2):
    n = signal.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        t = i*dt
        st = signal[i]
        h = dt; h2 = h*0.5
        c1 = np.cos(omega*t)
        k1x1 = omega*y1; k1y1 = omega*(-xi*y1 + x1 - x1*x1*x1 - k*xi*(x1-x2) - k*(y1-y2) + F*c1 + st)
        k1x2 = omega*y2; k1y2 = omega*(-xi*y2 + x2 - x2*x2*x2 - k*xi*(x2-x1) - k*(y2-y1) + F*c1)
        t2 = t+h2; c2 = np.cos(omega*t2)
        x1a,y1a = x1+h2*k1x1, y1+h2*k1y1; x2a,y2a = x2+h2*k1x2, y2+h2*k1y2
        k2x1 = omega*y1a; k2y1 = omega*(-xi*y1a + x1a - x1a*x1a*x1a - k*xi*(x1a-x2a) - k*(y1a-y2a) + F*c2 + st)
        k2x2 = omega*y2a; k2y2 = omega*(-xi*y2a + x2a - x2a*x2a*x2a - k*xi*(x2a-x1a) - k*(y2a-y1a) + F*c2)
        x1b,y1b = x1+h2*k2x1, y1+h2*k2y1; x2b,y2b = x2+h2*k2x2, y2+h2*k2y2
        k3x1 = omega*y1b; k3y1 = omega*(-xi*y1b + x1b - x1b*x1b*x1b - k*xi*(x1b-x2b) - k*(y1b-y2b) + F*c2 + st)
        k3x2 = omega*y2b; k3y2 = omega*(-xi*y2b + x2b - x2b*x2b*x2b - k*xi*(x2b-x1b) - k*(y2b-y1b) + F*c2)
        t4 = t+h; c4 = np.cos(omega*t4)
        x1c,y1c = x1+h*k3x1, y1+h*k3y1; x2c,y2c = x2+h*k3x2, y2+h*k3y2
        k4x1 = omega*y1c; k4y1 = omega*(-xi*y1c + x1c - x1c*x1c*x1c - k*xi*(x1c-x2c) - k*(y1c-y2c) + F*c4 + st)
        k4x2 = omega*y2c; k4y2 = omega*(-xi*y2c + x2c - x2c*x2c*x2c - k*xi*(x2c-x1c) - k*(y2c-y1c) + F*c4)
        x1 += (h/6.0)*(k1x1+2*k2x1+2*k3x1+k4x1); y1 += (h/6.0)*(k1y1+2*k2y1+2*k3y1+k4y1)
        x2 += (h/6.0)*(k1x2+2*k2x2+2*k3x2+k4x2); y2 += (h/6.0)*(k1y2+2*k2y2+2*k3y2+k4y2)
        out[i] = omega*(y1-y2)
    return out, x1, y1, x2, y2

def region_snr(seg, p, f=0.2):
    m = np.median(seg); A = seg[p]
    if A <= m: return None
    thr = m + f*(A-m)
    lo = p
    while lo-1 >= 0 and seg[lo-1] >= thr: lo -= 1
    hi = p
    while hi+1 < len(seg) and seg[hi+1] >= thr: hi += 1
    W = hi-lo+1; a = lo-W
    if a < 0: return None
    return (abs(seg[lo:hi+1]).sum()/(abs(seg[a:lo]).sum()+1e-30), W, A/m)

phi_main = 0.1846  # 旧全局值仅用于试验预测位置; 换用0.27?
phi_main = 0.27

for t0 in (10.0, 16.93):
    i0 = int(t0*fs)
    seg = np.asarray(s[i0:i0+int(0.02*fs)], dtype=np.float64)
    seg_n = (seg-seg.mean())/(seg.std()+1e-30)
    out, *_ = rk4_coupled(seg_n, 1/fs, 0.8, 4070.0, 3.0, 0.08, 0.0, 0.0, 0.0, 0.0)
    print('\n=== window t=%.2fs ===' % t0)
    for name, phi, color in (('MAIN', phi_main, 'M'), ('IP', (phi_main+0.5283) % 1.0, 'I')):
        print(' %s (phi=%.3f):' % (name, phi))
        for k in range(0, 13):
            tt = (k+phi)/f_eff
            if tt < 0 or tt > 0.02: continue
            i = int(round(tt*fs))
            lo_, hi_ = max(0, i-10), min(len(out), i+10)
            pk = lo_ + int(np.argmax(abs(out[lo_:hi_])))
            r = region_snr(abs(out), pk, 0.5)
            ri = region_snr(abs(seg_n), pk, 0.5)
            if r and r[2] > 3:
                print('   k=%2d out_snr=%.1f (W=%d pk/m=%.1f) | in_snr=%.1f' % (k, r[0], r[1], r[2], ri[0] if ri else -1))
