# -*- coding: utf-8 -*-
"""诊断6: 自适应 on-pulse 区域 (从峰值向下扩展到阈值以下) + 前沿等宽基线, 测真实巨脉冲与普通脉冲"""
import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
fs = 2.0e6
f_eff = 641.886226994
s = np.load('out_coh_full.npy', mmap_mode='r')

def zscore_block(i0, n=int(0.1*fs)):
    seg = np.asarray(s[i0:i0+n], dtype=np.float64)
    return (seg-seg.mean())/(seg.std()+1e-30)

def adapt_snr(z, p, f=0.5):
    """on-region: 从峰值向两侧扩展到 z < b + f*(A-b); b=median|z|; 基线=等宽紧邻前沿"""
    A = z[p]
    b = np.median(np.abs(z))
    thr = b + f*(A-b)
    lo = p
    while lo-1 >= 0 and z[lo-1] >= thr: lo -= 1
    hi = p
    while hi+1 < len(z) and z[hi+1] >= thr: hi += 1
    W = hi-lo+1
    a = lo - W
    if a < 0: return None
    on = np.abs(z[lo:hi+1]).sum()
    base = np.abs(z[a:lo]).sum()
    return dict(lo=lo, hi=hi, W=W, A=A, snr=on/(base+1e-30))

# 巨脉冲事件定位
i0 = int(16.93*fs)
z = zscore_block(i0-100)
giant = int(np.argmax(z))
print('giant block max: local idx', giant, 'abs idx', i0-100+giant, 'z=%.1f' % z[giant])
for f in (0.3, 0.5, 0.7):
    r = adapt_snr(z, giant, f)
    print(' f=%.1f ->' % f, r)
# 看巨脉冲周围形状
print('\ngiant neighbourhood z:')
lo, hi = giant-6, giant+8
print(' '.join('%d:%.1f' % (i, z[i]) for i in range(lo, hi+1)))

# 普通主脉冲 (取 t=10.000-10.020 段内相位聚类最强几个)
phi_main = 0.27   # 诊断5聚类近似
t0 = 10.0
z2 = zscore_block(int(t0*fs), int(0.02*fs))
P = 1.0/f_eff
print('\nnormal pulses @ t=10s window, predicted main phase %.3f:' % phi_main)
for k in range(0, 13):
    tt = (k+phi_main)/f_eff
    i = int(round(tt*fs))
    lo_, hi_ = max(0,i-10), min(len(z2), i+10)
    pk = lo_ + int(np.argmax(z2[lo_:hi_]))
    if z2[pk] < 2.5: continue
    r = adapt_snr(z2, pk, 0.5)
    print('k=%2d z=%5.2f' % (k, z2[pk]), '->', r)
