# -*- coding: utf-8 -*-
"""诊断4: 真实主脉冲在用户SNR定义下的数值"""
import numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
fs = 2.0e6
f_eff = 641.886226994
P = 1.0/f_eff
s = np.load('out_coh_full.npy', mmap_mode='r')

def fold_phase(seg):
    nb = 1024
    t = np.arange(len(seg))/fs
    ph = (t*f_eff) % 1.0
    prof = np.bincount((ph*nb).astype(np.int64), weights=seg, minlength=nb)
    cnt = np.bincount((ph*nb).astype(np.int64), minlength=nb)
    prof = prof/np.maximum(cnt,1)
    pk = int(np.argmax(prof))
    ip_c = (pk + int(round(0.5283*nb))) % nb
    return pk/nb, ip_c/nb

def snrW(z, p, W):
    half=W//2; a=p-half
    if a-W<0 or a+W+W>len(z): return np.nan
    return np.abs(z[a:a+W]).sum()/(np.abs(z[a-W:a]).sum()+1e-30)

t0 = 10.0
i0 = int(t0*fs); seg = np.asarray(s[i0:i0+int(0.02*fs)], dtype=np.float64)
z = (seg-seg.mean())/(seg.std()+1e-30)
phi_m, phi_ip = fold_phase(z)
print('fold: phi_main=%.4f phi_ip=%.4f sep=%.4f' % (phi_m, phi_ip, (phi_ip-phi_m)%1.0))

# 主脉冲预测位置 + 局部峰值搜索
for name, phi in (('MAIN', phi_m), ('IP', phi_ip)):
    print('\n===', name, 'phi=%.4f ===' % phi)
    for k in range(-1, 14):
        tt = (k + phi)/f_eff           # 相对窗内时间
        if tt < 0 or tt > 0.02: continue
        i = int(round(tt*fs))
        lo, hi = max(0,i-40), min(len(z), i+40)
        if hi-lo < 5: continue
        pk = lo + int(np.argmax(z[lo:hi]))      # 局部正峰
        zp = z[pk]
        row = []
        for W in (4, 8, 16, 40, 80, 160):
            row.append('W%-3d=%.2f' % (W, snrW(z, pk, W)))
        print('k=%2d pred=%7.1fus pk=%6d z=%5.2f | %s' % (k, tt*1e6, pk, zp, '  '.join(row)))
