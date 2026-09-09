# -*- coding: utf-8 -*-
"""诊断13: 在t10_2s上比较不同SNR定义在真实主/间脉冲位置的数值尺度"""
import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
fs = 2.0e6
f_eff = 641.886226994
s = np.load('out_coh_t10_2s.npy', mmap_mode='r')
N = int(0.02*fs)
t_start = 10.0

# 折叠相位 (绝对时间)
nb = 2048
prof = np.zeros(nb); cnt = np.zeros(nb)
for j in range(s.size//N):
    seg = np.asarray(s[j*N:(j+1)*N], dtype=np.float64)
    t = t_start + (j*N + np.arange(len(seg)))/fs
    b = ((t*f_eff) % 1.0 * nb).astype(np.int64)
    prof += np.bincount(b, weights=seg, minlength=nb)
    cnt += np.bincount(b, minlength=nb)
prof = prof/np.maximum(cnt,1)
pk = int(np.argmax(prof))
phi_m = pk/nb
sep = int(round(0.5283*nb)); lo=(pk+sep-int(0.08*nb)) % nb; hi=(pk+sep+int(0.08*nb)) % nb
ip = (lo + int(np.argmax(np.roll(prof,-lo)[:hi-lo+nb]))) % nb if lo > hi else lo+int(np.argmax(prof[lo:hi]))
phi_i = ip/nb
print('phi_m=%.4f phi_i=%.4f' % (phi_m, phi_i))

# 统计: 每个真实主/间脉冲位置的 |z| 峰值, 及不同SNR定义
z5 = {'main': 0, 'ip': 0}; z20 = {'main': 0, 'ip': 0}
r1_5 = {'main': 0, 'ip': 0}; r1_20 = {'main': 0, 'ip': 0}
tot = {'main': 0, 'ip': 0}
vals = {'main': [], 'ip': []}
for j in range(s.size//N):
    seg = np.asarray(s[j*N:(j+1)*N], dtype=np.float64)
    z = (seg-seg.mean())/(seg.std()+1e-30)
    for label, phi in (('main', phi_m), ('ip', phi_i)):
        # 本窗内该类的每个周期
        for k in range(0, 14):
            # 相对时间 = 相位在窗内的时刻
            # 窗起点绝对相位
            ph0 = ((t_start + j*0.020)*f_eff) % 1.0
            # 目标: 窗内相对相位 rel 满足 (ph0 + rel*f_eff) = phi + n
            rel = ((phi - ph0) % 1.0)/f_eff
            if rel > 0.020: continue
            i = int(round(rel*fs))
            if i < 0 or i >= N: continue
            lo_, hi_ = max(0, i-10), min(N, i+10)
            kp = lo_ + int(np.argmax(z[lo_:hi_]))
            zp = z[kp]
            if zp < 2.0: continue
            tot[label] += 1
            r1 = zp/max(abs(z[kp-1]), 0.05) if kp > 0 else 0
            if zp > 5: z5[label] += 1
            if zp > 20: z20[label] += 1
            if r1 > 5: r1_5[label] += 1
            if r1 > 20: r1_20[label] += 1
            vals[label].append(zp)
print('真实脉冲样本数 (z>=2):', tot)
for label in ('main','ip'):
    a = np.array(vals[label])
    print('%s: z峰值 中位=%.1f 90%%=%.1f 最大=%.1f | z>5: %d  z>20: %d | 比值(峰/前样本)>5: %d  >20: %d' % (
        label, np.median(a) if len(a) else -1,
        np.percentile(a,90) if len(a) else -1,
        a.max() if len(a) else -1,
        z5[label], z20[label], r1_5[label], r1_20[label]))
