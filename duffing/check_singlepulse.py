# -*- coding: utf-8 -*-
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
d = np.load('out_singlepulse_giant.npz')
rel, tr, t = d['rel'], d['tr'], d['t_us']
f = d['f_mhz']; P_ms = 1.0 / float(d['f_eff']) * 1e3
print('时间轴范围 %.1f..%.1f μs' % (t[0], t[-1]))
sel = np.abs(t) < 200
pk_t = []
for r in range(rel.shape[0]):
    s = rel[r]
    k = np.argmax(s * sel)
    pk_t.append(t[k])
    print('row %2d f=%7.1fMHz  峰t=%+7.2fμs  z=%.1f' % (r, f[r], t[k], s[k]))
pk_t = np.array(pk_t)
print('主脉冲各行峰到达: 中位=%+.2f μs, 离散 std=%.2f μs' % (np.median(pk_t), pk_t.std()))
# 峰z统计
zs = [rel[r, np.argmax(rel[r] * (np.abs(t) < 200))] for r in range(rel.shape[0])]
print('行峰z: 中位=%.1f  min=%.1f  max=%.1f' % (np.median(zs), np.min(zs), np.max(zs)))
# 间脉冲位置 (主+0.5293P = +824μs)
ip_us = 0.5293 * P_ms * 1e3
w = int(5 / (t[1] - t[0]))
if int(ip_us) < t[-1]:
    m = np.abs(t - ip_us) < 200
    mx = rel[:, m].max()
    print('间脉冲相位带(≈+%.0fμs, ±200μs窗): 行最大 z=%.1f' % (ip_us, mx))
# 时间结构: 巨脉冲总宽度
zz = tr * (np.abs(t) < 200)
thr = 0.5 * tr[np.argmax(zz)]
seg = np.where(zz >= thr)[0]
if seg.size:
    print('全带 trace 半高宽: %.2f μs' % ((seg[-1] - seg[0]) * (t[1] - t[0])))
