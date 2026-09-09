# -*- coding: utf-8 -*-
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
d = np.load('out_singlepulse_giant.npz')
rel, tr, t = d['rel'], d['tr'], d['t_us']
f = d['f_mhz']
dt = t[1] - t[0]
print('时间像素=%.3f μs' % dt)
# 全带 trace 形态 (z): ±30μs
m = np.abs(t) < 30
print('全带 trace @ -25,-20,-15,-12,-10,-8,-6,-4,-2,0,+2,+4,+6,+8,+10 μs:')
for tv in (-25, -20, -15, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10):
    k = int(np.argmin(np.abs(t - tv)))
    print('  t=%+3dμs: z=%7.1f' % (tv, tr[k]))
# 每行在固定 t 的 z
print('\n每行 z@ (t=-14,-12,-10,-8,-6,-4,-2,0,+2μs):')
for r in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30):
    row = rel[r]
    vals = []
    for tv in (-14, -12, -10, -8, -6, -4, -2, 0, 2):
        k = int(np.argmin(np.abs(t - tv)))
        vals.append(row[k])
    print('row%2d %6.1fMHz: ' % (r, f[r]) + ' '.join('%6.1f' % v for v in vals))
# 亮区统计: 每行 z>50 的时间范围
print('\n每行 z>50 的时间区间 (μs):')
for r in range(rel.shape[0]):
    s = rel[r]
    on = np.where(s > 50)[0]
    if on.size == 0:
        continue
    # 分组连续段
    gs = np.split(on, np.where(np.diff(on) > 3)[0] + 1)
    segs = ['%+.1f..%+.1f' % (t[g[0]], t[g[-1]]) for g in gs]
    print('row%2d %6.1fMHz: ' % (r, f[r]) + ' | '.join(segs))
