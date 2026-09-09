# -*- coding: utf-8 -*-
"""验证: 巨脉冲折叠动态谱各行对齐 + 主脉冲相位"""
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

d = np.load('out_fold2d_giant.npz')
img, rel, cnt = d['img'], d['rel'], d['cnt']
nb = int(d['nb']); nrow = int(d['nrow']); f_eff = float(d['f_eff'])
F_LO, F_HI = 1000.0, 1500.0
print('img', img.shape, 'rel', rel.shape)
xs = np.arange(nb) / nb

# 巨脉冲绝对相位 (与 2MHz 消色散数据一致)
t_g = 16.9302305
phi_g = (t_g * f_eff) % 1.0
print('预期巨脉冲绝对相位 φ=%.4f (bin %d/%d)' % (phi_g, phi_g * nb, nb))

# 各行最大列
cols = []
for r in range(nrow):
    k = int(np.argmax(rel[r]))
    cols.append(k)
    f_c = F_LO + (r + 0.5) * (F_HI - F_LO) / nrow
    print('row %2d f=%7.1f MHz 峰相位=%.4f (偏离预期 %+.4f)  峰σ=%.1f' % (
        r, f_c, xs[k], (xs[k] - phi_g + 0.5) % 1.0 - 0.5, rel[r, k]))
cols = np.array(cols)
print('列峰散布: std=%.2f bins, 中位偏离=%.3f' % (
    cols.std(), np.median(((xs[cols] - phi_g + 0.5) % 1.0) - 0.5)))
# 行平均剖面峰值
prof = img.mean(axis=0)
pm = np.median(prof); ps = 1.4826 * np.median(np.abs(prof - pm))
zp = (prof - pm) / ps
k0 = int(np.argmax(zp))
print('行平均剖面峰: φ=%.4f (预期 %.4f), z=%.1f' % (xs[k0], phi_g, zp[k0]))
# ip 处
phi_ip = (0.7622)
print('ip 位置 z=%.1f' % zp[int(phi_ip * nb)])
