# -*- coding: utf-8 -*-
"""合成最终折叠图: 上=相位-功率(z, 主/间SNR标注), 下=相位×1.0-1.5GHz 动态谱"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from duffing_mark_fold_t10_20ms import (fold_profile, pulse_snr,
                                        parse_par, compute_feff)
FS = 2.0e6
t_start = 10.0
series = np.load('out_coh_t10_2s.npy', mmap_mode='r')
par = parse_par(r'D:\VBshare\fast_1937+21\B1937+21.par')
f_eff, _, _, _ = compute_feff(par, '2024-11-22-05:49:28')

nb = 4096
prof = fold_profile(series, FS, f_eff, t_start, nb=nb)
mp = float(np.median(prof)); sp = float(1.4826 * np.median(np.abs(prof - mp)))
pk = int(np.argmax(prof))
# 主相位=argmax; 间相位=搜索 0.5283 附近次峰
sep_bin = int(round(0.5283 * nb))
lo = (pk + sep_bin - int(0.08 * nb)) % nb
hi = (pk + sep_bin + int(0.08 * nb)) % nb
if lo < hi:
    ip = lo + int(np.argmax(prof[lo:hi]))
else:
    roll = np.roll(prof, -lo)
    ip = (lo + int(np.argmax(roll[:hi - lo + nb]))) % nb
phi_m, phi_i = pk / nb, ip / nb
r_main = pulse_snr(prof, pk, mp, sp)
r_ip = pulse_snr(prof, ip, mp, sp)
print('phi_m=%.4f phi_i=%.4f  main_snr=%.4f(W=%d) ip_snr=%.4f(W=%d)' % (
    phi_m, phi_i, r_main['snr'], r_main['hi'] - r_main['lo'] + 1,
    r_ip['snr'], r_ip['hi'] - r_ip['lo'] + 1))

# 2D
d = np.load('out_fold2d_t10.npz')
rel = d['rel']; nb2 = int(d['nb']); nrow = int(d['nrow'])
F_LO, F_HI = 1000.0, 1500.0

prof_z = (prof - prof.mean()) / (prof.std() + 1e-30)
xs = np.arange(nb) / nb
fig, axes = plt.subplots(2, 1, figsize=(13, 10))
ax1 = axes[0]
ax1.plot(xs, prof_z, 'b-', lw=1.0)
ax1.set_xlim(0, 1)
ax1.set_ylabel('power (z)')
ax1.set_xlabel('phase')
ax1.set_title('折叠 profile (t=10.0-12.0s, 全带 2MHz 消色散数据, nb=%d)' % nb)
ax1.grid(alpha=0.3)
if r_main:
    lo_, hi_ = r_main['lo'], r_main['hi']
    W = hi_ - lo_ + 1
    ax1.axvspan(xs[lo_], xs[hi_], color='red', alpha=0.25, lw=0)
    ax1.axvspan(xs[lo_ - W], xs[lo_], color='gray', alpha=0.45, lw=0)
    ax1.text(0.02, 0.97,
             '主脉冲 SNR = %.3f  (on 幅度和 ÷ 前缘等宽基线幅度和, on区 %d bin)' % (r_main['snr'], W),
             transform=ax1.transAxes, fontsize=10, color='darkred',
             bbox=dict(boxstyle='round', fc='white', ec='darkred', alpha=0.92))
if r_ip:
    ax1.axvspan(xs[r_ip['lo']], xs[r_ip['hi']], color='orange', alpha=0.25, lw=0)
ax1.axvline(phi_m, color='red', ls='--', lw=1)
ax1.axvline(phi_i, color='orange', ls='--', lw=1)

ax2 = axes[1]
vmax = float(np.percentile(rel, 99.5))
im = ax2.imshow(rel, aspect='auto', origin='lower',
                extent=[0, 1, F_LO, F_HI], cmap='viridis',
                norm=Normalize(vmin=-2, vmax=max(3.0, vmax)))
ax2.set_xlabel('phase')
ax2.set_ylabel('frequency (MHz)')
ax2.set_title('相位×频段折叠图 (raw 相干消色散后频域切 32 子带 ×15.6MHz, 行归一化 σ; 颜色=功率)')
ax2.axvline(phi_m, color='white', ls='--', lw=1)
ax2.axvline(phi_i, color='white', ls='--', lw=0.8)
cb = fig.colorbar(im, ax=ax2, pad=0.01)
cb.set_label('power (σ/行)')
fig.tight_layout()
fig.savefig('out_t10_fold_2d.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('[saved] out_t10_fold_2d.png')
