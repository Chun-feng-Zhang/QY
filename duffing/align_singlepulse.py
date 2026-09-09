# -*- coding: utf-8 -*-
"""残余色散精测 + 对齐后的单脉冲图"""
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

d = np.load('out_singlepulse_giant.npz')
rel, img, tr, t = d['rel'], d['img'], d['tr'], d['t_us']
f = d['f_mhz']
nb = rel.shape[0]
dt = t[1] - t[0]
F_REF = 1500.0
f_r = f  # 行中心频率

# 每行脉冲质心 (z>0.5峰 ±30μs内)
cent = np.zeros(nb)
amp = np.zeros(nb)
for r in range(nb):
    s = rel[r]
    m = np.abs(t) < 30
    k = int(np.argmax(s * m))
    pk = s[k]
    mm = (np.abs(t) < 15) & (s > 0.5 * pk)
    if mm.sum() > 0:
        cent[r] = (t[mm] * s[mm]).sum() / s[mm].sum()
    amp[r] = pk
    print('row%2d f=%7.1fMHz  质心=%+7.3fμs  峰z=%.1f' % (r, f[r], cent[r], amp[r]))

# 拟合比较: 1/f² (色散), 1/f, 线性 f
x2 = (1.0 / f_r ** 2 - 1.0 / F_REF ** 2) * 1e6
x1 = (1.0 / f_r - 1.0 / F_REF) * 1e3
xl = (f_r - F_REF) / 1e3
A = np.vstack([x2, np.ones_like(x2)]).T
k, c = np.linalg.lstsq(A, cent, rcond=None)[0]
res = cent - (k * x2 + c)
for name, xx in (('1/f^2', x2), ('1/f', x1), ('线性f', xl)):
    AA = np.vstack([xx, np.ones_like(xx)]).T
    kk, cc = np.linalg.lstsq(AA, cent, rcond=None)[0]
    rr = cent - (kk * xx + cc)
    print('模型 %-5s: rms=%.3f μs  max=%.3f μs' % (name, rr.std(), np.abs(rr).max()))
print('\n采用 1/f² 模型: 斜率 k=%.3f (μs / [Δ(1/f²)·1e6]), 截距=%+.3f μs' % (k, c))
print('残差: rms=%.3f μs, max=%.3f μs' % (res.std(), np.abs(res).max()))
# δDM: Δt_μs = 1e3·4.148808e3·δDM·Δ(1/f²) = k·(Δ(1/f²)·1e6) → δDM = k·1e6/(1e3·4.148808e3) = k/4.148808
# 注意: 本数据色散为常规 1000 倍口径, 换算到"该数据有效DM"时直接用常规公式得到的 δDM 需除以1000的偏差已含在内
#   这里报告"若按常规色散常数解释"的参数: 直接给出 Δτ(1000→1500)
dDM_conv = k / 4.148808   # 常规口径下
print('若按常规色散常数解释: δDM=%+.4f pc/cm³' % dDM_conv)
print('Δτ(1000→1500MHz) = %+.2f μs (带宽 500MHz)' % (k * (1 / 1000.0 ** 2 - 1 / F_REF ** 2) * 1e6))

# 对齐: 每行按模型平移 (-Δt) 使主爆垂直
shift_bin = np.round(-(k * x2 + c) / dt).astype(int)
rel_a = np.zeros_like(rel)
for r in range(nb):
    rel_a[r] = np.roll(rel[r], int(shift_bin[r]))
# 对齐后全带 trace (行均值, 在 ±30μs 有意义)
prof_a = rel_a.mean(axis=0)
print('\n对齐后行均值 trace: 峰t=%+.2fμs, 半高宽=%.2f μs' % (
    t[int(np.argmax(prof_a))],
    np.ptp(np.where(prof_a > 0.5 * prof_a.max())[0][[-1, 0]]) * dt if (prof_a > 0.5 * prof_a.max()).any() else -1))

fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                         gridspec_kw=dict(height_ratios=[1, 3.2]))
ax1 = axes[0]
ax1.plot(t, tr, 'k-', lw=0.8, label='原始(含残余色散)')
ax1.plot(t, prof_a, 'r-', lw=0.8, label='对齐后(行平均)')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_ylabel('power (z)')
ax1.set_xlim(-40, 40)
ax1.grid(alpha=0.3)
ax2 = axes[1]
vm = float(np.percentile(rel_a, 99.3))
ext = [t[0] - dt / 2, t[-1] + dt / 2, 1000.0, 1500.0]
im = ax2.imshow(rel_a, aspect='auto', origin='lower', extent=ext, cmap='viridis',
                norm=Normalize(vmin=-2, vmax=max(3, vm)))
ax2.set_xlim(-40, 40)
ax2.set_xlabel('t - t_giant (μs)')
ax2.set_ylabel('frequency (MHz)')
ax2.set_title('巨脉冲单脉冲 (按 1/f² 模型平移对齐)' % dDM_conv)
ax2.axvline(0, color='white', ls='--', lw=1)
cb = fig.colorbar(im, ax=ax2, pad=0.01)
cb.set_label('power (σ/行)')
fig.suptitle('原始图 out_singlepulse_giant.png 中观测到对角带 = 残余色散 %.1f μs/500MHz; 本图为按行平移校正后' % (
    k * ((1 / 1000.0 ** 2 - 1 / F_REF ** 2) * 1e6)), fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('out_singlepulse_giant_aligned.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('[saved] out_singlepulse_giant_aligned.png')
