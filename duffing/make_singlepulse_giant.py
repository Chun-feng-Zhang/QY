# -*- coding: utf-8 -*-
"""
make_singlepulse_giant.py — 巨脉冲单脉冲动态谱 (不做折叠)
窗口: t_giant=16.9302305 s (f_ref=1500MHz 到达) ±1 ms
方法: 与 make_fold2d_raw 相同 → 相干消色散谱频域切 32 子带(15.625MHz) → IFFT
      → 子带功率流(64ns/点) → 不折叠, 直接按时间排列 → 时间×频段图
输出: out_singlepulse_giant.png / .npz
"""
import os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DADA = r'D:\VBshare\fast_1937+21\bcmin1937+21.dada'
PAR = r'D:\VBshare\fast_1937+21\B1937+21.par'
UTC_START = '2024-11-22-05:49:28'
FS = 1.0e9
NFFT = 1 << 28
L_START, L_END = 0, 163702053
HOP = NFFT - L_END
F_LO, F_HI, F_REF, DM = 1000.0, 1500.0, 1500.0, 71.0237

from coherent_dedisp import build_dedisp_filter
from duffing_mark_fold_t10_20ms import compute_feff, parse_par

T_GIANT = 16.9302305      # 消色散后巨脉冲时刻 (2MHz 数据测得, f_ref=1500MHz 到达)
T_BLOCK = 16.90           # 处理块起点 (巨脉冲位于块内有效区)
T_HALF = 1.0e-3           # ±1 ms
BIN_NS = 512              # 显示时间像素 = 8 行采样 × 64 ns

def main():
    f_eff, _, _, _ = compute_feff(parse_par(PAR), UTC_START)
    P_ms = 1.0 / f_eff * 1e3
    print('[timing] f_eff=%.9f  P=%.4f ms' % (f_eff, P_ms))
    nrow = 32
    L_row = (NFFT // 2) // nrow
    dec = NFFT // L_row          # 64
    H, _, _ = build_dedisp_filter(NFFT, FS, F_LO, F_HI, F_REF, DM)

    t0 = time.time()
    with open(DADA, 'rb') as fh:
        fh.seek(4096 + int(T_BLOCK * FS), os.SEEK_SET)
        raw = fh.read(NFFT)
    x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
    R = np.fft.rfft(x).astype(np.complex64)
    Z = np.zeros(NFFT, dtype=np.complex64)
    Z[0] = R[0]
    Z[1:NFFT // 2] = 2.0 * R[1:NFFT // 2]
    Z[NFFT // 2] = R[NFFT // 2]
    Z[:NFFT // 2 + 1] *= H
    del R, x
    m_valid = (NFFT - L_END) // dec
    sc = (L_row / float(NFFT)) ** 2
    # 每行功率流 (15.625 Msps)
    Prows = np.empty((nrow, m_valid), dtype=np.float32)
    for r in range(nrow):
        b0 = r * L_row
        Zs = Z[b0:b0 + L_row].copy()
        y = np.fft.ifft(Zs)
        P = (y.real ** 2 + y.imag ** 2)[:m_valid] * sc
        Prows[r] = P.astype(np.float32)
    del Z
    print('[proc] 行功率流完成 %.0fs' % (time.time() - t0))

    # 巨脉冲对应的行采样
    m_g = int(round((T_GIANT - T_BLOCK) * FS / dec))
    print('m_g =', m_g, ' (m_valid=%d)' % m_valid)
    # 全带折叠(行求和)功率 trace 确认巨脉冲位置
    full = Prows.sum(axis=0)
    t_full = T_BLOCK + np.arange(m_valid) * dec / FS
    w = int(0.01e-3 * FS / dec)  # ±10μs 窗口找峰
    i0 = max(0, m_g - w); i1 = min(m_valid, m_g + w)
    k = i0 + int(np.argmax(full[i0:i1]))
    print('全带功率峰 @ t=%.7f s (偏离预期 %+.2f μs), 峰/中值=%.1f' % (
        t_full[k], (t_full[k] - T_GIANT) * 1e6, full[k] / np.median(full)))

    # 时间窗口 & 显示 binning (BIN_NS)
    m0 = max(0, m_g - int(T_HALF * FS / dec))
    m1 = min(m_valid, m_g + int(T_HALF * FS / dec))
    nbin = (m1 - m0) // (BIN_NS // dec)
    img = np.zeros((nrow, nbin), dtype=np.float64)
    for r in range(nrow):
        seg = Prows[r, m0:m0 + nbin * (BIN_NS // dec)].astype(np.float64)
        img[r] = seg.reshape(nbin, BIN_NS // dec).mean(axis=1)
    t_img = T_BLOCK + (m0 + (np.arange(nbin) + 0.5) * (BIN_NS // dec)) * dec / FS
    dt_us = (t_img - T_GIANT) * 1e6

    # 行归一化 σ
    med = np.median(img, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(img - med), axis=1, keepdims=True)
    rel = (img - med) / np.maximum(mad, 1e-12)

    # 全带 trace (同时间网格)
    seg = full[m0:m0 + nbin * (BIN_NS // dec)]
    tr = seg.reshape(nbin, BIN_NS // dec).mean(axis=1)
    tr = (tr - np.median(tr)) / (1.4826 * np.median(np.abs(tr - np.median(tr))))

    np.savez('out_singlepulse_giant.npz', rel=rel, img=img, tr=tr,
             t_us=dt_us, f_mhz=F_LO + (np.arange(nrow) + 0.5) * (F_HI - F_LO) / nrow,
             t_giant=T_GIANT, f_eff=f_eff)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 3.2]))
    ax1 = axes[0]
    ax1.plot(dt_us, tr, 'k-', lw=0.8)
    ax1.set_ylabel('全带功率 (z)')
    ax1.set_title('B1937+21 巨脉冲单脉冲 (t=%.6f s 到达@1500MHz; 相干消色散; 主=%.6fms)' % (
        T_GIANT, P_ms))
    ax1.grid(alpha=0.3)
    ax2 = axes[1]
    vm = float(np.percentile(rel, 99.3))
    im = ax2.imshow(rel, aspect='auto', origin='lower',
                    extent=[dt_us[0] - (dt_us[1] - dt_us[0]) / 2,
                            dt_us[-1] + (dt_us[1] - dt_us[0]) / 2,
                            F_LO, F_HI],
                    cmap='viridis', norm=Normalize(vmin=-2, vmax=max(3, vm)))
    ax2.set_xlabel('t - t_giant (μs)')
    ax2.set_ylabel('frequency (MHz)')
    ax2.set_title('时间×频段单脉冲动态谱 (32子带×15.625MHz, 时间分辨率 %.0f ns, 行归一化 σ)' % BIN_NS)
    ax2.axvline(0, color='white', ls='--', lw=1)
    ax2.axvline(0.5293 * P_ms * 1e3, color='orange', ls='--', lw=1)  # 若巨脉冲伴中间脉冲
    cb = fig.colorbar(im, ax=ax2, pad=0.01)
    cb.set_label('power (σ/行)')
    fig.tight_layout()
    fig.savefig('out_singlepulse_giant.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('[saved] out_singlepulse_giant.png  (%.0fs)' % (time.time() - t0))
    print('时间轴: %.1f .. %.1f μs, %d 列, 每列 %.0f ns' % (
        dt_us[0], dt_us[-1], nbin, BIN_NS))

if __name__ == '__main__':
    main()
