# -*- coding: utf-8 -*-
"""
make_fold2d_raw.py — 从 raw .dada 直接做【相干消色散 × 频域子带】折叠动态谱
=========================================================================
原理: 与 coherent_dedisp.py 完全相同地: 每块 rfft → 解析谱 → 乘 H(消色散相位)
      → 把解析谱按子带切片(每个子带 500/R MHz) → 各自 IFFT → 子带功率
      → 按相位 φ=(t·f_eff) 折叠 → img[子带, 相位]。每个子带内部是相干消色散
      后的流, 各行与 f_ref=1500MHz 到达时刻对齐 → 主脉冲同一相位竖线。
用法:
  python make_fold2d_raw.py --t0 16.90 --dur 0.35 --rows 32 --nb 1024 --out out_fold2d_giant
  python make_fold2d_raw.py --t0 10.0  --dur 2.0  --rows 32 --nb 1024 --out out_fold2d_t10
"""
import argparse, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DADA = r'D:\VBshare\fast_1937+21\bcmin1937+21.dada'
HDR = r'D:\VBshare\fast_1937+21\bcmin1937+21.hdr'
PAR = r'D:\VBshare\fast_1937+21\B1937+21.par'
UTC_START = '2024-11-22-05:49:28'
FS = 1.0e9
NFFT = 1 << 28
L_START = 0
L_END = 163702053
HOP = NFFT - L_END            # 104733403
F_LO, F_HI, F_REF, DM = 1000.0, 1500.0, 1500.0, 71.0237

from coherent_dedisp import parse_dada_header
from duffing_mark_fold_t10_20ms import compute_feff, parse_par


def build_H():
    from coherent_dedisp import DISP_D_HZ, build_dedisp_filter
    H, _, _ = build_dedisp_filter(NFFT, FS, F_LO, F_HI, F_REF, DM)
    return H


def process(t0, dur, nrow, nb, out_png, out_npz, log_fp=None):
    f_eff, _, _, _ = compute_feff(parse_par(PAR), UTC_START)
    Pcur = 1.0 / f_eff
    print('[timing] f_eff=%.9f Hz  P=%.6f ms' % (f_eff, Pcur * 1e3))
    H = build_H()
    bw_row_mhz = 500.0 / nrow
    L_row = (NFFT // 2) // nrow          # 每子带 bin 数 (解析谱正频段划分)
    dec = NFFT // L_row                  # 子带流降采样因子(=2·nrow…注意: 正频段切片)
    nblk = int(np.ceil(dur / (HOP / FS))) + 1
    acc = np.zeros((nrow, nb), dtype=np.float64)
    cnt = np.zeros((nrow, nb), dtype=np.float64)
    t_start = time.time()
    i0_byte = 4096 + int(t0 * FS)
    with open(DADA, 'rb') as fh:
        for blk in range(nblk):
            fh.seek(i0_byte + blk * HOP, os.SEEK_SET)
            raw = fh.read(NFFT)
            if len(raw) < NFFT:
                break
            x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
            R = np.fft.rfft(x).astype(np.complex64)      # (NFFT//2+1)
            Z = np.zeros(NFFT, dtype=np.complex64)
            Z[0] = R[0]
            Z[1:NFFT // 2] = 2.0 * R[1:NFFT // 2]
            Z[NFFT // 2] = R[NFFT // 2]
            Z[:NFFT // 2 + 1] *= H
            # 行时间栅格: 子带流采样间距 = dec/FS?  见推导: y[m] ↔ t = blk*HOP/FS + m*dec/FS
            dec = NFFT // L_row
            m_valid = (NFFT - L_END) // dec
            t_abs0 = t0 + blk * HOP / FS
            tm = t_abs0 + (np.arange(m_valid) * dec) / FS
            ph = ((tm * f_eff) % 1.0)
            b = (ph * nb).astype(np.int64)
            b %= nb
            order = np.argsort(b, kind='stable')
            bs = b[order]
            cut = np.where(np.diff(bs) != 0)[0] + 1
            for r in range(nrow):
                b0 = r * L_row
                Zs = Z[b0:b0 + L_row].copy()
                y = np.fft.ifft(Zs)                     # 长度 L_row
                sc = (L_row / float(NFFT)) ** 2
                P = (y.real ** 2 + y.imag ** 2)[:m_valid] * sc
                Ps = P[order]
                if cut.size == 0:
                    acc[r, bs[0]] += Ps.sum()
                    cnt[r, bs[0]] += Ps.size
                else:
                    cs = np.concatenate(([0], cut, [Ps.size]))
                    sums = np.add.reduceat(Ps, cs[:-1])
                    cols = bs[cs[:-1]]
                    acc[r, cols] += sums
                    cc = np.concatenate(([0], cut, [Ps.size]))
                    cnts = np.diff(cc)
                    cnt[r, cols] += cnts
            if log_fp is None:
                print('[blk %d/%d] t=%.3fs  %.0fs' % (
                    blk, nblk, t_abs0 + (NFFT - L_END) / FS, time.time() - t_start), flush=True)
            elif blk % 5 == 0:
                print('[blk %d] %.0fs' % (blk, time.time() - t_start), flush=True, file=log_fp)
                log_fp.flush()
    img = acc / np.maximum(cnt, 1.0)
    # 行中值归一化(去带通/RFI基线) → 相对功率 σ/行
    row_med = np.median(img, axis=1, keepdims=True)
    row_mad = 1.4826 * np.median(np.abs(img - row_med), axis=1, keepdims=True)
    rel = (img - row_med) / np.maximum(row_mad, 1e-12)
    np.savez(out_npz, img=img, rel=rel, cnt=cnt,
             f_eff=f_eff, t0=t0, dur=dur, nrow=nrow, nb=nb)
    print('[done] %s  (%.0fs)' % (out_npz, time.time() - t_start))
    return img, rel, f_eff


def plot(img, rel, f_eff, t0, dur, nrow, nb, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12, 8))
    extent = [0, 1, F_LO, F_HI]
    vmax = float(np.percentile(rel, 99.0))
    im = ax.imshow(rel, aspect='auto', origin='lower', extent=extent,
                   cmap='viridis', norm=Normalize(vmin=-2, vmax=max(3, vmax)))
    ax.set_xlabel('phase'); ax.set_ylabel('frequency (MHz)')
    ax.set_title('折叠动态谱 t=[%.2f,%.2f]s  (%d 子带 × %.1f MHz, 相干消色散, 行归一化 σ)' % (
        t0, t0 + dur, nrow, 500.0 / nrow))
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label('power (σ/行)')
    # 顶行: 全带折叠剖面 (同 2MHz 数据折叠, 画在同一个图的右上方小窗? 不: 用第二页/顶部轴)
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    prof = img.sum(axis=0) / img.shape[0]
    pm = float(np.median(prof)); ps = float(1.4826 * np.median(np.abs(prof - pm)))
    xs = np.arange(nb) / nb
    ax2.plot(xs, (prof - pm) / ps, 'b-', lw=1)
    ax2.set_xlim(0, 1); ax2.set_xlabel('phase')
    ax2.set_ylabel('power (σ, 行平均)')
    ax2.set_title('行平均折叠剖面 (由子带折叠求和)')
    ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches='tight')
    fig2.tight_layout(); fig2.savefig(out_png.replace('.png', '_prof.png'), dpi=140, bbox_inches='tight')
    plt.close(fig); plt.close(fig2)
    print('[plot]', out_png, '&', out_png.replace('.png', '_prof.png'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--t0', type=float, default=10.0)
    ap.add_argument('--dur', type=float, default=2.0)
    ap.add_argument('--rows', type=int, default=32)
    ap.add_argument('--nb', type=int, default=1024)
    ap.add_argument('--out', default='out_fold2d')
    args = ap.parse_args()
    img, rel, f_eff = process(args.t0, args.dur, args.rows, args.nb,
                              args.out + '.png', args.out + '.npz')
    plot(img, rel, f_eff, args.t0, args.dur, args.rows, args.nb, args.out + '.png')


if __name__ == '__main__':
    main()
