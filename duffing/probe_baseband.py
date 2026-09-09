# -*- coding: utf-8 -*-
"""
probe_baseband.py  —  基带数据格式探针 / baseband format probe
============================================================
目的 / Purpose:
  确认 bcmin1937+21.dada 是 8-bit 实数还是复数、采样率、频带结构、
  直流偏移与噪声统计，为相干消色散 (coherent dedispersion) 提供依据。

用法 / Usage:
  python probe_baseband.py --dada <file.dada> --hdr <file.hdr> --par <file.par> \
      [--n-block 16] [--fft 262144]
"""
import argparse
import os
import numpy as np
import sys

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

C_LIGHT = 299792458.0
DISP_D = 4.148808e3          # 色散常数 / dispersion const (MHz^2 cm^3 pc^-1 s)


def parse_dada_header(path):
    hdr = {}
    with open(path, 'rb') as f:
        blob = f.read(4096)
    for line in blob.decode('latin1').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            hdr[parts[0].strip('*').strip()] = parts[1].split('#')[0].strip()
    return hdr


def parse_par(path):
    par = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] not in par:
                par[parts[0]] = parts[1]
    return par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dada', default=r'D:\VBshare\fast_1937+21\bcmin1937+21.dada')
    ap.add_argument('--hdr', default=r'D:\VBshare\fast_1937+21\bcmin1937+21.hdr')
    ap.add_argument('--par', default=r'D:\VBshare\fast_1937+21\B1937+21.par')
    ap.add_argument('--n-block', type=int, default=16, help='统计用块数 / blocks for stats')
    ap.add_argument('--fft', type=int, default=262144, help='谱分析 FFT 长度')
    args = ap.parse_args()

    hdr = parse_dada_header(args.hdr)
    par = parse_par(args.par)
    fs = 1.0e6 / float(hdr['TSAMP'])            # Hz
    f_lo = float(hdr['FREQ']) - float(hdr['BW']) / 2.0
    f_hi = float(hdr['FREQ']) + float(hdr['BW']) / 2.0
    dm = float(par.get('DM', 0.0))
    dt_smear = DISP_D * dm * (1.0 / f_lo ** 2 - 1.0 / f_hi ** 2)

    fsz = os.path.getsize(args.dada)
    dur = fsz / float(hdr.get('BYTES_PER_SECOND', fs))

    print('=' * 70)
    print('[hdr]', {k: hdr.get(k) for k in
                    ('TSAMP', 'BW', 'FREQ', 'NBIT', 'NCHAN', 'NDIM', 'NPOL',
                     'BYTES_PER_SECOND', 'UTC_START', 'SOURCE', 'INSTRUMENT')})
    print('[par] DM=%.4f  P0=%.12f s  F0=%.12f Hz' % (
        dm, float(par['P0']), float(par['F0'])))
    print('[derived] fs=%.6g Hz  f_lo=%.3f MHz  f_hi=%.3f MHz' % (fs, f_lo, f_hi))
    print('[derived] dispersive smear dt=%.4f ms  (决定相干消色散 FFT 长度)' % (dt_smear * 1e3))
    print('[file] size=%.6f GiB  dur=%.4f s' % (fsz / 2 ** 30, dur))

    # ---- 读取若干块，做幅值/DC/频谱统计 ----
    rng = np.random.default_rng(0)
    chunk = args.fft * 8
    fh = open(args.dada, 'rb')
    fh.seek(4096)
    means, stds, acc_psd = [], [], None
    npsd = 0
    nblk = 0
    for b in range(args.n_block):
        raw = fh.read(chunk)
        if len(raw) < args.fft:
            break
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = x[:x.size - x.size % args.fft].reshape(-1, args.fft)
        means.append(x.mean())
        stds.append(x.std())
        for row in x[::4]:
            X = np.fft.rfft(row)
            p = (X.real ** 2 + X.imag ** 2) / args.fft
            if acc_psd is None:
                acc_psd = p.copy()
            else:
                acc_psd += p
            npsd += 1
        nblk += 1
    fh.close()
    means = np.array(means)
    stds = np.array(stds)
    psd = acc_psd / npsd
    faxis = np.fft.rfftfreq(args.fft, 1.0 / fs) / 1e6   # MHz

    print('-' * 70)
    print('[amp stats]  uint8 mean=%.3f  std=%.3f  (mean~127.5 表示 unsigned 噪声)' % (
        means.mean(), stds.mean()))
    print('[amp stats]  per-block mean range [%.2f, %.2f]' % (means.min(), means.max()))

    # 对称性：若为实数，rfft 幅度谱应关于 DC 无共轭对问题；
    # 复数 I/Q 交错则均值/谱会有不同表现。这里检查频带边缘能量。
    half = args.fft // 2
    p_lo = psd[1:half // 4].mean()
    p_mid = psd[half // 4:half // 2].mean()
    p_hi = psd[half // 2:half].mean()
    print('[spectrum] PSD band avg  low=%.4g  mid=%.4g  high=%.4g' % (p_lo, p_mid, p_hi))
    # 顶部 bin（Nyquist）能量：实数带限信号在 500 MHz 边缘应下降
    print('[spectrum] PSD[DC]=%.4g  PSD[Nyquist]=%.4g  (实数带限→Nyquist 附近下降)' % (
        psd[0], psd[-1]))
    # 峰值频率位置
    ipk = int(np.argmax(psd[1:])) + 1
    print('[spectrum] peak bin=%d  f=%.3f MHz  (若接近 DC/窄带 → 可能存在直流/RFI)' % (
        ipk, faxis[ipk]))

    print('=' * 70)
    print('结论 / conclusion:')
    print('  采样率 fs=%.4g Hz, 实/复需结合谱对称性与均值判断。' % fs)
    print('  相干消色散需 FFT 块长 >= %.1f ms (=%d 样本) 覆盖色散展宽。' % (
        dt_smear * 1e3, int(np.ceil(dt_smear * fs))))


if __name__ == '__main__':
    main()
