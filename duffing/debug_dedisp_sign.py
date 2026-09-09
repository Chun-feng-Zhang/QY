# -*- coding: utf-8 -*-
"""debug_dedisp_sign.py — 验证相干消色散相位符号(对真实数据折叠 S/N 对比)"""
import numpy as np, os, json, time
from coherent_dedisp import (parse_dada_header, parse_par, tau_disp,
                             build_dedisp_filter)

DADA = r'D:\VBshare\fast_1937+21\bcmin1937+21.dada'
HDR = r'D:\VBshare\fast_1937+21\bcmin1937+21.hdr'
PAR = r'D:\VBshare\fast_1937+21\B1937+21.par'
feff = 641.886226994
nb = 1024

hdr = parse_dada_header(HDR); par = parse_par(PAR)
fs = 1.0e6 / float(hdr['TSAMP'])
f_lo = float(hdr['FREQ']) - float(hdr['BW']) / 2.0
f_hi = float(hdr['FREQ']) + float(hdr['BW']) / 2.0
f_ref = float(hdr['FREQ'])
dm = float(par['DM'])
tau_smear = tau_disp(f_lo, dm) - tau_disp(f_hi, dm)
nfft = 1
while nfft < int(tau_smear * fs) + 1:
    nfft <<= 1
H, L_start, L_end = build_dedisp_filter(nfft, fs, f_lo, f_hi, f_ref, dm)
hop = nfft - L_start - L_end
print('nfft', nfft, 'L_start', L_start, 'L_end', L_end, 'hop', hop)

def fold(series, t0):
    N = len(series)
    ph = ((t0 + np.arange(N) / 2e6) * feff) % 1.0
    prof = np.bincount((ph * nb).astype(np.int64), weights=series, minlength=nb)
    cnt = np.bincount((ph * nb).astype(np.int64), minlength=nb)
    return prof / np.maximum(cnt, 1)

def process(sign, n_blocks=5):
    import scipy.fft as sfft
    Hs = H if sign > 0 else np.conj(H)
    fh = open(DADA, 'rb')
    fh.seek(4096 + int(10.0 * fs))
    series = []
    out_pos = 0
    f_out = 2e6
    dec = 500
    for b in range(n_blocks):
        raw = fh.read(nfft)
        if len(raw) < nfft:
            break
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        R = sfft.rfft(x, workers=-1)
        Z = np.zeros(nfft, dtype=np.complex64)
        Z[0] = R[0]; Z[1:nfft // 2] = 2.0 * R[1:nfft // 2]; Z[nfft // 2] = R[nfft // 2]
        Z[:nfft // 2 + 1] *= Hs
        z = sfft.ifft(Z, workers=-1)
        p = (z.real ** 2 + z.imag ** 2).astype(np.float32)
        pv = p[L_start:nfft - L_end]
        nv = pv.size - pv.size % dec
        down = pv[:nv].reshape(-1, dec).mean(axis=1).astype(np.float64)
        series.append(down)
    fh.close()
    s = np.concatenate(series).astype(np.float64)
    prof = fold(s, 10.0)
    m, sd = prof.mean(), prof.std()
    pk = int(np.argmax(prof))
    return (prof[pk] - m) / sd, pk, prof

for sign in (+1, -1):
    snr, pk, prof = process(sign)
    print('sign=%+d  S/N=%.1f  peak bin=%d' % (sign, snr, pk))
