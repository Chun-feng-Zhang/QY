# -*- coding: utf-8 -*-
"""debug_final3.py — 单块真实数据: RAW / NEG / POS 折叠对比"""
import numpy as np, scipy.fft as sfft

DISP_D_HZ = 4.148808e15
DADA = r'D:\VBshare\fast_1937+21\bcmin1937+21.dada'
feff = 641.886226994
fs = 1.0e9
f_lo, f_hi, f_ref = 1000.0, 1500.0, 1500.0
dm = 71.0237
nfft = 1 << 28

k = np.arange(nfft // 2 + 1, dtype=np.float64)
f_hz = (f_lo + k * (fs / nfft) / 1e6) * 1e6
f_ref_hz = f_ref * 1e6
exact = 2.0 * np.pi * DISP_D_HZ * dm * (1.0 / f_hz + f_hz / f_ref_hz ** 2 - 2.0 / f_ref_hz)
H_pos = np.exp(+1j * exact).astype(np.complex64)
H_neg = np.exp(-1j * exact).astype(np.complex64)

fh = open(DADA, 'rb'); fh.seek(4096 + int(10.0 * fs)); raw = fh.read(nfft); fh.close()
x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0


def fold(p, nb=256):
    ph = ((10.0 + np.arange(len(p)) / 2e6) * feff) % 1.0
    prof = np.bincount((ph * nb).astype(np.int64), weights=p, minlength=nb)
    cnt = np.bincount((ph * nb).astype(np.int64), minlength=nb)
    prof /= np.maximum(cnt, 1)
    m, sd = prof.mean(), prof.std()
    pk = int(np.argmax(prof))
    half = (prof[pk] + m) / 2.0
    return (prof[pk] - m) / sd, (prof > half).sum(), pk


def dedisp(H):
    R = sfft.rfft(x, workers=-1)
    Z = np.zeros(nfft, dtype=np.complex64)
    Z[0] = R[0]; Z[1:nfft // 2] = 2.0 * R[1:nfft // 2]; Z[nfft // 2] = R[nfft // 2]
    Z[:nfft // 2 + 1] *= H
    z = sfft.ifft(Z, workers=-1)
    p = (z.real ** 2 + z.imag ** 2).astype(np.float32)
    pv = p[:nfft - 163702053]
    nv = pv.size - pv.size % 500
    return pv[:nv].reshape(-1, 500).mean(axis=1).astype(np.float64)


# RAW
pr = (x ** 2).astype(np.float32); nv = pr.size - pr.size % 500
p_raw = pr[:nv].reshape(-1, 500).mean(axis=1).astype(np.float64)
print('RAW : S/N %.2f width %d peak %d' % fold(p_raw))
for name, H in [('NEG', H_neg), ('POS', H_pos)]:
    print('%s : S/N %.2f width %d peak %d' % ((name,) + fold(dedisp(H))))
print('>>> S/N 最高者为正确消色散符号')
