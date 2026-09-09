# -*- coding: utf-8 -*-
"""debug_sign_real.py — 真实数据双符号对照(用 scipy FFT, 3块), 折叠比 S/N 与宽度"""
import numpy as np, time
import scipy.fft as sfft
from coherent_dedisp import parse_dada_header, parse_par, tau_disp

DADA = r'D:\VBshare\fast_1937+21\bcmin1937+21.dada'
HDR = r'D:\VBshare\fast_1937+21\bcmin1937+21.hdr'
PAR = r'D:\VBshare\fast_1937+21\B1937+21.par'
DISP_D_HZ = 4.148808e15
feff = 641.886226994
nb = 256

hdr = parse_dada_header(HDR); par = parse_par(PAR)
fs = 1e6 / float(hdr['TSAMP'])
f_lo = float(hdr['FREQ']) - float(hdr['BW']) / 2.0
f_hi = float(hdr['FREQ']) + float(hdr['BW']) / 2.0
f_ref = float(hdr['FREQ'])
dm = float(par['DM'])
tau_smear = tau_disp(f_lo, dm) - tau_disp(f_hi, dm)
nfft = 1
while nfft < int(tau_smear * fs) + 1:
    nfft <<= 1

k = np.arange(nfft // 2 + 1, dtype=np.float64)
f_rf_mhz = f_lo + k * (fs / nfft) / 1e6
nu_hz = (f_rf_mhz - f_ref) * 1e6
f_ref_hz = f_ref * 1e6
chirp = 2.0 * np.pi * DISP_D_HZ * dm * nu_hz ** 2 / (f_ref_hz ** 2 * (f_ref_hz + nu_hz))
H_neg = np.exp(-1j * chirp).astype(np.complex64)
H_pos = np.exp(+1j * chirp).astype(np.complex64)

L_start = int(np.ceil((tau_disp(f_ref, dm) - tau_disp(f_hi, dm)) * fs))
L_end = int(np.ceil((tau_disp(f_lo, dm) - tau_disp(f_ref, dm)) * fs))
hop = nfft - L_start - L_end


def run(H, n_blocks=3):
    fh = open(DADA, 'rb'); fh.seek(4096 + int(10.0 * fs))
    parts = []
    for b in range(n_blocks):
        raw = fh.read(nfft)
        if len(raw) < nfft:
            break
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        R = sfft.rfft(x, workers=-1)
        Z = np.zeros(nfft, dtype=np.complex64)
        Z[0] = R[0]; Z[1:nfft // 2] = 2.0 * R[1:nfft // 2]; Z[nfft // 2] = R[nfft // 2]
        Z[:nfft // 2 + 1] *= H
        z = sfft.ifft(Z, workers=-1)
        p = (z.real ** 2 + z.imag ** 2).astype(np.float32)
        pv = p[L_start:nfft - L_end]
        nv = pv.size - pv.size % 500
        parts.append(pv[:nv].reshape(-1, 500).mean(axis=1))
    fh.close()
    s = np.concatenate(parts).astype(np.float64)
    ph = ((10.0 + np.arange(len(s)) / 2e6) * feff) % 1.0
    prof = np.bincount((ph * nb).astype(np.int64), weights=s, minlength=nb)
    cnt = np.bincount((ph * nb).astype(np.int64), minlength=nb)
    prof /= np.maximum(cnt, 1)
    m, sd = prof.mean(), prof.std()
    pk = int(np.argmax(prof))
    half = (prof[pk] + m) / 2.0
    return (prof[pk] - m) / sd, (prof > half).sum(), pk


for name, H in [('NEG', H_neg), ('POS', H_pos)]:
    t0 = time.time()
    snr, w, pk = run(H)
    print('%s : S/N %.2f  width %d bins  peak %d  (%.1fs)' % (name, snr, w, pk, time.time() - t0))
