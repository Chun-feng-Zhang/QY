# -*- coding: utf-8 -*-
"""debug_sign_impulse.py — 冲激测试, 用能量集中度确定消色散物理符号"""
import numpy as np
import scipy.fft as sfft

DISP_D_HZ = 4.148808e15  # Hz^2 cm^3 pc^-1 s

nfft = 1 << 18
fs = 1.0e9
f_lo, f_hi, f_ref = 1000.0, 1500.0, 1250.0
dm = 0.05

# 冲激
sig = np.zeros(nfft, dtype=np.float32)
sig[nfft // 4] = 1.0


def analytic_spectrum(x):
    R = sfft.rfft(x, workers=-1)
    Z = np.zeros(nfft, dtype=np.complex64)
    Z[0] = R[0]
    Z[1:nfft // 2] = 2.0 * R[1:nfft // 2]
    Z[nfft // 2] = R[nfft // 2]
    return Z


def apply_filter(x, H):
    Z = analytic_spectrum(x)
    Z[:nfft // 2 + 1] *= H
    z = sfft.ifft(Z, workers=-1)
    return (z.real ** 2 + z.imag ** 2).astype(np.float32)


def conc(p, top=100):
    s = np.sort(p)[::-1]
    return s[:top].sum() / s.sum()


k = np.arange(nfft // 2 + 1, dtype=np.float64)
f_rf_mhz = f_lo + k * (fs / nfft) / 1e6
f_ref_hz = f_ref * 1e6
nu_hz = (f_rf_mhz - f_ref) * 1e6

# 物理 ISM: H_ism = exp(+i 2pi D_Hz DM / f_Hz), 群延迟 = +D DM/f^2
phase_ism = 2.0 * np.pi * DISP_D_HZ * dm / (f_rf_mhz * 1e6)
H_ism = np.exp(1j * phase_ism).astype(np.complex64)

# 消色散 chirp 两种符号
chirp = 2.0 * np.pi * DISP_D_HZ * dm * nu_hz ** 2 / (f_ref_hz ** 2 * (f_ref_hz + nu_hz))
H_neg = np.exp(-1j * chirp).astype(np.complex64)
H_pos = np.exp(+1j * chirp).astype(np.complex64)

# 分散
Z = analytic_spectrum(sig)
Z[:nfft // 2 + 1] *= H_ism
sig_disp = sfft.ifft(Z, workers=-1).real.astype(np.float32)

p_orig = sig ** 2
p_disp = apply_filter(sig, H_ism)
p_neg = apply_filter(sig_disp, H_neg)
p_pos = apply_filter(sig_disp, H_pos)

print('energy concentration (top 100 of %d samples):' % nfft)
print('  original impulse       : %.4f' % conc(p_orig))
print('  after ISM dispersion   : %.4f' % conc(p_disp))
print('  disperse->dedisp NEG   : %.4f' % conc(p_neg))
print('  disperse->dedisp POS   : %.4f' % conc(p_pos))
