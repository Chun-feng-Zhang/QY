# -*- coding: utf-8 -*-
"""debug_sign_final.py — 与真实管线完全一致(f_ref=f_hi)的冲激测试"""
import numpy as np
import scipy.fft as sfft

DISP_D_HZ = 4.148808e15

nfft = 1 << 18
fs = 1.0e9
f_lo, f_hi, f_ref = 1000.0, 1500.0, 1500.0  # f_ref = f_hi, 与真实管线一致
dm = 0.05

sig = np.zeros(nfft, dtype=np.float32)
sig[nfft // 4] = 1.0


def analytic(x):
    R = sfft.rfft(x, workers=-1)
    Z = np.zeros(nfft, dtype=np.complex64)
    Z[0] = R[0]
    Z[1:nfft // 2] = 2.0 * R[1:nfft // 2]
    Z[nfft // 2] = R[nfft // 2]
    return Z


def apply(x, H):
    Z = analytic(x)
    Z[:nfft // 2 + 1] *= H
    z = sfft.ifft(Z, workers=-1)
    return (z.real ** 2 + z.imag ** 2).astype(np.float32)


k = np.arange(nfft // 2 + 1, dtype=np.float64)
f_rf_mhz = f_lo + k * (fs / nfft) / 1e6
f_ref_hz = f_ref * 1e6
nu_hz = (f_rf_mhz - f_ref) * 1e6

# 物理 ISM: 群延迟 = +D DM/f^2 → φ = +2π D_Hz DM/f_Hz
phase_ism = 2.0 * np.pi * DISP_D_HZ * dm / (f_rf_mhz * 1e6)
H_ism = np.exp(1j * phase_ism).astype(np.complex64)

chirp = 2.0 * np.pi * DISP_D_HZ * dm * nu_hz ** 2 / (f_ref_hz ** 2 * (f_ref_hz + nu_hz))
H_neg = np.exp(-1j * chirp).astype(np.complex64)
H_pos = np.exp(+1j * chirp).astype(np.complex64)


def conc(p, top=100):
    s = np.sort(p)[::-1]
    return s[:top].sum() / s.sum()


# 分散 (把冲激经 ISM 色散)
sig_disp = apply(sig, H_ism)
# 消色散
p_neg = apply(sig_disp, H_neg)
p_pos = apply(sig_disp, H_pos)

print('f_ref = %.0f MHz (与真实管线一致)' % f_ref)
print('original impulse conc : %.4f' % conc(sig ** 2))
print('after ISM dispersion  : %.4f' % conc(sig_disp))
print('dedisp NEG (-i chirp) : %.4f' % conc(p_neg))
print('dedisp POS (+i chirp) : %.4f' % conc(p_pos))
print('>>> 恢复率高者为正确消色散符号')
