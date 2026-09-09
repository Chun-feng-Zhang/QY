# -*- coding: utf-8 -*-
"""debug_sign_synthetic2.py — 物理正确 ISM 色散, 确定消色散符号"""
import numpy as np
import scipy.fft as sfft

DISP_D = 4.148808e3
DISP_D_HZ = DISP_D * 1.0e12

nfft = 1 << 18
fs = 1.0e9
f_lo, f_hi, f_ref = 1000.0, 1500.0, 1250.0
dm = 0.05

# 窄高斯脉冲
t = np.arange(nfft) / fs
sig = np.exp(-0.5 * ((t - nfft * 0.25 / fs) / 2e-6) ** 2).astype(np.float32)
sig += 0.02 * np.random.default_rng(0).standard_normal(nfft).astype(np.float32)


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


k = np.arange(nfft // 2 + 1, dtype=np.float64)
f_rf_mhz = f_lo + k * (fs / nfft) / 1e6
nu_mhz = f_rf_mhz - f_ref
f_ref_hz = f_ref * 1e6
nu_hz = nu_mhz * 1e6

# 物理 ISM 色散 (群延迟 = +D DM/f^2): H_ism = exp(+i 2pi D_Hz DM / f_Hz)
phase_ism = 2.0 * np.pi * DISP_D_HZ * dm / (f_rf_mhz * 1e6)
H_ism = np.exp(1j * phase_ism).astype(np.complex64)

# 精确消色散 (逆): H_dd_exact = exp(-i 2pi D_Hz DM / f_Hz)
H_dd_exact = np.exp(-1j * phase_ism).astype(np.complex64)

# chirp 近似两种符号
chirp = 2.0 * np.pi * DISP_D_HZ * dm * nu_hz ** 2 / (f_ref_hz ** 2 * (f_ref_hz + nu_hz))
H_neg = np.exp(-1j * chirp).astype(np.complex64)
H_pos = np.exp(+1j * chirp).astype(np.complex64)


def width(p):
    i = int(np.argmax(p))
    half = (p[i] + p.min()) / 2.0
    return int((p > half).sum())


p0 = sig ** 2
p_disp = apply_filter(sig, H_ism)
# 色散后的复电压
Z = analytic_spectrum(sig)
Z[:nfft // 2 + 1] *= H_ism
sig_disp = sfft.ifft(Z, workers=-1).real.astype(np.float32)

print('original width          :', width(p0))
print('after ISM dispersion    :', width(p_disp))
print('disp->dedisp exact(neg) :', width(apply_filter(sig_disp, H_dd_exact)))
print('disp->dedisp chirp NEG  :', width(apply_filter(sig_disp, H_neg)))
print('disp->dedisp chirp POS  :', width(apply_filter(sig_disp, H_pos)))
