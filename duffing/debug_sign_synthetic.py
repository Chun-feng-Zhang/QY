# -*- coding: utf-8 -*-
"""debug_sign_synthetic.py — 用物理正确的 ISM 色散合成测试, 确定消色散相位符号"""
import numpy as np
import scipy.fft as sfft

DISP_D = 4.148808e3
DISP_D_HZ = DISP_D * 1.0e12

nfft = 1 << 18
fs = 1.0e9
f_lo, f_hi, f_ref = 1000.0, 1500.0, 1250.0
dm = 0.05  # 小 DM, smear << 块长

# 窄高斯脉冲 (实信号)
t = np.arange(nfft) / fs
sig = np.exp(-0.5 * ((t - nfft * 0.25 / fs) / 2e-6) ** 2).astype(np.float32)
sig += 0.05 * np.random.default_rng(0).standard_normal(nfft).astype(np.float32)


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
# 物理正确 ISM 色散: H_ism(f)=exp(+i 2pi D_Hz DM / f_Hz)
phase_ism = 2.0 * np.pi * DISP_D_HZ * dm / (f_rf_mhz * 1e6)
H_ism = np.exp(1j * phase_ism).astype(np.complex64)
# 消色散 chirp 两种符号
chirp = 2.0 * np.pi * DISP_D_HZ * dm * nu_hz ** 2 / (f_ref_hz ** 2 * (f_ref_hz + nu_hz))
H_dd_neg = np.exp(-1j * chirp).astype(np.complex64)
H_dd_pos = np.exp(+1j * chirp).astype(np.complex64)


def width(p):
    i = int(np.argmax(p))
    half = p[i] / 2.0
    return int((p > half).sum())


p0 = sig ** 2
p_disp = apply_filter(sig, H_ism)          # 色散后的功率 (应比原脉冲宽)
p_rec_neg = apply_filter(sig, H_dd_neg)     # 只消色散(对未色散信号) — 观察效果
p_rec_pos = apply_filter(sig, H_dd_pos)
# 关键: 色散后再消色散
sig_disp = apply_filter_amp = None
Z = analytic_spectrum(sig)
Z[:nfft // 2 + 1] *= H_ism
z_disp = sfft.ifft(Z, workers=-1)
sig_disp = z_disp.real.astype(np.float32)
p_neg = apply_filter(sig_disp, H_dd_neg)
p_pos = apply_filter(sig_disp, H_dd_pos)

print('original pulse width (bins):', width(p0))
print('after ISM dispersion      :', width(p_disp))
print('disperse->dedisp(-)       :', width(p_neg))
print('disperse->dedisp(+)       :', width(p_pos))
