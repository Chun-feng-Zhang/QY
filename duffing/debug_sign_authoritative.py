# -*- coding: utf-8 -*-
"""debug_sign_authoritative.py — 物理符号正确的权威验证"""
import numpy as np
import scipy.fft as sfft

DISP_D_HZ = 4.148808e15
nfft = 1 << 18
fs = 1.0e9
f_lo, f_hi, f_ref = 1000.0, 1500.0, 1500.0
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
f_hz = (f_lo + k * (fs / nfft) / 1e6) * 1e6
f_ref_hz = f_ref * 1e6

# 物理 ISM: H_ism = exp(-i 2π D_Hz DM / f_Hz)  [低频延迟更大]
H_ism = np.exp(-1j * (2.0 * np.pi * DISP_D_HZ * dm / f_hz)).astype(np.complex64)
# 精确消色散(候选): phi = +2π D_Hz DM(1/f + f/f_ref^2 - 2/f_ref)
H_dd = np.exp(+1j * (2.0 * np.pi * DISP_D_HZ * dm *
                     (1.0 / f_hz + f_hz / f_ref_hz ** 2 - 2.0 / f_ref_hz))).astype(np.complex64)


def conc(p, top=100):
    s = np.sort(p)[::-1]
    return s[:top].sum() / s.sum()


p_disp = apply(sig, H_ism)
# 色散后复电压 → 再消色散
Z = analytic(sig)
Z[:nfft // 2 + 1] *= H_ism
sig_disp = sfft.ifft(Z, workers=-1).real.astype(np.float32)
p_rec = apply(sig_disp, H_dd)

print('物理正确 ISM = exp(-i 2π D DM/f)')
print('original conc       : %.4f' % conc(sig ** 2))
print('after ISM           : %.4f' % conc(p_disp))
print('dedisp (POS exact)  : %.4f' % conc(p_rec))
print('>>> 若 dedisp 恢复≈original, 则正号精确相位正确')
