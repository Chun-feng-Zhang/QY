# -*- coding: utf-8 -*-
"""debug_exact.py — 精确消色散相位 冲激验证"""
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

# 物理 ISM: φ = +2π D_Hz DM/f  (群延迟 +D DM/f^2)
H_ism = np.exp(1j * (2.0 * np.pi * DISP_D_HZ * dm / f_hz)).astype(np.complex64)
# 精确消色散: φ = 2π D_Hz DM(f/f_ref^2 + 1/f - 2/f_ref)
H_dd = np.exp(1j * (2.0 * np.pi * DISP_D_HZ * dm *
                    (f_hz / f_ref_hz ** 2 + 1.0 / f_hz - 2.0 / f_ref_hz))).astype(np.complex64)


def conc(p, top=100):
    s = np.sort(p)[::-1]
    return s[:top].sum() / s.sum()


p_disp = apply(sig, H_ism)
p_rec = apply(sig, H_dd)  # 直接对未色散冲激应用 — 若正确应色散(变宽)
# 关键: 色散后 → 消色散
sig_disp = apply(sig, H_ism)
Z = analytic(np.sign(sig_disp) * np.sqrt(sig_disp))  # 用包络做回实信号(近似)
p_rec2 = apply(np.sign(sig_disp) * np.sqrt(sig_disp).astype(np.float32), H_dd)

print('exact phase test (f_ref=%.0f)' % f_ref)
print('original conc        : %.4f' % conc(sig ** 2))
print('after ISM            : %.4f' % conc(p_disp))
print('dedisp on dispersed  : %.4f' % conc(p_rec2))
