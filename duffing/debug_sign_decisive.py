# -*- coding: utf-8 -*-
"""debug_sign_decisive.py — 真实数据单块, 双符号折叠 S/N 对照"""
import numpy as np, scipy.fft as sfft, time

DISP_D_HZ = 4.148808e15
DADA = r'D:\VBshare\fast_1937+21\bcmin1937+21.dada'
feff = 641.886226994
fs = 1.0e9
f_lo, f_hi, f_ref = 1000.0, 1500.0, 1500.0
dm = 71.0237
nfft = 1 << 28  # 268.4 ms, 精确消色散足够

k = np.arange(nfft // 2 + 1, dtype=np.float64)
f_hz = (f_lo + k * (fs / nfft) / 1e6) * 1e6
f_ref_hz = f_ref * 1e6

# 精确消色散相位 (正/负)
exact = 2.0 * np.pi * DISP_D_HZ * dm * (1.0 / f_hz + f_hz / f_ref_hz ** 2 - 2.0 / f_ref_hz)
H_pos = np.exp(+1j * exact).astype(np.complex64)
H_neg = np.exp(-1j * exact).astype(np.complex64)

# 读取 3 块数据拼接 (约 0.3s)
fh = open(DADA, 'rb'); fh.seek(4096 + int(10.0 * fs))


def run(H, n_blocks=3):
    fh.seek(4096 + int(10.0 * fs))
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
        # 无重叠丢弃(单块独立), 仅去边缘
        pv = p[:nfft - 163702053]
        nv = pv.size - pv.size % 500
        parts.append(pv[:nv].reshape(-1, 500).mean(axis=1))
    s = np.concatenate(parts).astype(np.float64)
    nb = 256
    ph = ((10.0 + np.arange(len(s)) / 2e6) * feff) % 1.0
    prof = np.bincount((ph * nb).astype(np.int64), weights=s, minlength=nb)
    cnt = np.bincount((ph * nb).astype(np.int64), minlength=nb)
    prof /= np.maximum(cnt, 1)
    m, sd = prof.mean(), prof.std()
    pk = int(np.argmax(prof))
    half = (prof[pk] + m) / 2.0
    return (prof[pk] - m) / sd, (prof > half).sum(), pk


for name, H in [('POS(+i)', H_pos), ('NEG(-i)', H_neg)]:
    t0 = time.time()
    snr, w, pk = run(H)
    print('%s : S/N %.2f  width %d bins  peak %d  (%.1fs)' % (name, snr, w, pk, time.time() - t0))
fh.close()
print('>>> 折叠 S/N 更高者为正确消色散符号')
