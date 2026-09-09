# -*- coding: utf-8 -*-
"""debug_verify_dedisp.py — 用正确 bin 数验证相干消色散是否真正锐化脉冲"""
import numpy as np

DADA = r'D:\VBshare\fast_1937+21\bcmin1937+21.dada'
feff = 641.886226994
fs = 1.0e9
t0 = 10.0
dur = 2.0


def fold_power(p, t0, f_out, nb):
    N = len(p)
    ph = ((t0 + np.arange(N) / f_out) * feff) % 1.0
    prof = np.bincount((ph * nb).astype(np.int64), weights=p, minlength=nb)
    cnt = np.bincount((ph * nb).astype(np.int64), minlength=nb)
    return prof / np.maximum(cnt, 1)


def peak_snr_width(prof):
    m, sd = prof.mean(), prof.std()
    pk = int(np.argmax(prof))
    snr = (prof[pk] - m) / sd
    # 半高全宽 (bins)
    half = (prof[pk] + m) / 2.0
    above = prof > half
    return snr, pk, int(above.sum())


# 原始数据 (未消色散)
n_raw = int(dur * fs)
fh = open(DADA, 'rb')
fh.seek(4096 + int(t0 * fs))
x = np.frombuffer(fh.read(n_raw), dtype=np.uint8).astype(np.float32) - 128.0
fh.close()
p_raw = x ** 2
dec = 500
nv = p_raw.size - p_raw.size % dec
p_raw_ds = p_raw[:nv].reshape(-1, dec).mean(axis=1).astype(np.float64)

# 相干消色散
new = np.load('out_coh_t10_2s.npy').astype(np.float64)
# 旧通道化
old = np.load(r'C:\Users\29094\Documents\ChatGPT\QY pulsar\diffuse_oscilator_0824\out_b1937_fold_fix_series.npy',
              mmap_mode='r')
i0 = int(10 * 2e6); i1 = int(12 * 2e6)
old2 = np.asarray(old[i0:i1]).astype(np.float64)

for nb in (128, 256):
    print('=== nb =', nb, '===')
    for name, p in [('RAW   ', p_raw_ds), ('COHER ', new), ('OLD   ', old2)]:
        prof = fold_power(p, t0, 2e6, nb)
        snr, pk, w = peak_snr_width(prof)
        print('  %s : peak bin %3d  S/N %.1f  width %d bins' % (name, pk, snr, w))
