# -*- coding: utf-8 -*-
"""debug_raw_vs_dedisp.py — 对比原始(未消色散)数据 vs 相干消色散 的折叠 S/N"""
import numpy as np

DADA = r'D:\VBshare\fast_1937+21\bcmin1937+21.dada'
feff = 641.886226994
nb = 1024
fs = 1.0e9


def fold_power(p, t0, f_out):
    N = len(p)
    ph = ((t0 + np.arange(N) / f_out) * feff) % 1.0
    prof = np.bincount((ph * nb).astype(np.int64), weights=p, minlength=nb)
    cnt = np.bincount((ph * nb).astype(np.int64), minlength=nb)
    return prof / np.maximum(cnt, 1)


def snr(prof):
    m, sd = prof.mean(), prof.std()
    pk = int(np.argmax(prof))
    return (prof[pk] - m) / sd, pk


# 1) 原始数据 (2s): 平方检波(去直流) -> 降采样 -> 折叠
t0 = 10.0
n_raw = int(2.0 * fs)
fh = open(DADA, 'rb')
fh.seek(4096 + int(t0 * fs))
x = np.frombuffer(fh.read(n_raw), dtype=np.uint8).astype(np.float32) - 128.0
fh.close()
p_raw = x ** 2
dec = 500
nv = p_raw.size - p_raw.size % dec
p_raw_ds = p_raw[:nv].reshape(-1, dec).mean(axis=1).astype(np.float64)
prof_raw = fold_power(p_raw_ds, t0, 2e6)
print('RAW (undedispersed)  : peak bin %d  S/N %.2f' % snr(prof_raw))

# 2) 相干消色散
new = np.load('out_coh_t10_2s.npy').astype(np.float64)
prof_new = fold_power(new, t0, 2e6)
print('COHERENT dedispersed : peak bin %d  S/N %.2f' % snr(prof_new))

# 3) 旧通道化
old = np.load(r'C:\Users\29094\Documents\ChatGPT\QY pulsar\diffuse_oscilator_0824\out_b1937_fold_fix_series.npy',
              mmap_mode='r')
i0 = int(10 * 2e6); i1 = int(12 * 2e6)
prof_old = fold_power(np.asarray(old[i0:i1]).astype(np.float64), 10.0, 2e6)
print('OLD channelized      : peak bin %d  S/N %.2f' % snr(prof_old))
