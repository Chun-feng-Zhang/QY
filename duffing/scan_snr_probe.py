# -*- coding: utf-8 -*-
"""扫描探测: 用 'on-pulse 幅度和 / 前沿等宽基线幅度和' 定义, 看真实输入数据里 SNR>5 的峰有多少"""
import numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SER = 'out_coh_full.npy'
META = 'out_coh_full.json'
meta = json.load(open(META, encoding='utf-8'))
fs = float(meta['f_out'])

s = np.load(SER, mmap_mode='r')
print('series', SER, s.size, 'samples, fs=%.1f MHz' % (fs/1e6))


def snr_at(z, p, W):
    """on窗口 [p-W/2,p+W/2) 幅度和 / 前沿基线窗口(等宽, 紧邻on窗口前)幅度和"""
    half = W // 2
    a = p - half
    if a - W < 0 or a + W + W > len(z):
        return np.nan
    on = np.abs(z[a:a + W]).sum()
    base = np.abs(z[a - W:a]).sum()
    return on / (base + 1e-30)


def scan_window(t0, Ws=(10, 20, 40)):
    i0 = int(t0 * fs)
    i1 = i0 + int(0.020 * fs)
    seg = np.asarray(s[i0:i1], dtype=np.float64)
    z = (seg - seg.mean()) / (seg.std() + 1e-30)
    az = np.abs(z)
    # 候选: |z| 局部极大 且 z>2.0 (正峰)
    cand = []
    for i in range(1, len(z) - 1):
        if az[i] >= az[i-1] and az[i] > az[i+1] and z[i] > 2.0:
            cand.append(i)
    print('\n[t=%.0f s window 20ms] n_cand(z>2) = %d' % (t0, len(cand)))
    for W in Ws:
        snrs = []
        for p in cand:
            r = snr_at(z, p, W)
            if np.isfinite(r):
                snrs.append((r, p, z[p]))
        snrs.sort(reverse=True)
        over5 = [x for x in snrs if x[0] > 5]
        print('  W=%3d: SNR>5: %2d/%-3d   top: %s' % (
            W, len(over5), len(snrs),
            ' '.join('%.1f@%d(z%.1f)' % (x[0], x[1], x[2]) for x in snrs[:5])))


for t0 in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 33.0):
    scan_window(t0)
