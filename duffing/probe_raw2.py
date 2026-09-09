# -*- coding: utf-8 -*-
"""探测15: 细看 raw 中巨脉冲附近的形态 (2μs 分辨率, 两个候选时间)"""
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DADA = r'D:\VBshare\fast_1937+21\bcmin1937+21.dada'
fs = 1.0e9

def dump(t0, t1, label):
    i0, i1 = int(t0 * fs), int(t1 * fs)
    fh = open(DADA, 'rb')
    fh.seek(4096 + i0)
    raw = np.frombuffer(fh.read(i1 - i0), dtype=np.uint8).astype(np.float32) - 128.0
    fh.close()
    w = int(2e-6 * fs)  # 2μs
    nw = raw.size // w
    pw = raw[:nw * w].reshape(nw, w).mean(axis=1)
    pw -= pw.mean()
    base = i0 / fs
    ipk = int(np.argmax(pw))
    print('[%s] 峰值 @ %+.6f ms (相对窗起点), val=%.1f, 全域中值附近σ=%.1f' % (
        label, (base + ipk * 2e-6 - t0) * 1e3, pw[ipk],
        1.4826 * np.median(np.abs(pw - np.median(pw)))))
    # 打印峰 ±1ms 的 2μs 功率序列 (500点, 用压缩方式: 每0.02ms的max)
    a = max(0, ipk - 500); b = min(nw, ipk + 500)
    seg = pw[a:b]
    n2 = seg.size // 10
    if n2 > 0:
        mx = seg[:n2 * 10].reshape(n2, 10).max(axis=1)
        tms = (base + (a + np.arange(n2) * 10) * 2e-6 - t0) * 1e3
        for k in range(n2):
            if mx[k] > 0.15 * pw[ipk]:
                print('    t=%.3fms rel | p=%.0f' % (tms[k], mx[k]))

dump(16.9300, 16.9310, 'giant 预测 16.9302305 附近 ±0.5ms')
dump(16.9198, 16.9202, 'raw 强峰 16.919985 附近')
