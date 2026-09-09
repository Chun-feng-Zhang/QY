# -*- coding: utf-8 -*-
"""探测14: raw .dada 中 t≈16.93023s 巨脉冲的真实色散跨度(ms级 vs 百ms级)"""
import sys, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DADA = r'D:\VBshare\fast_1937+21\bcmin1937+21.dada'
fs = 1.0e9
t_giant = 16.9302305   # 消色散后巨脉冲时刻 ≈ raw 中 t1500 到达时刻
center = int(t_giant * fs)

fh = open(DADA, 'rb')
# 读 center±300ms 的一段 (若色散 164ms 则跨度在此范围内)
span = int(0.35 * fs)   # ±350 ms → 0.7s ≈ 700MB
fh.seek(4096 + center - span // 2)
raw = np.frombuffer(fh.read(span), dtype=np.uint8).astype(np.float32) - 128.0
fh.close()
print('read %.1f MB raw' % (raw.size / 1e6))

# 每 5μs 平均功率
win = int(5e-6 * fs)   # 5000 samples
nw = raw.size // win
pw = raw[:nw * win].reshape(nw, win).mean(axis=1)
pw = pw - pw.mean()
# 找功率最高的时间位置
i_peak = int(np.argmax(pw))
t_peak = t_giant - 0.35 + i_peak * 5e-6
print('raw 功率峰: t=%.6f s (相对 giant 预测 %.6f ms)' % (t_peak, (t_peak - t_giant) * 1e3))
print('5μs-bin 功率峰/中值 = %.2f' % (pw.max() / np.median(pw)))

# 半功率包络宽度: 以峰为中心找连片高于 30% 峰值的连续区间(在 ±100ms 内)
base = int((t_giant - 0.35) // 5e-6)
ipk = i_peak
thr = 0.30 * pw[ipk]
# 限制搜索 ±150ms
lo = max(0, ipk - 30000); hi = min(nw - 1, ipk + 30000)
a = ipk
while a > lo and pw[a - 1] >= thr: a -= 1
b = ipk
while b < hi and pw[b + 1] >= thr: b += 1
print('30%%峰值包络宽度 = %.3f ms (bin=5μs, %d bins)' % ((b - a) * 5e-3, b - a + 1))

# 更长尺度: 0.5s 窗口内 vs 峰值
# 直接打印峰附近 ±30ms 的功率(每0.1ms) 看色散扫掠形状
win2 = int(0.1e-3 * fs)
for t0 in (t_peak - 0.030, t_peak - 0.003, t_peak, t_peak + 0.003, t_peak + 0.030):
    i0 = int((t0 - (t_giant - 0.35)) // win2)
    if 0 <= i0 < nw // 20:
        pass
