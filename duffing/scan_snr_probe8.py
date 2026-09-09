# -*- coding: utf-8 -*-
"""诊断8: 巨脉冲周围原始功率值 + 噪声水平, 理解真实形态"""
import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
fs = 2.0e6
s = np.load('out_coh_full.npy', mmap_mode='r')

# 巨脉冲精确位置 (diag5: t=16.9302s; diag6: abs idx 33860461)
idx = 33860461
seg = np.asarray(s[idx-20:idx+30], dtype=np.float64)
print('raw values around giant (idx %d):' % idx)
print(' '.join('%d:%.5g' % (idx-20+i, v) for i, v in enumerate(seg)))

# 该 100ms 窗口的噪声统计(不含爆发)
i0 = int(16.90*fs)
blk = np.asarray(s[i0:i0+int(0.1*fs)], dtype=np.float64)
m, sd = blk.mean(), blk.std()
print('\n100ms block mean=%.5g std=%.5g  -> 占空比检查' % (m, sd))
print('giant sample raw = %.5g = %.1f x mean' % (float(s[idx]), float(s[idx])/m))

# z 是怎么算的: (raw-mean)/std
z_giant = (s[idx]-m)/sd
print('z(recompute)=(raw-mean)/std = %.1f' % z_giant)
# 检查该块中位
print('block median=%.5g  99.9百分位=%.5g' % (np.median(blk), np.percentile(blk, 99.9)))
print('max in block=%.5g at %s' % (blk.max(), (i0+int(np.argmax(blk)))/fs))
