# -*- coding: utf-8 -*-
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
d = np.load('out_fold2d_t10.npz')
rel, img = d['rel'], d['img']
nb2, nrow = int(d['nb']), int(d['nrow'])
phi_m, phi_i = 0.2363, 0.7656
bm, bi = int(phi_m * nb2), int(phi_i * nb2)
# 行噪声尺度
row_med = np.median(rel, axis=1)
row_mad = 1.4826 * np.median(np.abs(rel - row_med[:, None]), axis=1)
print('row noise mad 中位=%.2f σ' % np.median(row_mad))
# 主/间 相位 ±3bin 窗口内的行均值(相对行σ)
w = 3
main_col = rel[:, bm - w:bm + w + 1].mean(axis=1)
ip_col = rel[:, bi - w:bi + w + 1].mean(axis=1)
print('主相位窗口行平均(σ/行): 中位=%.2f  最大=%.2f  >2σ的行数: %d/%d' % (
    np.median(main_col), main_col.max(), (main_col > 2).sum(), nrow))
print('间相位窗口行平均(σ/行): 中位=%.2f  最大=%.2f  >2σ的行数: %d/%d' % (
    np.median(ip_col), ip_col.max(), (ip_col > 2).sum(), nrow))
# 整幅最大
print('整幅 rel max=%.1f (99.9分位 %.1f)' % (rel.max(), np.percentile(rel, 99.9)))
