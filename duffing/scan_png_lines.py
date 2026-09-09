# -*- coding: utf-8 -*-
"""分析 PNG 中的红/橙标注线像素"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
try:
    import matplotlib.image as mpimg
except Exception as e:
    print('no mpimg', e); sys.exit(1)

def count_lines(fn):
    img = mpimg.imread(fn)
    h, w, c = img.shape
    rgb = img[:, :, :3]
    # red-ish: R高 G低 B低
    red = (rgb[:, :, 0] > 0.6) & (rgb[:, :, 1] < 0.4) & (rgb[:, :, 2] < 0.4)
    # orange-ish: R高 G中 B低
    orange = (rgb[:, :, 0] > 0.7) & (rgb[:, :, 1] > 0.3) & (rgb[:, :, 1] < 0.7) & (rgb[:, :, 2] < 0.3)
    cols_r = np.where(red.sum(axis=0) > 30)[0]
    cols_o = np.where(orange.sum(axis=0) > 30)[0]
    # 聚类为竖线
    def cluster(cols):
        if len(cols) == 0: return []
        groups = []
        start = cols[0]; prev = cols[0]
        for x in cols[1:]:
            if x - prev > 3:
                groups.append((start + prev) // 2); start = x
            prev = x
        groups.append((start + prev) // 2)
        return groups
    print(fn)
    print('  red lines at cols:', cluster(cols_r.tolist()))
    print('  orange lines at cols:', cluster(cols_o.tolist()))

import numpy as np
for fn in ['out_duffing_mark_20ms_seg1334.png', 'out_duffing_20ms_seg1334.png',
           'out_duffing_mark_20ms_seg0846.png']:
    try:
        count_lines(fn)
    except Exception as e:
        print(fn, 'ERR', e)
