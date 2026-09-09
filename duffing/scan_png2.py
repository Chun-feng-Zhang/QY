# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np, matplotlib.image as mpimg

def lines(fn):
    img = mpimg.imread(fn); rgb = img[:,:,:3]
    red = (rgb[:,:,0]>0.6)&(rgb[:,:,1]<0.4)&(rgb[:,:,2]<0.4)
    orange = (rgb[:,:,0]>0.7)&(rgb[:,:,1]>0.3)&(rgb[:,:,1]<0.7)&(rgb[:,:,2]<0.3)
    def colgroups(mask):
        cols = np.where(mask.sum(axis=0) > 40)[0]
        if len(cols)==0: return []
        g=[]; s=cols[0]; p=cols[0]
        for x in cols[1:]:
            if x-p>3: g.append((s+p)//2); s=x
            p=x
        g.append((s+p)//2)
        return g
    print(fn)
    print(' red cols:', colgroups(red))
    print(' orange cols:', colgroups(orange))
    # 两面板各自的行范围 (红/橙像素出现在哪些行)
    rr = np.where(red.any(axis=1))[0]
    oo = np.where(orange.any(axis=1))[0]
    print(' red rows: %d-%d (%d)   orange rows: %d-%d (%d)' % (
        rr.min(), rr.max(), len(rr), oo.min(), oo.max(), len(oo)))

for i in (0, 10, 50):
    try:
        lines('out_t10_mark_20ms_seg%04d.png' % i)
    except Exception as e:
        print(i, 'ERR', e)
