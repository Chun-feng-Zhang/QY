# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np, matplotlib.image as mpimg

def lines(fn, col_min=30):
    img = mpimg.imread(fn); rgb = img[:,:,:3]
    red = (rgb[:,:,0]>0.6)&(rgb[:,:,1]<0.45)&(rgb[:,:,2]<0.45)
    orange = (rgb[:,:,0]>0.7)&(rgb[:,:,1]>0.3)&(rgb[:,:,1]<0.7)&(rgb[:,:,2]<0.3)
    def colgroups(mask):
        cols = np.where(mask.sum(axis=0) > col_min)[0]
        if len(cols)==0: return []
        g=[]; s=cols[0]; p=cols[0]
        for x in cols[1:]:
            if x-p>3: g.append((s+p)//2); s=x
            p=x
        g.append((s+p)//2)
        return g
    print(fn)
    print(' red cols (%d):' % len(colgroups(red)), colgroups(red))
    print(' orange cols (%d):' % len(colgroups(orange)), colgroups(orange))

for fn in ['out_t10_mark_20ms_seg0005.png', 'out_t10_fold_2d.png']:
    try:
        lines(fn)
    except Exception as e:
        print(fn, 'ERR', e)
