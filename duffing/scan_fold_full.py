# -*- coding: utf-8 -*-
"""诊断11: 绝对时间折叠全34s → 可靠的 phi_main_abs / phi_ip_abs"""
import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
fs = 2.0e6
f_eff = 641.886226994
s = np.load('out_coh_full.npy', mmap_mode='r')
nb = 4096
prof = np.zeros(nb); cnt = np.zeros(nb)
chunk = int(2.0*fs)
t_off = 0.0
for i0 in range(0, s.size, chunk):
    seg = np.asarray(s[i0:i0+chunk], dtype=np.float64)
    t = t_off + np.arange(len(seg))/fs
    ph = (t*f_eff) % 1.0
    b = (ph*nb).astype(np.int64)
    prof += np.bincount(b, weights=seg, minlength=nb)
    cnt += np.bincount(b, minlength=nb)
    t_off += len(seg)/fs
prof = prof/np.maximum(cnt, 1)
pk = int(np.argmax(prof))
# ip 候选在 pk+0.5283 附近 ±0.06
sep = int(round(0.5283*nb))
lo = (pk + sep - int(0.06*nb)) % nb
hi = (pk + sep + int(0.06*nb)) % nb
if lo < hi:
    ip = lo + int(np.argmax(prof[lo:hi]))
else:
    roll = np.roll(prof, -lo)
    ip = (lo + int(np.argmax(roll[:hi-lo+nb]))) % nb
print('phi_main_abs = %.4f (%d/%d)   phi_ip_abs = %.4f (%d)   sep=%.4f' % (
    pk/nb, pk, nb, ip/nb, ip, (ip-pk)/nb % 1.0))
# 主峰宽度 (半高)
half = (prof.max()+np.median(prof))/2
above = np.where(prof > half)[0]
print('main peak profile max=%.2f  med=%.2f' % (prof.max(), np.median(prof)))
# 相邻 bin 统计
pm = prof[pk-3:pk+4]
print('bins around main:', ['%.1f' % x for x in pm])
# 保存
np.save('fold_full_profile.npy', prof)
print('saved fold_full_profile.npy (nb=%d)' % nb)
