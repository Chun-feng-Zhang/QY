# -*- coding: utf-8 -*-
"""
duffing_mark_20ms.py — 20ms 单脉冲标注 + 强耦合 Duffing 增强对比
================================================================
标注规范 (用户定义):
  1. 在【原始输入波形】(相干消色散功率序列, 未经 Duffing 增强) 中确定首个主脉冲:
     - 候选: 局部极大且幅度 > 噪声底
     - 用区域SNR取最高者, 以此脉冲的相位锚定主脉冲网格 (t_k = (n + phi_main)/f_eff)
  2. SNR 定义 (on-pulse 幅度和 / on-pulse 前缘等宽基线幅度和):
       on  = 从峰值向下扩展的连续区域 (含峰值的脉冲区)
       base= on 区域前沿之前的等宽样本
       SNR = sum(on) / sum(base)
  3. 只标注 SNR > 5 的峰 (主脉冲红虚线, 中间脉冲橙点线)
  4. Duffing 增强后的输出波形上, 主脉冲标注位置与输入波形完全相同
  5. 每 20 ms 一图 (输入 / Duffing 输出上下两幅)

物理: 强耦合双振子 Duffing v5 (RK4):
  dx1/dt=w*y1;  dy1/dt=w*(-xi*y1+x1-x1^3-k*xi*(x1-x2)-k*(y1-y2)+F*cos(wt)+s(t))
  dx2/dt=w*y2;  dy2/dt=w*(-xi*y2+x2-x2^3-k*xi*(x2-x1)-k*(y2-y1)+F*cos(wt))
  output = w*(y1-y2),  w≈2*pi*f_eff 谐振于脉冲重复频率
"""
import argparse, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from numba import njit

AU_M = 1.495978707e11
C_LIGHT = 2.99792458e8
OMEGA_EARTH = 7.292115e-5
CFG = dict(F=0.8, omega=4070.0, k=3.0, xi=0.08,
           x1_init=0.0, y1_init=0.0, x2_init=0.0, y2_init=0.0)
PULSE_SEP = 0.5283      # 间脉冲滞后主脉冲的相位 (历史折叠测量 541/1024)
SNR_THRESH = 5.0        # 只标注 SNR>5
SEG_MS = 20.0


def parse_par(path):
    par = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            if len(p) >= 2 and p[0] not in par:
                par[p[0]] = p[1]
    return par


def parse_hms(s):
    p = s.split(':')
    return float(p[0]) + float(p[1]) / 60.0 + float(p[2]) / 3600.0


def parse_dms(s):
    sign = -1.0 if s.strip().startswith('-') else 1.0
    p = s.strip().lstrip('+-').split(':')
    return sign * (float(p[0]) + float(p[1]) / 60.0 + float(p[2]) / 3600.0)


def compute_feff(par, utc_start, lon=106.8567, lat=25.6531, h=1110.0):
    """顶心自转频率 / topocentric spin frequency"""
    from astropy.time import Time
    from astropy.coordinates import EarthLocation
    import erfa
    s = utc_start.strip()
    if len(s) > 11 and s[10] == '-':
        s = s[:10] + ' ' + s[11:]
    t0 = Time(s, scale='utc', format='iso')
    mjd0 = t0.mjd
    f0 = float(par['F0'])
    f1 = float(par['F1']) if 'F1' in par else 0.0
    pepoch = float(par.get('PEPOCH', mjd0))
    f0_epoch = f0 + f1 * (mjd0 - pepoch) * 86400.0
    pvh, pvb = erfa.epv00(t0.tt.jd1, t0.tt.jd2)
    v_earth = pvb[1] * AU_M / 86400.0
    loc = EarthLocation.from_geodetic(lon=lon, lat=lat, height=h)
    g = loc.geocentric
    r_site = np.array([float(g[i].to_value('m')) for i in range(3)])
    v_site = np.cross(np.array([0.0, 0.0, OMEGA_EARTH]), r_site)
    ar, dr = np.deg2rad(parse_hms(par['RAJ']) * 15.0), np.deg2rad(parse_dms(par['DECJ']))
    nhat = np.array([np.cos(dr) * np.cos(ar), np.cos(dr) * np.sin(ar), np.sin(dr)])
    v_proj = float(np.dot(v_earth + v_site, nhat))
    return f0_epoch * (1.0 + v_proj / C_LIGHT), f0_epoch, v_proj, mjd0


@njit
def rk4_coupled(signal, dt, F, omega, k, xi, x1, y1, x2, y2):
    n = signal.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        t = i * dt
        st = signal[i]
        h = dt
        h2 = h * 0.5
        c1 = np.cos(omega * t)
        k1x1 = omega * y1
        k1y1 = omega * (-xi * y1 + x1 - x1 * x1 * x1 - k * xi * (x1 - x2)
                        - k * (y1 - y2) + F * c1 + st)
        k1x2 = omega * y2
        k1y2 = omega * (-xi * y2 + x2 - x2 * x2 * x2 - k * xi * (x2 - x1)
                        - k * (y2 - y1) + F * c1)
        t2 = t + h2
        c2 = np.cos(omega * t2)
        x1a, y1a = x1 + h2 * k1x1, y1 + h2 * k1y1
        x2a, y2a = x2 + h2 * k1x2, y2 + h2 * k1y2
        k2x1 = omega * y1a
        k2y1 = omega * (-xi * y1a + x1a - x1a * x1a * x1a - k * xi * (x1a - x2a)
                        - k * (y1a - y2a) + F * c2 + st)
        k2x2 = omega * y2a
        k2y2 = omega * (-xi * y2a + x2a - x2a * x2a * x2a - k * xi * (x2a - x1a)
                        - k * (y2a - y1a) + F * c2)
        x1b, y1b = x1 + h2 * k2x1, y1 + h2 * k2y1
        x2b, y2b = x2 + h2 * k2x2, y2 + h2 * k2y2
        k3x1 = omega * y1b
        k3y1 = omega * (-xi * y1b + x1b - x1b * x1b * x1b - k * xi * (x1b - x2b)
                        - k * (y1b - y2b) + F * c2 + st)
        k3x2 = omega * y2b
        k3y2 = omega * (-xi * y2b + x2b - x2b * x2b * x2b - k * xi * (x2b - x1b)
                        - k * (y2b - y1b) + F * c2)
        t4 = t + h
        c4 = np.cos(omega * t4)
        x1c, y1c = x1 + h * k3x1, y1 + h * k3y1
        x2c, y2c = x2 + h * k3x2, y2 + h * k3y2
        k4x1 = omega * y1c
        k4y1 = omega * (-xi * y1c + x1c - x1c * x1c * x1c - k * xi * (x1c - x2c)
                        - k * (y1c - y2c) + F * c4 + st)
        k4x2 = omega * y2c
        k4y2 = omega * (-xi * y2c + x2c - x2c * x2c * x2c - k * xi * (x2c - x1c)
                        - k * (y2c - y1c) + F * c4)
        x1 += (h / 6.0) * (k1x1 + 2 * k2x1 + 2 * k3x1 + k4x1)
        y1 += (h / 6.0) * (k1y1 + 2 * k2y1 + 2 * k3y1 + k4y1)
        x2 += (h / 6.0) * (k1x2 + 2 * k2x2 + 2 * k3x2 + k4x2)
        y2 += (h / 6.0) * (k1y2 + 2 * k2y2 + 2 * k3y2 + k4y2)
        out[i] = omega * (y1 - y2)
    return out, x1, y1, x2, y2


def pulse_snr(seg, p, m, sigma, gap_tol=2):
    """区域 SNR: on=含峰连续区(容忍窄凹陷), base=on前沿等宽样本.
    阈值: m + max(0.05*(A-m), 5*sigma); sigma=MAD估计噪声σ"""
    A = float(seg[p])
    if A <= m:
        return None
    thr = m + max(0.05 * (A - m), 5.0 * sigma)
    n = len(seg)
    lo = p
    gap = 0
    while lo - 1 >= 0:
        if seg[lo - 1] >= thr:
            lo -= 1
            gap = 0
        elif gap < gap_tol:
            lo -= 1
            gap += 1
        else:
            break
    hi = p
    gap = 0
    while hi + 1 < n:
        if seg[hi + 1] >= thr:
            hi += 1
            gap = 0
        elif gap < gap_tol:
            hi += 1
            gap += 1
        else:
            break
    W = hi - lo + 1
    if lo - W < 0 or W < 1:
        return None
    on = float(seg[lo:hi + 1].sum())
    base = float(seg[lo - W:lo].sum())
    snr = on / (base + 1e-30)
    if not (np.isfinite(snr) and snr > 0):
        return None
    return dict(snr=snr, W=W, lo=lo, hi=hi, A=A, m=m, pk_over_m=A / m)


def fold_phase_abs(series, fs, f_eff, nb=4096):
    """绝对时间折叠全序列 → (phi_main, phi_ip) 标签参考 (仅用于区分主/间, 不做增强)"""
    prof = np.zeros(nb)
    cnt = np.zeros(nb)
    chunk = int(2.0 * fs)
    t_off = 0.0
    N = series.size
    for i0 in range(0, N, chunk):
        seg = np.asarray(series[i0:i0 + chunk], dtype=np.float64)
        t = t_off + np.arange(len(seg)) / fs
        b = ((t * f_eff) % 1.0 * nb).astype(np.int64)
        prof += np.bincount(b, weights=seg, minlength=nb)
        cnt += np.bincount(b, minlength=nb)
        t_off += len(seg) / fs
    prof = prof / np.maximum(cnt, 1)
    pk = int(np.argmax(prof))
    sep_bin = int(round(PULSE_SEP * nb))
    lo = (pk + sep_bin - int(0.08 * nb)) % nb
    hi = (pk + sep_bin + int(0.08 * nb)) % nb
    if lo < hi:
        ip = lo + int(np.argmax(prof[lo:hi]))
    else:
        roll = np.roll(prof, -lo)
        ip = (lo + int(np.argmax(roll[:hi - lo + nb]))) % nb
    return pk / nb, ip / nb, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--series', default='out_coh_full.npy')
    ap.add_argument('--meta', default='out_coh_full.json')
    ap.add_argument('--par', default=r'D:\VBshare\fast_1937+21\B1937+21.par')
    ap.add_argument('--utc-start', default='2024-11-22-05:49:28')
    ap.add_argument('--t0', type=float, default=0.0)
    ap.add_argument('--t1', type=float, default=34.0)
    ap.add_argument('--seg-ms', type=float, default=SEG_MS)
    ap.add_argument('--snr-thresh', type=float, default=SNR_THRESH)
    ap.add_argument('--out', default='out_duffing_mark_20ms')
    ap.add_argument('--only', type=int, default=-1, help='只画第 only 个窗口(调试)')
    ap.add_argument('--F', type=float, default=CFG['F'])
    ap.add_argument('--omega', type=float, default=CFG['omega'])
    ap.add_argument('--k', type=float, default=CFG['k'])
    ap.add_argument('--xi', type=float, default=CFG['xi'])
    args = ap.parse_args()

    series = np.load(args.series, mmap_mode='r')
    fs = 2.0e6
    t_start_abs = args.t0
    if os.path.exists(args.meta):
        with open(args.meta, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        fs = float(meta.get('f_out', fs))
        t_start_abs = float(meta.get('t_start', args.t0))
    N = series.size
    t_obs_end = t_start_abs + N / fs

    par = parse_par(args.par)
    f_eff, f0_epoch, v_proj, mjd0 = compute_feff(par, args.utc_start)

    print('=' * 78)
    print('[duffing mark 20ms] 单脉冲标注 SNR>%.0f + Duffing 增强' % args.snr_thresh)
    print('[series] %s: %d samples @ %.1f MHz (%.4f s)' % (args.series, N, fs / 1e6, N / fs))
    print('[timing] f_eff=%.9f Hz  P=%.6f ms' % (f_eff, 1e3 / f_eff))

    # 折叠仅作主/间标签参考 (标注定位用, 非增强方法)
    t_fold0 = time.time()
    phi_main_f, phi_ip_f, prof = fold_phase_abs(series, fs, f_eff)
    print('[fold-ref] phi_main=%.4f phi_ip=%.4f sep=%.4f  (%.1fs, 仅作主/间区分)' % (
        phi_main_f, phi_ip_f, (phi_ip_f - phi_main_f) % 1.0, time.time() - t_fold0))

    cfg = dict(CFG, F=args.F, omega=args.omega, k=args.k, xi=args.xi)
    P = 1.0 / f_eff
    n_seg = int(round((args.t1 - args.t0) / (args.seg_ms * 1e-3)))
    t0s = args.t0 + np.arange(n_seg) * (args.seg_ms * 1e-3)

    # ---- 第一步: 全程扫描输入, 找所有 SNR>thr 事件 ----
    scan_res = []          # (seg_index, lo_idx_in_seg, snr, W, pk/m)
    print('[scan] 全程输入 SNR 扫描 ...')
    for si, seg_t0 in enumerate(t0s):
        i0 = int(round((seg_t0 - t_start_abs) * fs))
        i1 = i0 + int(round(args.seg_ms * 1e-3 * fs))
        if i1 > N:
            break
        seg = np.asarray(series[i0:i1], dtype=np.float64)
        m = float(np.median(seg))
        sigma = float(1.4826 * np.median(np.abs(seg - m)))
        # 数学约束: SNR = sum(on)/sum(base) <= pk/m (base>=W*m), 故 SNR>thr ⟹ pk > thr*m
        pk_lim = args.snr_thresh * m
        mid = seg[1:-1]
        cand = np.where((mid >= seg[:-2]) & (mid > seg[2:]) & (mid > pk_lim))[0] + 1
        for p in cand.tolist():
            r = pulse_snr(seg, p, m, sigma)
            if r and r['snr'] > args.snr_thresh:
                scan_res.append((si, p, r))
    print('[scan] SNR>%.0f 事件: %d 个' % (args.snr_thresh, len(scan_res)))

    # 合并同一爆发簇内重复事件 (region 重叠视为同一脉冲, 保留 SNR 最高者)
    if scan_res:
        merged = []
        scan_res.sort(key=lambda x: (x[0], x[2]['lo']))
        for si, p, r in scan_res:
            if merged and merged[-1][0] == si:
                ls, _, lr = merged[-1]
                if r['lo'] <= lr['hi']:   # 区间重叠 → 同一脉冲/爆发
                    if r['snr'] > lr['snr']:
                        merged[-1] = (si, p, r)
                    continue
            merged.append((si, p, r))
        scan_res = merged
        print('[scan] 合并同簇后: %d 个' % len(scan_res))

    # ---- 第二步: 锚定首个主脉冲 ----
    if scan_res:
        # 首个事件绝对相位, 与折叠主/间相位比对定标签
        si, p, r = scan_res[0]
        t_ev = t0s[si] + p / fs
        ph_ev = (t_ev * f_eff) % 1.0
        d_main = min(abs(ph_ev - phi_main_f), 1 - abs(ph_ev - phi_main_f))
        d_ip = min(abs(ph_ev - phi_ip_f), 1 - abs(ph_ev - phi_ip_f))
        is_main = d_main <= d_ip
        phi_main_abs = ph_ev if is_main else (ph_ev - PULSE_SEP) % 1.0
        # 用信噪比最高的做精锚: 取前几个事件中SNR最高且在同类
        best = scan_res[0]
        for si2, p2, r2 in scan_res[:min(10, len(scan_res))]:
            t2 = t0s[si2] + p2 / fs
            ph2 = (t2 * f_eff) % 1.0
            d2m = min(abs(ph2 - phi_main_f), 1 - abs(ph2 - phi_main_f))
            d2i = min(abs(ph2 - phi_ip_f), 1 - abs(ph2 - phi_ip_f))
            if (d2m <= d2i) == is_main and r2['snr'] > best[2]['snr']:
                best = (si2, p2, r2)
        sb, pb, rb = best
        t_first = t0s[sb] + pb / fs
        phi_main_abs = (t_first * f_eff) % 1.0
        print('[anchor] 首个主脉冲: t=%.6f s  SNR=%.1f W=%d pk/m=%.1f  '
              'phi_main_abs=%.4f' % (t_first, rb['snr'], rb['W'], rb['pk_over_m'],
                                     phi_main_abs))
    else:
        phi_main_abs = phi_main_f
        print('[anchor] 无 SNR>%.0f 事件, 退回折叠相位 phi_main=%.4f' % (
            args.snr_thresh, phi_main_abs))
    phi_ip_abs = (phi_main_abs + PULSE_SEP) % 1.0

    # ---- 第三步: 逐窗口 Duffing + 标注画图 ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    rows = []
    print('[duffing] 逐窗口增强 + 画图 ...')
    for si, seg_t0 in enumerate(t0s):
        i0 = int(round((seg_t0 - t_start_abs) * fs))
        i1 = i0 + int(round(args.seg_ms * 1e-3 * fs))
        if i1 > N:
            break
        seg = np.asarray(series[i0:i1], dtype=np.float64)
        seg_n = (seg - seg.mean()) / (seg.std() + 1e-30)
        # 本窗口命中事件 (输入)
        hits = [x for x in scan_res if x[0] == si]
        evs = []
        for _, p, r in hits:
            t_ev = seg_t0 + p / fs
            ph = (t_ev * f_eff) % 1.0
            dm = min(abs(ph - phi_main_abs), 1 - abs(ph - phi_main_abs))
            di = min(abs(ph - phi_ip_abs), 1 - abs(ph - phi_ip_abs))
            kind = 'main' if dm <= di else 'ip'
            evs.append(dict(kind=kind, i=p, t=t_ev, snr=r['snr'], W=r['W'],
                            pk_over_m=r['pk_over_m']))
        evs.sort(key=lambda e: e['i'])
        rows.append(dict(seg_t0=seg_t0, n_hits=len(evs), events=evs))

        # --only 调试: 只计算目标窗口的 Duffing, 其余只登记事件
        if args.only >= 0 and si != args.only:
            continue
        dt = 1.0 / fs
        t0rk = time.time()
        out, _, _, _, _ = rk4_coupled(seg_n, dt, cfg['F'], cfg['omega'],
                                      cfg['k'], cfg['xi'],
                                      cfg['x1_init'], cfg['y1_init'],
                                      cfg['x2_init'], cfg['y2_init'])

        if args.only >= 0 and si != args.only:
            continue

        # 画图: 输入 (上) / Duffing输出 (下), 标注位置完全相同
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        t_ms = (np.arange(len(seg)) / fs) * 1e3
        axes[0].plot(t_ms, seg_n, 'b-', lw=0.6)
        axes[0].set_ylabel('input power (z-score)')
        axes[0].set_title('Input coherent-dedispersed series  t0=%.3f s  '
                          '(红虚线=主脉冲, 橙点线=中间脉冲; 仅 SNR>%.0f)' % (
                              seg_t0, args.snr_thresh))
        axes[0].grid(alpha=0.3)
        axes[1].plot(t_ms, out, 'r-', lw=0.6)
        axes[1].set_ylabel(r'Duffing output $\omega(y_1-y_2)$')
        axes[1].set_xlabel('time (ms)')
        axes[1].set_title('Strongly-coupled Duffing output  (标注位置与输入一致)')
        axes[1].grid(alpha=0.3)
        for e in evs:
            color = 'red' if e['kind'] == 'main' else 'tab:orange'
            ls = '--' if e['kind'] == 'main' else ':'
            for ax in axes:
                ax.axvline(t_ms[e['i']], color=color, ls=ls, lw=1.0, alpha=0.9)
        fig.tight_layout()
        fname = '%s_seg%04d.png' % (args.out, si)
        fig.savefig(fname, dpi=130, bbox_inches='tight')
        plt.close(fig)
        if args.only >= 0:
            print('[plot] %s (仅调试窗口 %d)' % (fname, si))
        elif si % 100 == 0:
            print('[seg %d/%d] t0=%.3fs  RK4 %.2fs  hits=%d -> %s' % (
                si, len(t0s), seg_t0, time.time() - t0rk, len(evs), fname), flush=True)

    summary = dict(f_eff=f_eff, phi_main_abs=phi_main_abs,
                   phi_ip_abs=phi_ip_abs, phi_main_fold=phi_main_f,
                   phi_ip_fold=phi_ip_f, snr_thresh=args.snr_thresh,
                   seg_ms=args.seg_ms, cfg=cfg, t_start=t_start_abs,
                   n_seg=len(rows), total_hits=sum(r['n_hits'] for r in rows),
                   segments=rows)
    with open(args.out + '.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('[done] %d segments; %d hits; summary -> %s.json' % (
        len(rows), summary['total_hits'], args.out))


if __name__ == '__main__':
    main()
