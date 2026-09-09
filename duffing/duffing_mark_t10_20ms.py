# -*- coding: utf-8 -*-
"""
duffing_mark_t10_20ms.py — 基于 out_coh_t10_2s.npy 的 20ms 单脉冲标注 + Duffing 增强对比
====================================================================================
标注规范:
  1. 每张图在【原始输入波形】中锚定首个脉冲 = SNR 最高者 (区域SNR, 定义见 pulse_snr)
  2. 主脉冲/中间脉冲用等间隔竖线标记:
     - 间距 P: 优先 = 相邻两个 SNR>20 主脉冲的时间间隔 (实测当前自转周期)
               (par 文件的 1.5578 ms 只是历元周期, 仅作回退参考)
     - 无实测对时回退 P = 1/f_eff (含 F1 历元外推 + 顶心改正)
  3. 只标注 SNR > 5 的峰: 主脉冲 = 红 dash-dot, 中间脉冲 = 橙 dotted
  4. Duffing 输出面板标注位置与输入面板完全一致
  5. 每 20 ms 一图
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
SNR_THRESH = 5.0
SNR_PAIR = 20.0
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
    """区域 SNR (用户定义): on = 含峰连续区(峰值向下扩展, 容忍窄凹陷),
    base = on 前沿等宽样本;  SNR = sum(on)/sum(base)"""
    A = float(seg[p])
    if A <= m:
        return None
    thr = m + max(0.05 * (A - m), 5.0 * sigma)
    lo = p
    while lo - 1 >= 0 and seg[lo - 1] >= thr:
        lo -= 1
    hi = p
    while hi + 1 < len(seg) and seg[hi + 1] >= thr:
        hi += 1
    W = hi - lo + 1
    if lo - W < 0 or W < 1:
        return None
    on = float(seg[lo:hi + 1].sum())
    base = float(seg[lo - W:lo].sum())
    snr = on / (base + 1e-30)
    if not (np.isfinite(snr) and snr > 0):
        return None
    return dict(snr=snr, W=W, lo=lo, hi=hi, A=A, m=m, pk_over_m=A / m)


def circ_dist(a, b):
    return (a - b + 0.5) % 1.0 - 0.5


def fold_phase(series, fs, f_eff, t_start, nb=4096):
    """绝对时间折叠 → (phi_main, phi_ip), 仅作主/间标签与间距参考"""
    prof = np.zeros(nb)
    cnt = np.zeros(nb)
    chunk = int(2.0 * fs)
    N = series.size
    for i0 in range(0, N, chunk):
        seg = np.asarray(series[i0:i0 + chunk], dtype=np.float64)
        t = t_start + (i0 + np.arange(len(seg))) / fs
        b = ((t * f_eff) % 1.0 * nb).astype(np.int64)
        prof += np.bincount(b, weights=seg, minlength=nb)
        cnt += np.bincount(b, minlength=nb)
    prof = prof / np.maximum(cnt, 1)
    pk = int(np.argmax(prof))
    sep_bin = int(round(0.5283 * nb))
    lo = (pk + sep_bin - int(0.08 * nb)) % nb
    hi = (pk + sep_bin + int(0.08 * nb)) % nb
    if lo < hi:
        ip = lo + int(np.argmax(prof[lo:hi]))
    else:
        roll = np.roll(prof, -lo)
        ip = (lo + int(np.argmax(roll[:hi - lo + nb]))) % nb
    return pk / nb, ip / nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--series', default='out_coh_t10_2s.npy')
    ap.add_argument('--meta', default='out_coh_t10_2s.json')
    ap.add_argument('--par', default=r'D:\VBshare\fast_1937+21\B1937+21.par')
    ap.add_argument('--utc-start', default='2024-11-22-05:49:28')
    ap.add_argument('--seg-ms', type=float, default=SEG_MS)
    ap.add_argument('--snr-thresh', type=float, default=SNR_THRESH)
    ap.add_argument('--snr-pair', type=float, default=SNR_PAIR)
    ap.add_argument('--out', default='out_t10_mark_20ms')
    ap.add_argument('--only', type=int, default=-1)
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--count', type=int, default=-1)
    ap.add_argument('--F', type=float, default=CFG['F'])
    ap.add_argument('--omega', type=float, default=CFG['omega'])
    ap.add_argument('--k', type=float, default=CFG['k'])
    ap.add_argument('--xi', type=float, default=CFG['xi'])
    args = ap.parse_args()

    series = np.load(args.series, mmap_mode='r')
    fs = 2.0e6
    t_start_abs = 0.0
    if os.path.exists(args.meta):
        with open(args.meta, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        fs = float(meta.get('f_out', fs))
        t_start_abs = float(meta.get('t_start', 0.0))
    N = series.size
    par = parse_par(args.par)
    f_eff, f0_epoch, v_proj, mjd0 = compute_feff(par, args.utc_start)

    print('=' * 80)
    print('[t10 mark 20ms] 输入波形锚定 + SNR>%.0f 标注 + Duffing 对比' % args.snr_thresh)
    print('[series] %s: %d samples @ %.1f MHz  t_start=%.3f s' % (
        args.series, N, fs / 1e6, t_start_abs))
    print('[timing] f_eff=%.9f Hz  P_par(1/f_eff)=%.6f ms  (par F0 原始周期 1.5578 ms 仅历元值)' % (
        f_eff, 1e3 / f_eff))

    t_f = time.time()
    phi_main_f, phi_ip_f = fold_phase(series, fs, f_eff, t_start_abs)
    delta = (phi_ip_f - phi_main_f) % 1.0
    print('[fold-ref] phi_main=%.4f phi_ip=%.4f delta=%.4f  (%.1fs; 仅标签/参考)' % (
        phi_main_f, phi_ip_f, delta, time.time() - t_f))

    cfg = dict(CFG, F=args.F, omega=args.omega, k=args.k, xi=args.xi)
    P_par = 1.0 / f_eff

    n_seg_all = int(N // int(args.seg_ms * 1e-3 * fs))
    t0s = t_start_abs + np.arange(n_seg_all) * (args.seg_ms * 1e-3)
    n_seg = n_seg_all

    # ---- 全程扫描候选 (pk > snr_pair*m 已含 pk>snr_thresh*m) ----
    cand_all = []   # (seg_i, p, r)
    for si in range(n_seg):
        i0 = int(round((t0s[si] - t_start_abs) * fs))
        seg = np.asarray(series[i0:i0 + int(args.seg_ms * 1e-3 * fs)], dtype=np.float64)
        m = float(np.median(seg))
        sigma = float(1.4826 * np.median(np.abs(seg - m)))
        mid = seg[1:-1]
        pk_lim = args.snr_pair * m   # pk > thresh*m 才可能 SNR>thresh
        cand = np.where((mid >= seg[:-2]) & (mid > seg[2:]) & (mid > pk_lim))[0] + 1
        for p in cand.tolist():
            r = pulse_snr(seg, p, m, sigma)
            if r:
                cand_all.append((si, p, r))
    print('[scan] 候选(峰值>%.0f*噪声中值): %d' % (args.snr_pair, len(cand_all)))

    # ---- 实测周期: 相邻两个 SNR>20 主脉冲的时间间隔 ----
    P_data, P_src = None, 'par'
    strong = []
    for si, p, r in cand_all:
        if r['snr'] > args.snr_pair:
            t_abs = t0s[si] + p / fs
            ph = ((t_abs * f_eff) % 1.0)
            # 主脉冲类: 相位靠近折叠主相位 (delta 空间内最近)
            dm = abs(circ_dist(ph, phi_main_f))
            di = abs(circ_dist(ph, phi_ip_f))
            strong.append((t_abs, ph, dm < di, r['snr']))
    strong.sort(key=lambda x: x[0])
    mains = [x for x in strong if x[2]]
    print('[scan] SNR>%.0f 事件: %d  (主类 %d, 间类 %d)' % (
        args.snr_pair, len(strong), len(mains), len(strong) - len(mains)))
    for x in strong[:12]:
        print('    t=%.6fs ph=%.4f %s snr=%.1f' % (x[0], x[1], 'MAIN' if x[2] else 'IP', x[3]))
    if len(mains) >= 2:
        gaps = np.diff([x[0] for x in mains])
        P_data = float(np.median(gaps))
        P_src = 'measured(SNR>%.0f main pairs)' % args.snr_pair
        print('[period] 实测相邻 SNR>%.0f 主脉冲间距: P=%.6f ms (n=%d, 中值)' % (
            args.snr_pair, P_data * 1e3, len(gaps)))
    else:
        P_data = P_par
        print('[period] 无相邻 SNR>%.0f 主脉冲对 → 回退 P=1/f_eff=%.6f ms '
              '(含 F1 历元改正, 非 par 原始 1.5578ms)' % (args.snr_pair, P_data * 1e3))
    P = P_data

    # ---- 逐窗口处理 ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    rows = []
    for si in range(n_seg):
        i0 = int(round((t0s[si] - t_start_abs) * fs))
        seg = np.asarray(series[i0:i0 + int(args.seg_ms * 1e-3 * fs)], dtype=np.float64)
        m = float(np.median(seg))
        sigma = float(1.4826 * np.median(np.abs(seg - m)))
        seg_n = (seg - seg.mean()) / (seg.std() + 1e-30)

        # 本窗候选 (用较低的 pk 下限 m+3σ 找局部极大, 供锚定)
        mid = seg[1:-1]
        cand_all_w = np.where((mid >= seg[:-2]) & (mid > seg[2:]) &
                              (mid > m + 3 * sigma))[0] + 1
        wins = []
        for p in cand_all_w.tolist():
            r = pulse_snr(seg, p, m, sigma)
            if r:
                r['t'] = t0s[si] + p / fs
                r['i'] = p
                wins.append(r)

        # 锚定 = 本窗 SNR 最高候选 (首个主脉冲)
        row = dict(seg_t0=t0s[si], p_source=P_src, p_ms=P * 1e3,
                   n_cand=len(wins), max_snr=wins[0]['snr'] if wins else None,
                   anchor=None, events=[])
        if wins:
            wins.sort(key=lambda r: -r['snr'])
            anc = wins[0]
            ph_a = (anc['t'] * f_eff) % 1.0
            dm = abs(circ_dist(ph_a, phi_main_f))
            di = abs(circ_dist(ph_a, phi_ip_f))
            anchor_main = dm <= di
            phi_main_g = ph_a if anchor_main else (ph_a - delta) % 1.0
            row['anchor'] = dict(t=anc['t'], i=anc['i'], snr=anc['snr'],
                                 W=anc['W'], pk_over_m=anc['pk_over_m'],
                                 main=anchor_main, phi=ph_a)
            # 网格: 等间隔 P, 从锚定相位出发覆盖整个窗口
            t_win0, t_win1 = t0s[si], t0s[si] + args.seg_ms * 1e-3
            n0 = int(np.floor((t_win0 - anc['t']) / P)) - 1
            for k in range(0, 2 * int(args.seg_ms * 1e-3 / P) + 4):
                tt_main = anc['t'] + (n0 + k) * P
                tt_ip = tt_main + delta * P
                for label, tt in (('main', tt_main), ('ip', tt_ip)):
                    if tt < t_win0 - 0.05 * P or tt > t_win1 + 0.05 * P:
                        continue
                    # 找该网格位置附近的候选峰 (局部极大, ±0.1P)
                    jc = int(round((tt - t0s[si]) * fs))
                    lo_, hi_ = max(0, jc - int(0.1 * P * fs)), \
                               min(len(seg), jc + int(0.1 * P * fs))
                    segm = seg[lo_:hi_]
                    if len(segm) == 0:
                        continue
                    kp = lo_ + int(np.argmax(segm))
                    r2 = pulse_snr(seg, kp, m, sigma)
                    if r2 and r2['snr'] > args.snr_thresh:
                        ph_c = ((t0s[si] + kp / fs) * f_eff) % 1.0
                        ph_g = phi_main_g if label == 'main' else (phi_main_g + delta) % 1.0
                        if abs(circ_dist(ph_c, ph_g)) > 0.2:
                            continue  # 网格相位附近必须真有该类的峰
                        row['events'].append(dict(kind=label, i=kp,
                                                  t=t0s[si] + kp / fs,
                                                  snr=r2['snr'], W=r2['W']))
            # 去重
            seen = set()
            uniq = []
            for e in row['events']:
                key = (e['kind'], e['i'])
                if key not in seen:
                    seen.add(key)
                    uniq.append(e)
            uniq.sort(key=lambda e: e['i'])
            row['events'] = uniq
        rows.append(row)

        if args.only >= 0:
            if si != args.only:
                continue
        elif args.count >= 0 and not (args.first <= si < args.first + args.count):
            continue  # 本段不画图, 但仍已登记 rows

        # Duffing
        dt = 1.0 / fs
        t0rk = time.time()
        out, _, _, _, _ = rk4_coupled(seg_n, dt, cfg['F'], cfg['omega'],
                                      cfg['k'], cfg['xi'],
                                      cfg['x1_init'], cfg['y1_init'],
                                      cfg['x2_init'], cfg['y2_init'])
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        t_ms = np.arange(len(seg)) / fs * 1e3
        axes[0].plot(t_ms, seg_n, 'b-', lw=0.6)
        axes[0].set_ylabel('input power (z-score)')
        st = ('Input  t0=%.3fs  anchor SNR=%.1f  P=%.6fms(%s)  ' % (
            t0s[si], row['anchor']['snr'] if row['anchor'] else -1,
            P * 1e3, 'data' if P_src.startswith('measured') else 'f_eff'))
        axes[0].set_title(st + '  红-.=主脉冲 橙:=中间脉冲 (仅 SNR>%.0f)' % args.snr_thresh)
        axes[0].grid(alpha=0.3)
        axes[1].plot(t_ms, out, 'r-', lw=0.6)
        axes[1].set_ylabel(r'Duffing output $\omega(y_1-y_2)$')
        axes[1].set_xlabel('time (ms)')
        axes[1].set_title('Duffing output  (标注与输入同位置)')
        axes[1].grid(alpha=0.3)
        for e in row['events']:
            color = 'red' if e['kind'] == 'main' else 'tab:orange'
            ls = '-.' if e['kind'] == 'main' else ':'
            for ax in axes:
                ax.axvline(t_ms[e['i']], color=color, ls=ls, lw=1.1, alpha=0.9)
        # 锚定线(首个主脉冲)恒画, 便于目视检查
        if row['anchor'] is not None:
            ac = row['anchor']
            col = 'red' if ac['main'] else 'tab:orange'
            ls = '-.' if ac['main'] else ':'
            for ax in axes:
                ax.axvline(t_ms[ac['i']], color=col, ls=ls, lw=2.0, alpha=0.95)
        fig.tight_layout()
        fname = '%s_seg%04d.png' % (args.out, si)
        fig.savefig(fname, dpi=130, bbox_inches='tight')
        plt.close(fig)
        if args.only >= 0:
            print('[plot] %s  events=%d' % (fname, len(row['events'])))
        elif si % 20 == 0:
            print('[seg %d/%d] t0=%.3fs RK4=%.2fs hits=%d' % (
                si, n_seg, t0s[si], time.time() - t0rk, len(row['events'])), flush=True)

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    with open(args.out + '.json', 'w', encoding='utf-8') as f:
        json.dump(_clean(dict(f_eff=f_eff, phi_main_fold=phi_main_f,
                              phi_ip_fold=phi_ip_f,
                              delta=delta, p_ms=P * 1e3, p_source=P_src,
                              snr_thresh=args.snr_thresh, seg_ms=args.seg_ms,
                              cfg=cfg, t_start=t_start_abs, segments=rows)),
                  f, indent=2, ensure_ascii=False)
    nh = sum(len(r['events']) for r in rows)
    na = sum(1 for r in rows if r['anchor'])
    print('[done] %d windows; anchors=%d; SNR>%.0f marks=%d; json=%s.json' % (
        len(rows), na, args.snr_thresh, nh, args.out))


if __name__ == '__main__':
    main()
