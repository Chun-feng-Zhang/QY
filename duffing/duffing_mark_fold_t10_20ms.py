# -*- coding: utf-8 -*-
"""
duffing_mark_fold_t10_20ms.py — t10_2s 20ms 窗标注图 (锚定+等间隔网格) + 2D折叠图
================================================================================
窗图 (每 20ms 一张, 类似 out_duffing_mark_20ms_seg1334.png 的旧版满网格风格):
  - 锚定: 原始输入波形中 SNR 最高者 (首个脉冲), 线加粗
  - 主脉冲: 红 dash-dot (竖虚点线), 中间脉冲: 橙 dotted (点线)
  - 全部网格线等间隔 P: 优先 = 相邻两个 SNR>20 主脉冲实测间距
    (par 1.5578ms 是历元周期, 非现在) → 无实测对时回退 P=1/f_eff=1.557908ms
  - Duffing 输出面板标注与输入面板同一绝对时刻
折叠图 (整段 2s 数据, out_t10_fold_2d.png):
  - 上子图: 横轴相位, 纵轴功率(z), 主脉冲 SNR = on幅度和/前缘等宽基线幅度和
  - 下子图: 横轴相位, 纵轴频段(子带), 颜色=功率
"""
import argparse, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

AU_M = 1.495978707e11
C_LIGHT = 2.99792458e8
OMEGA_EARTH = 7.292115e-5
CFG = dict(F=0.8, omega=4070.0, k=3.0, xi=0.08,
           x1_init=0.0, y1_init=0.0, x2_init=0.0, y2_init=0.0)


from numba import njit


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


def pulse_snr(seg, p, m, sigma):
    """区域SNR(用户定义): on=含峰连续区, base=on前沿等宽样本; SNR=sum(on)/sum(base)"""
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


def fold_profile(series, fs, f_eff, t_start, nb=4096):
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
    return prof / np.maximum(cnt, 1)


def component_phases(prof, nb):
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
    ap.add_argument('--seg-ms', type=float, default=20.0)
    ap.add_argument('--snr-thresh', type=float, default=5.0)
    ap.add_argument('--snr-pair', type=float, default=20.0)
    ap.add_argument('--out', default='out_t10_mark_20ms')
    ap.add_argument('--only', type=int, default=-1)
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--count', type=int, default=-1)
    ap.add_argument('--no-windows', dest='windows', action='store_false', default=True)
    ap.add_argument('--no-fold', dest='fold', action='store_false', default=True)
    args = ap.parse_args()

    series = np.load(args.series, mmap_mode='r')
    fs = 2.0e6
    t_start_abs = 0.0
    if os.path.exists(args.meta):
        with open(args.meta, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        fs = float(meta.get('f_out', fs))
        t_start_abs = float(meta.get('t_start', 0.0))
    par = parse_par(args.par)
    f_eff, f0_epoch, v_proj, mjd0 = compute_feff(par, args.utc_start)
    P_par = 1.0 / f_eff
    print('=' * 80)
    print('[series] %s: %d samples @ %.1f MHz, t_start=%.3f s' % (
        args.series, series.size, fs / 1e6, t_start_abs))
    print('[timing] f_eff=%.9f Hz  P_current=%.6f ms (历元外推+顶心改正; par 原始 F0=1.5578ms 仅历元值)' % (
        f_eff, P_par * 1e3))

    # ---------- 折叠 (绝对时间, 2s) ----------
    prof = fold_profile(series, fs, f_eff, t_start_abs, nb=4096)
    phi_main_f, phi_ip_f = component_phases(prof, 4096)
    delta = (phi_ip_f - phi_main_f) % 1.0
    print('[fold] phi_main=%.4f phi_ip=%.4f delta=%.4f' % (phi_main_f, phi_ip_f, delta))

    # ---------- SNR>20 主脉冲对 → 实测周期 ----------
    P_data, P_src = None, 'par'
    n_all = series.size // int(args.seg_ms * 1e-3 * fs)
    strong = []
    for si in range(n_all):
        i0 = int(round((t_start_abs + si * args.seg_ms * 1e-3 - t_start_abs) * fs))
        seg = np.asarray(series[i0:i0 + int(args.seg_ms * 1e-3 * fs)], dtype=np.float64)
        m = float(np.median(seg))
        sig = float(1.4826 * np.median(np.abs(seg - m)))
        mid = seg[1:-1]
        cand = np.where((mid >= seg[:-2]) & (mid > seg[2:]) & (mid > args.snr_pair * m))[0] + 1
        for p in cand.tolist():
            r = pulse_snr(seg, p, m, sig)
            if r and r['snr'] > args.snr_pair:
                t_abs = t_start_abs + si * args.seg_ms * 1e-3 + p / fs
                ph = (t_abs * f_eff) % 1.0
                dm = abs(circ_dist(ph, phi_main_f)); di = abs(circ_dist(ph, phi_ip_f))
                strong.append((t_abs, ph, dm < di, r['snr']))
    strong.sort(key=lambda x: x[0])
    mains = [x for x in strong if x[2]]
    print('[scan] SNR>%.0f: %d 个 (主类 %d)' % (args.snr_pair, len(strong), len(mains)))
    if len(mains) >= 2:
        gaps = np.diff([x[0] for x in mains])
        P_data = float(np.median(gaps))
        P_src = 'measured(SNR>%.0f)' % args.snr_pair
        print('[P] 实测: %.6f ms (n=%d 对)' % (P_data * 1e3, len(gaps)))
    else:
        P_data = P_par
        print('[P] 无相邻 SNR>%.0f 主脉冲对 → 回退 P=1/f_eff=%.6f ms' % (
            args.snr_pair, P_data * 1e3))
    P = P_data

    seg_len = int(args.seg_ms * 1e-3 * fs)
    n_seg_all = series.size // seg_len
    rows = []

    # ---------- 逐窗 ----------
    if args.windows:
        t_proc = time.time()
        for si in range(n_seg_all):
            if args.only >= 0 and si != args.only:
                continue
            if args.only < 0 and args.count >= 0 and not (args.first <= si < args.first + args.count):
                continue
            i0 = int(round((t_start_abs + si * args.seg_ms * 1e-3 - t_start_abs) * fs))
            seg = np.asarray(series[i0:i0 + seg_len], dtype=np.float64)
            m = float(np.median(seg))
            sig = float(1.4826 * np.median(np.abs(seg - m)))
            seg_n = (seg - seg.mean()) / (seg.std() + 1e-30)
            mid = seg[1:-1]
            cand = np.where((mid >= seg[:-2]) & (mid > seg[2:]) &
                            (mid > m + 3 * sig))[0] + 1
            wins = []
            for p in cand.tolist():
                r = pulse_snr(seg, p, m, sig)
                if r:
                    r['t'] = t_start_abs + si * args.seg_ms * 1e-3 + p / fs
                    r['i'] = p
                    wins.append(r)

            row = dict(seg_t0=t_start_abs + si * args.seg_ms * 1e-3,
                       max_snr=wins[0]['snr'] if wins else None,
                       anchor=None, phi_main_g=None, n_grid=0)
            # 锚定: 窗内 SNR 最高
            if wins:
                wins.sort(key=lambda r: -r['snr'])
                anc = wins[0]
                ph_a = (anc['t'] * f_eff) % 1.0
                dm = abs(circ_dist(ph_a, phi_main_f))
                di = abs(circ_dist(ph_a, phi_ip_f))
                anchor_main = dm <= di
                phi_g = ph_a if anchor_main else (ph_a - delta) % 1.0
                row['anchor'] = dict(t=anc['t'], i=anc['i'], snr=anc['snr'],
                                     W=anc['W'], pk_over_m=anc['pk_over_m'],
                                     main=anchor_main, phi=ph_a)
                row['phi_main_g'] = phi_g
            else:
                phi_g = phi_main_f  # 无候选时退回折叠主相位
                row['phi_main_g'] = phi_g
            rows.append(row)

            # ---- RK4 Duffing ----
            dt = 1.0 / fs
            t_rk = time.time()
            out, _, _, _, _ = rk4_coupled(seg_n, dt, CFG['F'], CFG['omega'],
                                          CFG['k'], CFG['xi'],
                                          0.0, 0.0, 0.0, 0.0)

            # ---- 画图: 两面板 + 等间隔网格 ----
            fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
            t_ms = np.arange(seg_len) / fs * 1e3
            axes[0].plot(t_ms, seg_n, 'b-', lw=0.6)
            axes[0].set_ylabel('input power (z)')
            st_t = ('Input  t0=%.3fs  锚定SNR=%.2f  P=%.6fms(%s)  红-.=主脉冲 橙:=中间脉冲' % (
                row['seg_t0'], row['anchor']['snr'] if row['anchor'] else -1,
                P * 1e3, 'data' if P_src.startswith('measured') else 'f_eff'))
            axes[0].set_title(st_t)
            axes[0].grid(alpha=0.3)
            axes[1].plot(t_ms, out, 'r-', lw=0.6)
            axes[1].set_ylabel('Duffing out')
            axes[1].set_xlabel('time (ms)')
            axes[1].set_title('Duffing output (标注与输入同一绝对时刻)')
            axes[1].grid(alpha=0.3)

            def grid_times(phi_kind):
                ph_g = row['phi_main_g']
                ph_k = ph_g if phi_kind == 'main' else (ph_g + delta) % 1.0
                # 相位 → 窗内时刻: t 满足 ((t-t0)*f_eff + ph0) = ph_k (mod 1), ph0=窗起点相位
                ph0 = (row['seg_t0'] * f_eff) % 1.0
                rel = ((ph_k - ph0) % 1.0) / f_eff  # 0..P 内的相对时刻
                out_t = []
                n = int((args.seg_ms * 1e-3 - rel) / P) + 1
                for kk in range(-1, n + 1):
                    tt = row['seg_t0'] + rel + kk * P
                    if -0.5e-3 <= tt - row['seg_t0'] <= args.seg_ms * 1e-3 + 0.5e-3:
                        out_t.append(tt)
                return out_t

            n_g = 0
            for kind, color, ls in (('main', 'red', '-.'), ('ip', 'tab:orange', ':')):
                for tt in grid_times(kind):
                    x = (tt - row['seg_t0']) * 1e3
                    for ax in axes:
                        ax.axvline(x, color=color, ls=ls, lw=1.0, alpha=0.85)
                    n_g += 1
            row['n_grid'] = n_g
            if row['anchor'] is not None:
                xa = (row['anchor']['t'] - row['seg_t0']) * 1e3
                col = 'red' if row['anchor']['main'] else 'tab:orange'
                ls_a = '-.' if row['anchor']['main'] else ':'
                for ax in axes:
                    ax.axvline(xa, color=col, ls=ls_a, lw=2.2, alpha=1.0)
            fig.tight_layout()
            fname = '%s_seg%04d.png' % (args.out, si)
            fig.savefig(fname, dpi=130, bbox_inches='tight')
            plt.close(fig)
            if si % 20 == 0:
                print('[seg %d/%d] t0=%.3fs RK4=%.2fs grid=%d' % (
                    si, n_seg_all, row['seg_t0'], time.time() - t_rk, n_g), flush=True)
        print('[windows done] %d 张, %.1fs' % (n_seg_all, time.time() - t_proc))

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    with open(args.out + '.json', 'w', encoding='utf-8') as f:
        json.dump(_clean(dict(f_eff=f_eff, phi_main_fold=phi_main_f,
                              phi_ip_fold=phi_ip_f, delta=delta,
                              p_ms=P * 1e3, p_source=P_src,
                              snr_pair=args.snr_pair, seg_ms=args.seg_ms,
                              t_start=t_start_abs, segments=rows)),
                  f, indent=2, ensure_ascii=False)

    # ---------- 折叠图 (2D) ----------
    if args.fold:
        t_fold = time.time()
        data = np.asarray(series[:], dtype=np.float64)
        z = (data - data.mean()) / (data.std() + 1e-30)
        nb = 4096
        nb2 = 128
        t = t_start_abs + np.arange(data.size) / fs
        ph = (t * f_eff) % 1.0

        # --- 上: 折叠 profile (z + 原始) ---
        b = (ph * nb).astype(np.int64)
        csum = np.bincount(b, weights=data, minlength=nb)
        ccnt = np.bincount(b, minlength=nb)
        prof_raw = csum / np.maximum(ccnt, 1)
        prof_z = (prof_raw - prof_raw.mean()) / (prof_raw.std() + 1e-30)

        mp = float(np.median(prof_raw))
        sp = float(1.4826 * np.median(np.abs(prof_raw - mp)))
        # 主脉冲 on 区 (用户定义区域SNR, 用原始功率profile)
        pk = int(np.argmax(prof_raw))
        A = prof_raw[pk]
        r_main = pulse_snr(prof_raw, pk, mp, sp)
        # 间脉冲 on 区
        pk2 = int(round((phi_ip_f % 1.0) * nb)) % nb
        A2 = prof_raw[pk2]
        r_ip = pulse_snr(prof_raw, pk2, mp, sp)

        fig, axes = plt.subplots(2, 1, figsize=(13, 10))
        ax1 = axes[0]
        xs = np.arange(nb) / nb
        ax1.plot(xs, prof_z, 'b-', lw=1.0)
        ax1.set_xlim(0, 1)
        ax1.set_ylabel('power (z-score)')
        ax1.set_xlabel('phase')
        ax1.set_title('2s 折叠 profile (t=%.1f-%.1fs), nb=%d' % (
            t_start_abs, t_start_abs + data.size / fs, nb))
        ax1.grid(alpha=0.3)
        # 标注主脉冲 on/base 窗口与 SNR
        if r_main:
            lo, hi = r_main['lo'], r_main['hi']
            W = hi - lo + 1
            ytop = float(np.max(prof_z)) * 0.96
            ax1.axvspan(xs[lo], xs[hi], color='red', alpha=0.25, lw=0)
            ax1.axvspan(xs[lo - W], xs[lo], color='gray', alpha=0.4, lw=0)
            ax1.annotate('主脉冲\non区', xy=(xs[(lo + hi) // 2], ytop), fontsize=9, color='red')
            ax1.annotate('等宽基线', xy=(xs[(2 * lo - W) // 2], -0.1), fontsize=9, color='gray')
            snr_txt = '主脉冲 SNR = %.3f\n(on幅度和/前缘等宽基线幅度和)' % r_main['snr']
            ax1.text(0.02, 0.96, snr_txt, transform=ax1.transAxes, fontsize=11,
                     color='darkred',
                     bbox=dict(boxstyle='round', fc='white', ec='darkred', alpha=0.9))
            print('[fold-snr] main W=%d snr=%.4f peak_z=%.1f' % (W, r_main['snr'], prof_z[pk]))
        if r_ip:
            lo, hi = r_ip['lo'], r_ip['hi']
            ax1.axvspan(xs[lo], xs[hi], color='orange', alpha=0.25, lw=0)
            print('[fold-snr] ip W=%d snr=%.4f' % (r_ip['hi'] - r_ip['lo'] + 1, r_ip['snr']))
        ax1.axvline(phi_main_f, color='red', ls='--', lw=1)
        ax1.axvline(phi_ip_f, color='orange', ls='--', lw=1)

        # --- 下: 相位×频段 2D (STFT) ---
        wlen = 64          # 31.25 kHz/bin
        hop = 64
        nwin = data.size // hop - (wlen // hop) + 1
        X = np.empty((nwin, wlen))
        for j in range(nwin):
            s0 = j * hop
            X[j] = data[s0:s0 + wlen]
        S = np.abs(np.fft.rfft(X, axis=1)) ** 2   # (nwin, 33)
        t_c = t_start_abs + (np.arange(nwin) * hop + wlen / 2) / fs
        ph_c = (t_c * f_eff) % 1.0
        b2 = (ph_c * nb2).astype(np.int64)
        img = np.zeros((S.shape[1], nb2))
        cnt2 = np.zeros(nb2)
        for j in range(nwin):
            img[:, b2[j]] += S[j]
            cnt2[b2[j]] += 1
        img /= np.maximum(cnt2[None, :], 1)
        img = img[:32, :]  # 0..1MHz, 每行31.25kHz
        # 行归一化: 相对功率 σ
        row_med = np.median(img, axis=1, keepdims=True)
        row_mad = 1.4826 * np.median(np.abs(img - row_med), axis=1, keepdims=True)
        rel = (img - row_med) / np.maximum(row_mad, 1e-12)
        ax2 = axes[1]
        im = ax2.imshow(rel, aspect='auto', origin='lower',
                        extent=[0, 1, 0, 1.0], cmap='viridis',
                        norm=Normalize(vmin=-2, vmax=8))
        ax2.set_xlabel('phase')
        ax2.set_ylabel('frequency (MHz)')
        ax2.set_title('相位×频段折叠图 (2s, 行归一化相对功率 σ; 颜色=功率)')
        for phx in (phi_main_f, phi_ip_f):
            ax2.axvline(phx, color='white', ls='--', lw=1)
        cb = fig.colorbar(im, ax=ax2, pad=0.01)
        cb.set_label('power (σ/行)')
        fig.tight_layout()
        fn = 'out_t10_fold_2d.png'
        fig.savefig(fn, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print('[fold-fig] %s (%.1fs)' % (fn, time.time() - t_fold))

    print('[done] P=%.6fms(%s) windows=%d fold=1' % (P * 1e3, P_src, n_seg_all))


if __name__ == '__main__':
    main()
