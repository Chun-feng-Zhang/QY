# -*- coding: utf-8 -*-
"""
duffing_singlepulse_30ms.py
===========================
强耦合 Duffing 振子单脉冲 SNR 增强 (无折叠) / strongly-coupled Duffing
single-pulse SNR enhancement (NO folding).

管线 / Pipeline:
  1. 读取相干消色散后基带功率序列 (coherent_dedisp.py 输出, 默认 2 MHz)
  2. 从 par 星历 + 站点运动计算顶心自转频率 f_eff
  3. 窄高斯模板匹配滤波检测单脉冲 → 相位标定 (主脉冲 + 中间脉冲两簇)
  4. 强耦合双振子 Duffing (RK4, 参考 5-diffuse v5 最优参数)
  5. 30 ms 分段: 标注每个预测主/中间脉冲, 画输入/输出对比, 算时域 SNR 增强

物理 / Physics:
  传统 Holmes-Duffing 仅能判别周期成分存在性, 无法重建时域波形; 本脚本用强耦合
  改进 Duffing 双振子 (v5):
    dx1/dt = w*y1
    dy1/dt = w*( -xi*y1 + x1 - x1^3 - k*xi*(x1-x2) - k*(y1-y2) + F*cos(wt) + s(t) )
    dx2/dt = w*y2
    dy2/dt = w*( -xi*y2 + x2 - x2^3 - k*xi*(x2-x1) - k*(y2-y1) + F*cos(wt) )
    output  = w*(y1 - y2)
  s(t) 为归一化消色散功率序列, w≈2*pi*f_eff 使振子谐振于脉冲重复频率。

用法:
  python duffing_singlepulse_30ms.py --series out_coh_t10_2s.npy --meta out_coh_t10_2s.json \
      --par D:\\VBshare\\fast_1937+21\\B1937+21.par --t0 10.0 --seg-ms 30 --n-seg 4
"""
import argparse, json, os, sys
import numpy as np

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False
    def njit(*a, **k):
        def _d(f):
            return f
        return _d

C_LIGHT = 299792458.0
AU_M = 1.495978707e11
OMEGA_EARTH = 2.0 * np.pi / 86164.0905

CFG = dict(F=0.8, omega=4070.0, k=3.0, xi=0.08,
           x1_init=1.4, y1_init=0.3, x2_init=1.4, y2_init=0.3)


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
    """强耦合双振子 Duffing RK4, 返回 output=w*(y1-y2) 及末态"""
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


def circ_dist(a, b):
    return (a - b + 0.5) % 1.0 - 0.5


def calibrate_phase(series, fs, f_eff, t0, nb=1024, sep=538.0):
    """通过折叠定位主/间脉冲相位(仅用于标注, 非增强方法)
    返回 (phi_main, phi_ip)"""
    t = t0 + np.arange(len(series)) / fs
    ph = (t * f_eff) % 1.0
    prof = np.bincount((ph * nb).astype(np.int64), weights=series, minlength=nb)
    cnt = np.bincount((ph * nb).astype(np.int64), minlength=nb)
    prof = prof / np.maximum(cnt, 1)
    # 主脉冲 = 全局峰值
    pk = int(np.argmax(prof))
    # 间脉冲 = 距主脉冲 sep 处 ±0.05 周期内峰值
    ip_c = (pk + sep) % nb
    lo = int(ip_c - 0.05 * nb); hi = int(ip_c + 0.05 * nb)
    ip_pk = lo + int(np.argmax(np.roll(prof, -lo)[:hi - lo]))
    ip_pk = ip_pk % nb
    return pk / nb, ip_pk / nb


def predicted_pulse_times(t0, seg_ms, f_eff, phi_main, phi_ip):
    t1 = t0 + seg_ms * 1e-3
    t_main, t_ip = [], []
    n0 = int(np.floor(t0 * f_eff - phi_main)) - 1
    while True:
        tt = (n0 + phi_main) / f_eff
        if tt > t1:
            break
        if tt >= t0:
            t_main.append(tt)
        n0 += 1
    n0 = int(np.floor(t0 * f_eff - phi_ip)) - 1
    while True:
        tt = (n0 + phi_ip) / f_eff
        if tt > t1:
            break
        if tt >= t0:
            t_ip.append(tt)
        n0 += 1
    return np.array(t_main), np.array(t_ip)


def detect_strong_peaks(signal, threshold=5.0, min_dist=3):
    """检测峰值/基线 > threshold 的强峰, 返回 (sample_indices, peak_values)"""
    baseline = np.median(np.abs(signal)) + 1e-30
    abs_s = np.abs(signal)
    cand = abs_s > threshold * baseline
    peaks_idx = []
    peaks_val = []
    i = 0
    n = len(signal)
    while i < n:
        if cand[i]:
            j = i
            while j + 1 < n and cand[j + 1] and abs_s[j + 1] > abs_s[j]:
                j += 1
            # 局部极大
            lo = max(0, i - min_dist); hi = min(n, j + min_dist + 1)
            k = lo + int(np.argmax(abs_s[lo:hi]))
            peaks_idx.append(k)
            peaks_val.append(signal[k])
            i = j + min_dist + 1
        else:
            i += 1
    return np.array(peaks_idx), np.array(peaks_val)


def on_off_snr(signal, t_abs, f_eff, phi_main, phi_ip, half=0.03):
    ph = (t_abs * f_eff) % 1.0
    on = (np.abs(circ_dist(ph, phi_main)) < half) | (np.abs(circ_dist(ph, phi_ip)) < half)
    off = ~on
    if on.sum() < 10 or off.sum() < 10:
        return np.nan, np.nan
    p_on = np.mean(signal[on] ** 2)
    p_off = np.mean(signal[off] ** 2)
    m_off = np.mean(signal[off])
    s_off = np.std(signal[off]) + 1e-30
    snr_on = (np.mean(signal[on]) - m_off) / s_off
    return p_on / p_off, snr_on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--series', default='out_coh_t10_2s.npy')
    ap.add_argument('--meta', default='out_coh_t10_2s.json')
    ap.add_argument('--par', default=r'D:\VBshare\fast_1937+21\B1937+21.par')
    ap.add_argument('--utc-start', default='2024-11-22-05:49:28')
    ap.add_argument('--t0', type=float, default=10.0)
    ap.add_argument('--seg-ms', type=float, default=30.0)
    ap.add_argument('--n-seg', type=int, default=4)
    ap.add_argument('--out', default='out_duffing_30ms')
    ap.add_argument('--F', type=float, default=0.8)
    ap.add_argument('--omega', type=float, default=4070.0)
    ap.add_argument('--k', type=float, default=3.0)
    ap.add_argument('--xi', type=float, default=0.08)
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
    t_abs_all = t_start_abs + np.arange(N, dtype=np.float64) / fs

    par = parse_par(args.par)
    f_eff, f0_epoch, v_proj, mjd0 = compute_feff(par, args.utc_start)

    print('=' * 74)
    print('[duffing singlepulse 30ms] no-folding single-pulse SNR enhancement')
    print('[series] %s: %d samples @ %.3f MHz (%.4f s)' % (
        args.series, N, fs / 1e6, N / fs))
    print('[timing] f_eff = %.9f Hz  (f0_epoch=%.9f, v_proj=%.2f m/s)' % (
        f_eff, f0_epoch, v_proj))
    print('[cfg] F=%.2f omega=%.1f k=%.1f xi=%.3f  (强耦合 Duffing v5)' % (
        args.F, args.omega, args.k, args.xi))

    # 相位标定 (折叠定位主/间脉冲, 仅用于标注)
    phi_main, phi_ip = calibrate_phase(np.asarray(series), fs, f_eff, t_start_abs)
    print('[calib] phi_main=%.4f (%d bin/1024)  phi_ip=%.4f (%d bin/1024)' % (
        phi_main, round(phi_main * 1024), phi_ip, round(phi_ip * 1024)))

    cfg = dict(CFG, F=args.F, omega=args.omega, k=args.k, xi=args.xi)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    rows = []
    t0s = t_start_abs + np.arange(args.n_seg) * (args.seg_ms * 1e-3)
    for si, seg_t0 in enumerate(t0s):
        i0 = int(round((seg_t0 - t_start_abs) * fs))
        i1 = i0 + int(round(args.seg_ms * 1e-3 * fs))
        if i1 > N:
            break
        seg = np.asarray(series[i0:i1], dtype=np.float64)
        seg_n = (seg - seg.mean()) / (seg.std() + 1e-30)
        t_abs = t_abs_all[i0:i1]
        dt = 1.0 / fs
        out, _, _, _, _ = rk4_coupled(seg_n, dt, cfg['F'], cfg['omega'],
                                      cfg['k'], cfg['xi'],
                                      cfg['x1_init'], cfg['y1_init'],
                                      cfg['x2_init'], cfg['y2_init'])
        t_main, t_ip = predicted_pulse_times(seg_t0, args.seg_ms, f_eff,
                                             phi_main, phi_ip)
        # 检测强峰 (峰/基线 > 5)
        pk_in, _ = detect_strong_peaks(seg_n)
        pk_out, _ = detect_strong_peaks(out)
        c_in, s_in = on_off_snr(seg_n, t_abs, f_eff, phi_main, phi_ip)
        c_out, s_out = on_off_snr(out, t_abs, f_eff, phi_main, phi_ip)
        rows.append(dict(seg_t0=seg_t0, c_in=c_in, s_in=s_in,
                         c_out=c_out, s_out=s_out,
                         enh_db=10 * np.log10(c_out / c_in) if c_in > 0 else np.nan))
        print('[seg %d] t0=%.3f s  contrast_in=%.3f (%.2fσ)  contrast_out=%.3f (%.2fσ)  '
              '增强=%.1f dB' % (si, seg_t0, c_in, s_in, c_out, s_out,
                                10 * np.log10(c_out / c_in) if c_in > 0 else np.nan))

        # 画图
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        t_ms = (t_abs - seg_t0) * 1e3
        axes[0].plot(t_ms, seg_n, 'b-', lw=0.6)
        for tt in t_main:
            axes[0].axvline((tt - seg_t0) * 1e3, color='red', ls='--', lw=0.7, alpha=0.8)
        for tt in t_ip:
            axes[0].axvline((tt - seg_t0) * 1e3, color='tab:orange', ls=':', lw=0.7, alpha=0.8)
        axes[0].set_ylabel('input power (z-score)')
        axes[0].set_title('Input coherent-dedispersed series  (红虚线=主脉冲, 橙点线=中间脉冲)')
        axes[0].grid(alpha=0.3)
        # 添加强峰标注 (峰/基线 > 5)
        for idx in pk_in:
            axes[0].axvline(t_ms[idx], color='green', ls='--', lw=0.5, alpha=0.6)
        axes[1].plot(t_ms, out, 'r-', lw=0.6)
        for tt in t_main:
            axes[1].axvline((tt - seg_t0) * 1e3, color='red', ls='--', lw=0.7, alpha=0.8)
        for tt in t_ip:
            axes[1].axvline((tt - seg_t0) * 1e3, color='tab:orange', ls=':', lw=0.7, alpha=0.8)
        axes[1].set_ylabel(r'Duffing output $\omega(y_1-y_2)$')
        axes[1].set_xlabel('time (ms)')
        axes[1].set_title('Strongly-coupled Duffing output  (无折叠增强)')
        axes[1].grid(alpha=0.3)
        # 添加强峰标注 (峰/基线 > 5)
        for idx in pk_out:
            axes[1].axvline(t_ms[idx], color='green', ls=':', lw=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig('%s_seg%d.png' % (args.out, si), dpi=130, bbox_inches='tight')
        plt.close(fig)

    with open(args.out + '.json', 'w', encoding='utf-8') as f:
        json.dump(dict(f_eff=f_eff, phi_main=phi_main, phi_ip=phi_ip,
                       segments=rows,
                       cfg=cfg, t_start=t_start_abs), f, indent=2,
                  ensure_ascii=False)
    print('[done] %d segments plotted; summary -> %s.json' % (len(rows), args.out))


if __name__ == '__main__':
    main()

