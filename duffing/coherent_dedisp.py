# -*- coding: utf-8 -*-
"""
coherent_dedisp.py  —  DADA 基带相干消色散管线 / baseband coherent dedispersion
读取 8-bit 实数基带 (fs=1 GHz, 500 MHz 带宽, 中心 1250 MHz) → 解析信号(复基带)
→ Hankins-Rickett(1975) 相干消色散 → 平方检波 → 降采样输出功率序列(默认 2 MHz)。

物理 / Physics:
  色散群延迟 tau(f) = D*DM/f^2 (f 为射频绝对频率 MHz, D=4.148808e3)
  消色散滤波器(相对参考频率 f_ref, 偏移 nu=f-f_ref):
      H(nu) = exp[+i*2*pi*D*DM*nu^2/(f_ref^2*(f_ref+nu))]
  其群延迟 = -(tau(f)-tau(f_ref)), 恰好抵消 ISM 相对色散延迟。

数值: dt_smear ≈ 163.7 ms → N_fft 取 2^28 (268.4 ms)。overlap-save:
  L_start=(tau_ref-tau_hi)*fs; L_end=(tau_lo-tau_ref)*fs; hop=N-L_start-L_end。

用法:
  python coherent_dedisp.py --selftest
  python coherent_dedisp.py --dur 0.6 --plot
  python coherent_dedisp.py --dur 0
"""
import argparse, json, os, time, sys
import numpy as np
try:
    import scipy.fft as sfft
    _FFT = lambda a: sfft.rfft(a, workers=-1)
    _IFFT = lambda a: sfft.ifft(a, workers=-1)
except Exception:
    _FFT = np.fft.rfft
    _IFFT = np.fft.ifft

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

DISP_D = 4.148808e3
DISP_D_HZ = DISP_D * 1.0e12


def parse_dada_header(path):
    hdr = {}
    with open(path, 'rb') as f:
        blob = f.read(4096)
    for line in blob.decode('latin1').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        p = line.split()
        if len(p) >= 2:
            hdr[p[0].strip('*').strip()] = p[1].split('#')[0].strip()
    return hdr


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


def tau_disp(f_mhz, dm):
    return DISP_D * dm / (f_mhz * f_mhz)


def build_dedisp_filter(nfft, fs, f_lo_mhz, f_hi_mhz, f_ref_mhz, dm):
    k = np.arange(nfft // 2 + 1, dtype=np.float64)
    f_rf_mhz = f_lo_mhz + k * (fs / nfft) / 1e6
    f_hz = f_rf_mhz * 1e6
    f_ref_hz = f_ref_mhz * 1e6
    # 精确消色散相位(相对 f_ref): 群延迟 = D*DM*(1/f_ref^2 - 1/f^2)
    #   phi(f) = -2*pi*D*DM*( 1/f + f/f_ref^2 - 2/f_ref ), phi(f_ref)=0
    phase = -2.0 * np.pi * DISP_D_HZ * dm * (1.0 / f_hz + f_hz / f_ref_hz ** 2 - 2.0 / f_ref_hz)
    H = np.exp(1j * phase).astype(np.complex64)
    L_start = int(np.ceil((tau_disp(f_ref_mhz, dm) - tau_disp(f_hi_mhz, dm)) * fs))
    L_end = int(np.ceil((tau_disp(f_lo_mhz, dm) - tau_disp(f_ref_mhz, dm)) * fs))
    return H, L_start, L_end


def selftest():
    print('=' * 70)
    print('[selftest] coherent dedispersion sign & direction')
    print('=' * 70)
    nfft, fs = 1 << 18, 1.0e9
    f_lo, f_hi, f_ref, dm = 1000.0, 1500.0, 1250.0, 0.05
    print('[setup] nfft=%d  smear=%.4f ms' % (nfft, (tau_disp(f_lo, dm) - tau_disp(f_hi, dm)) * 1e3))
    x = np.zeros(nfft, dtype=np.float32)
    x[nfft // 4] = 1.0
    k = np.arange(nfft // 2 + 1, dtype=np.float64)
    f_rf = f_lo + k * (fs / nfft) / 1e6
    nu_hz = (f_rf - f_ref) * 1e6
    f_ref_hz = f_ref * 1e6
    ph = 2.0 * np.pi * DISP_D_HZ * dm * nu_hz ** 2 / (f_ref_hz ** 2 * (f_ref_hz + nu_hz))
    H_ism = np.exp(1j * ph).astype(np.complex64)
    H_dd, _, _ = build_dedisp_filter(nfft, fs, f_lo, f_hi, f_ref, dm)

    def analytic(r):
        X = np.fft.rfft(r)
        Z = np.zeros(nfft, dtype=np.complex64)
        Z[0] = X[0]
        Z[1:nfft // 2] = 2.0 * X[1:nfft // 2]
        Z[nfft // 2] = X[nfft // 2]
        return Z

    Z = analytic(x)
    Z[:nfft // 2 + 1] *= H_ism
    x_disp = np.fft.ifft(Z).real.astype(np.float32)
    Z = analytic(x_disp)
    Z[:nfft // 2 + 1] *= H_dd
    z = np.fft.ifft(Z)
    p_rec = (z.real ** 2 + z.imag ** 2).astype(np.float32)

    def sharp(sig, top=50):
        return np.sort(sig ** 2)[::-1][:top].sum() / (sig ** 2).sum()

    s_orig, s_disp, s_rec = sharp(x), sharp(x_disp), sharp(p_rec)
    print('[result] energy concentration: orig=%.4f  dispersed=%.4f  recovered=%.4f'
          % (s_orig, s_disp, s_rec))
    ipk = int(np.argmax(p_rec))
    print('[result] recovered peak sample %d (impulse %d)' % (ipk, nfft // 4))
    ok = s_disp < 0.5 * s_orig and s_rec > 2 * s_disp and s_rec > 0.3 * s_orig
    print('[selftest]', 'PASS' if ok else 'FAIL')
    return ok


def coherent_dedisp_stream(args):
    hdr = parse_dada_header(args.hdr)
    par = parse_par(args.par)
    fs = 1.0e6 / float(hdr['TSAMP'])
    f_lo = float(hdr['FREQ']) - float(hdr['BW']) / 2.0
    f_hi = float(hdr['FREQ']) + float(hdr['BW']) / 2.0
    f_ref = f_hi   # 参考频带顶端, 与 akilli_core 主/间脉冲相位约定一致
    dm = float(par.get('DM', 71.0237))
    tau_smear = tau_disp(f_lo, dm) - tau_disp(f_hi, dm)

    nfft = 1
    while nfft < int(tau_smear * fs) + 1:
        nfft <<= 1
    H, L_start, L_end = build_dedisp_filter(nfft, fs, f_lo, f_hi, f_ref, dm)
    hop = nfft - L_start - L_end

    dur_total = os.path.getsize(args.dada) / float(hdr.get('BYTES_PER_SECOND', fs))
    t_obs = min(args.dur, dur_total) if args.dur > 0 else dur_total
    f_out = args.f_out
    n_out = int(np.ceil(t_obs * f_out)) + 2
    dec = max(1, int(round(fs / f_out)))

    print('=' * 70)
    print('[coherent dedisp] fs=%.4g Hz  band=%.0f-%.0f MHz  DM=%.4f' % (fs, f_lo, f_hi, dm))
    print('[disp] tau_smear=%.4f ms' % (tau_smear * 1e3))
    print('[fft] N=%d (%.3f ms)  L_start=%d L_end=%d hop=%d eff=%.1f%%' % (
        nfft, nfft / fs * 1e3, L_start, L_end, hop, 100.0 * hop / nfft))
    print('[out] t_obs=%.4f s  f_out=%.4g Hz  dec=%d' % (t_obs, f_out, dec))

    series = np.zeros(n_out, dtype=np.float64)
    series_cnt = np.zeros(n_out, dtype=np.float64)
    t0 = time.time()
    fh = open(args.dada, 'rb')
    base_byte = 4096 + int(args.t_start * fs)
    out_pos, n_blk = 0, 0
    carry = np.zeros(0, dtype=np.float32)
    while out_pos < n_out - 1:
        # overlap-save: 每块从 base_byte + n_blk*hop 读取 nfft 字节(与前一块重叠 L_end)
        fh.seek(base_byte + n_blk * hop, os.SEEK_SET)
        raw = fh.read(nfft)
        if len(raw) < nfft:
            break
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        R = _FFT(x)
        Z = np.zeros(nfft, dtype=np.complex64)
        Z[0] = R[0]
        Z[1:nfft // 2] = 2.0 * R[1:nfft // 2]
        Z[nfft // 2] = R[nfft // 2]
        Z[:nfft // 2 + 1] *= H
        z = _IFFT(Z)
        p = (z.real ** 2 + z.imag ** 2).astype(np.float32)
        pv = p[L_start:nfft - L_end]
        # 跨块携带降采样余数, 避免每块丢弃 hop%dec 采样导致时间轴压缩
        pv = np.concatenate([carry, pv])
        nv = pv.size - pv.size % dec
        down = pv[:nv].reshape(-1, dec).mean(axis=1).astype(np.float64)
        carry = np.ascontiguousarray(pv[nv:])
        n_add = min(down.size, n_out - 1 - out_pos)
        series[out_pos:out_pos + n_add] += down[:n_add]
        series_cnt[out_pos:out_pos + n_add] += 1.0
        out_pos += n_add
        n_blk += 1
        if n_blk % 5 == 0:
            print('[proc] block %d  t=%.3f/%.3f s  %.1f s' % (
                n_blk, out_pos / f_out, t_obs, time.time() - t0), flush=True)
    fh.close()
    series = np.divide(series, np.maximum(series_cnt, 1.0),
                       out=np.zeros_like(series), where=series_cnt > 0)
    series = series[:out_pos].astype(np.float32)
    meta = dict(fs=fs, f_lo_mhz=f_lo, f_hi_mhz=f_hi, f_ref_mhz=f_ref, dm=dm,
                tau_smear_s=tau_smear, nfft=nfft, L_start=L_start, L_end=L_end,
                hop=hop, f_out=f_out, t_start=args.t_start, t_obs=out_pos / f_out,
                n_out=int(series.size), dec=dec)
    print('[done] %d samples (%.4f s) in %.1f s' % (series.size, series.size / f_out,
                                                     time.time() - t0))
    return series, meta, par


def plot_series(series, f_out, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    t = np.arange(series.size) / f_out
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax[0].plot(t * 1e3, series, lw=0.4, color='k')
    ax[0].set_ylabel('dedispersed power (a.u.)')
    ax[0].set_title('Coherent-dedispersed power series')
    ax[0].grid(alpha=0.3)
    n = min(series.size, int(10e-3 * f_out))
    ax[1].plot(t[:n] * 1e3, series[:n], lw=0.7, color='tab:red')
    ax[1].set_xlabel('time (ms)')
    ax[1].set_title('zoom: first 10 ms')
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('[plot] saved', out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dada', default=r'D:\VBshare\fast_1937+21\bcmin1937+21.dada')
    ap.add_argument('--hdr', default=r'D:\VBshare\fast_1937+21\bcmin1937+21.hdr')
    ap.add_argument('--par', default=r'D:\VBshare\fast_1937+21\B1937+21.par')
    ap.add_argument('--dur', type=float, default=0.0)
    ap.add_argument('--t-start', type=float, default=0.0)
    ap.add_argument('--f-out', type=float, default=2.0e6)
    ap.add_argument('--out', default='out_coherent_series')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    series, meta, _ = coherent_dedisp_stream(args)
    np.save(args.out + '.npy', series)
    with open(args.out + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print('[save] %s.npy (%d samples) + %s.json' % (args.out, series.size, args.out))
    if args.plot:
        plot_series(series, meta['f_out'], args.out + '.png')


if __name__ == '__main__':
    main()
