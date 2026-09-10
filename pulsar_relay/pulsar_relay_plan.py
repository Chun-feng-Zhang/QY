# -*- coding: utf-8 -*-
"""
=============================================================================
 双40米射电望远镜 · 脉冲星共视接力24小时不间断观测方案
=============================================================================
 台站1: 陕西省商洛市洛南县保安镇       34.170851°N, 109.951457°E, 海拔≈1080 m
 台站2: 浙江省富阳(用户提供坐标)     29.934176°N, 119.829719°E, 海拔≈30 m

 输入: 脉冲星列表文件(每行一个J名称, 如 J0437-4715; 赤经=HH:MM, 赤纬=±DD:MM)

 功能:
   1) 用 astropy 计算每颗脉冲星在两地的升起/落下(高度角-时间)曲线;
   2) 计算"共视"接力观测方案: 同一时刻两台望远镜观测同一颗脉冲星,
      任一台看不到(高度角低于该方位遮挡角)即切换下一颗;
      目标24小时不间断; 若不可行则给出最长覆盖方案;
      并枚举多种完整接力方案(展示前3种, 均含J0437-4715, 按平均高度角排序);
   3) 计算满足该接力方案的"杭州全方位 方位-遮挡角上限"要求曲线
      (洛南遮挡剖面默认全周均匀取7°, 即已知的"正南7°", 可自定义);
   4) 输出图形:
      fig1_riseset_curves.png   两地脉冲星升起落下曲线图
      fig2_relay_plan.png       共视接力观测规划图(Gantt, 3种含J0437方案)
      fig3_obstruction_req.png  两地DEM地形遮挡角下限图(极坐标+直角坐标)
      fig4_coverage_vs_obs.png  最大覆盖时长 vs 杭州均匀遮挡角(辅助)
      fig5_altaz_tracks.png    两地所有脉冲星的方位-俯仰(alt-az)轨迹图
      fig6_azel_time_Luonan.png   洛南 全部脉冲星 俯仰-时间(上)/方位-时间(下)复合图(每台站1张)
      fig6_azel_time_Hangzhou.png 杭州 全部脉冲星 俯仰-时间(上)/方位-时间(下)复合图(每台站1张)
      relay_schedule.csv        接力排班表(北京时间/UTC/脉冲星/两地高度角方位角)
      relay_plans.csv           多种接力方案一览表(每段一行)
      hangzhou_obstruction_req.csv  杭州各方位遮挡角上限表
      luonan_dem_obstruction.csv    洛南DEM地形遮挡角曲线表(启用DEM时输出, 每方位一行)
      hangzhou_dem_obstruction.csv  杭州DEM地形遮挡角曲线表(启用DEM时输出, 每方位一行)

 用法:
   python pulsar_relay_plan.py
   python pulsar_relay_plan.py --pulsars 10-MSP.txt --date 2026-08-20 \
        --luonan-obs 0.0 --hangzhou-obs 0.0 --min-elev 5.0 \
        --margin 0.0 --strategy longest
   # 叠加DEM地形遮挡(自动查找同目录 ASTGTM_N34E109*.img / ASTGTM_N29E119*.img):
   python pulsar_relay_plan.py --pulsars 10-MSP.txt \
        --luonan-obs 7 --hangzhou-obs 15

 说明:
   * 观测日期默认取脚本运行当天(北京时间). 恒星日比平太阳日短约3m56s,
     同一方案每天提前约3m56s, 实际排班请以观测日为准重新运行。
   * 望远镜存在俯仰极限: --min-elev 为最低仰角(默认5°), 两侧相同;
     有效地平 = max(地形遮挡角, 俯仰下限), 源必须高于 有效地平+余量。
   * 可选DEM地形遮挡: 用 --luonan-dem/--hangzhou-dem 指定 ASTER GDEM 高程文件
     (.img/.tif, 需 pip install rasterio); 缺省自动查找脚本目录中覆盖台站1°格及
     8邻格的 ASTGTM_*.img(.zip自动解压)。对每个方位角沿视线采样地形, 视线仰角
     含地球曲率与4/3大气折射修正; 波束FWHM默认0.3°(--beam-fwhm), 判定: 脉冲星
     位于波束中央, 波束下边缘与地形最高处初次相切即遮挡下限=地形遮挡角+FWHM/2;
     最终剖面: 默认逐方位取 max(均匀/CSV基准, DEM下限)(基准保底; 需以DEM曲线为主时
     用 --dem-mode replace, 仅无DEM数据方位用基准)。DEM栅格可为投影坐标(如UTM)。
   * 洛南遮挡剖面仅知"正南方向7°", 默认全周取7°; 如有实测数据,
     可用 --luonan-profile az_obs.csv 提供完整剖面(两列: 方位角, 遮挡角)。
   * 杭州遮挡剖面默认取0°(理想), 脚本据此算出"杭州各方位遮挡角上限":
     实际选址要求杭州各方位遮挡角严格低于该曲线(工程上建议再留2°以上余量)。
   * 切换策略 --strategy longest: 在当前共视脉冲星不可见时, 切换至"当时共视
     且剩余可观测时间最长"的脉冲星, 该策略可使覆盖时长达到理论最大
     (=所有共视窗口的并集); --strategy list: 严格按列表顺序切换。
=============================================================================
"""

import argparse
import csv
import datetime as dt
import math
import re
import shutil
import sys
import zipfile
from pathlib import Path

import astropy.units as u
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

# ---------------- 全局配置 ----------------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

# 控制台 UTF-8 输出(避免 Windows 默认代码页下中文乱码)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT_DIR = Path(__file__).resolve().parent

SITES = {
    "Luonan":   {"cn": "洛南(保安镇)",      "lat": 34.17085105, "lon": 109.95145724, "height": 1080.0},
    "Hangzhou": {"cn": "杭州(富阳)",       "lat": 29.934176, "lon": 119.829719, "height": 30.0},
}


# ---------------- 工具函数 ----------------
def parse_pulsar(name):
    """J0437-4715 -> (RA小时, Dec度)"""
    m = re.fullmatch(r"J(\d{2})(\d{2})([+-])(\d{2})(\d{2})", name)
    if not m:
        raise ValueError(f"无法解析脉冲星名称: {name!r} (应为 JHHMM±DDMM)")
    ra = int(m.group(1)) + int(m.group(2)) / 60.0
    dec = (int(m.group(4)) + int(m.group(5)) / 60.0) * (1 if m.group(3) == "+" else -1)
    return ra, dec


def load_obs_profile(uniform_deg, csv_path=None):
    """遮挡剖面: 返回360个值(每方位角1度). 默认均匀; 或从CSV(az,obs)插值."""
    if csv_path:
        az, obs = [], []
        with open(csv_path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"[,;\t ]+", line)
                if len(parts) < 2:
                    continue
                try:
                    az.append(float(parts[0]) % 360.0)
                    obs.append(float(parts[1]))
                except ValueError:
                    continue
        if not az:
            raise ValueError(f"遮挡剖面文件为空或格式错误: {csv_path}")
        a = np.array(az)
        o = np.array(obs)
        order = np.argsort(a)
        return np.interp(np.arange(360.0), a[order], o[order], period=360.0)
    return np.full(360, float(uniform_deg))


def obs_at(profile, az_deg):
    """按方位角(度)查遮挡角, 支持数组"""
    idx = np.round(np.asarray(az_deg, dtype=float)).astype(int) % 360
    return profile[idx]


# ================= DEM 地形遮挡(逐方位) =================
# 几何模型: 40米望远镜波束(FWHM 默认0.3°, 脉冲星在波束最中央), 当波束下边缘恰与
#   视线上地形最高处初次相切时源开始可见 -> 该方位遮挡下限 = 地形视线仰角 + FWHM/2。
#   视线仰角考虑地球曲率与标准 4/3 大气折射: 视落差 drop = d^2 / (2 * (4/3) * R)。
#   读取 ASTER GDEM(ERDAS HFA .img / GeoTIFF 均可, 经 GDAL/rasterio); 缺省自动查找
#   脚本目录中覆盖台站 1°格及 8 邻格的 ASTGTM_*.img(.zip 自动解压)。
R_EARTH_M = 6371000.0
REFR_K = 4.0 / 3.0
DEM_STEP_M = 30.0       # 视线采样间距(m), 与 ASTER 1"≈30m 分辨率相当
DEM_D0_M = 500.0        # 采样起点(m): 跳过台站近场(近场30m格网像元含台站建筑/树冠,
                        #   会形成数十度的假遮挡峰); 需恢复旧行为可设 --dem-d0 60
DEM_RMAX_KM = 100.0     # 视线最大半径: 更远处地形被地球曲率压平, 不再构成遮挡


def _ensure_dem_file(path):
    """*.img 直接返回; 若只有 *.img.zip 则自动解压; 找不到返回 None"""
    if path is None:
        return None
    p = Path(path)
    low = str(p).lower()
    if low.endswith(".img"):
        if p.is_file():
            return p
        zp = Path(str(p) + ".zip")
    elif low.endswith(".zip"):
        zp = p
    else:
        return None
    if zp is not None and zp.is_file():
        try:
            with zipfile.ZipFile(zp) as z:
                m = z.infolist()[0]
                out = OUT_DIR / m.filename
                if not out.exists():
                    z.extract(m, OUT_DIR)
                return out
        except Exception as e:
            print(f"  [警告] 解压DEM失败 {zp.name}: {e}")
            return None
    return None


def _auto_dem_files(site):
    """自动查找覆盖台站1°格及其8邻格的 ASTGTM_*.img / .img.zip; 同一1°格去重"""
    lat0, lon0 = int(math.floor(site["lat"])), int(math.floor(site["lon"]))
    hits = []
    seen = set()
    for dla in (-1, 0, 1):
        for dlo in (-1, 0, 1):
            base = f"ASTGTM_N{lat0 + dla:02d}E{lon0 + dlo:03d}"
            for pat in (base + "*.img", base + "*.img.zip"):
                for p in sorted(OUT_DIR.glob(pat)):
                    m = re.match(r"(ASTGTM_N\d{2}E\d{3})", p.name)
                    key = m.group(1) if m else p.name
                    if key in seen:
                        continue
                    seen.add(key)
                    pp = _ensure_dem_file(p)
                    if pp:
                        hits.append(pp)
    return hits


def _load_dem(path):
    """读取DEM栅格 -> (z, tr, crs, bounds, name); 高程无效值置NaN。
    crs 可能为 UTM(如EPSG:32649/32650)或经纬度, 采样时统一用 rasterio.warp 转换。"""
    import rasterio
    with rasterio.open(path) as src:
        z = src.read(1).astype(np.float64)
        nod = src.nodata
        if nod is not None:
            z = np.where(z == nod, np.nan, z)
        if nod == 0.0:   # ASTER 常以 0 表无效/填海(如内陆块)
            z = np.where(z == 0.0, np.nan, z)
        z[z < -1000.0] = np.nan   # ASTER 亦可能以 -9999 表无效
        return (z, src.transform, src.crs,
                (src.bounds.left, src.bounds.bottom,
                 src.bounds.right, src.bounds.top), Path(path).name)


def _bilinear(z, tr, crs, lon, lat):
    """双线性采样DEM高程(经纬度->投影->像元); 越界/无效返回NaN。lon/lat可为数组。"""
    from rasterio.warp import transform as _warp
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    xs, ys = _warp("EPSG:4326", crs, lon, lat)
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    col = (xs - tr.c) / tr.a
    row = (ys - tr.f) / tr.e
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    h, w = z.shape
    ok = (c0 >= 0) & (c0 + 1 < w) & (r0 >= 0) & (r0 + 1 < h)
    fx = col - c0
    fy = row - r0
    cc = np.clip(c0, 0, w - 2)
    rr = np.clip(r0, 0, h - 2)
    z00 = z[rr, cc]
    z01 = z[rr, cc + 1]
    z10 = z[rr + 1, cc]
    z11 = z[rr + 1, cc + 1]
    out = (z00 * (1 - fx) + z01 * fx) * (1 - fy) + (z10 * (1 - fx) + z11 * fx) * fy
    return np.where(ok, out, np.nan)


def _geo_forward(lat0, lon0, az_deg, dists):
    """球面大圆前进: 由台站(lat0,lon0,度)沿方位 az_deg 走 dists(m) -> (lat1,lon1,弧度)"""
    lat0 = math.radians(lat0)
    lon0 = math.radians(lon0)
    azr = math.radians(az_deg)
    d = np.asarray(dists, float) / R_EARTH_M
    sd = np.sin(d)
    cd = np.cos(d)
    cl0 = math.cos(lat0)
    sl0 = math.sin(lat0)
    lat1 = np.arcsin(sl0 * cd + cl0 * sd * math.cos(azr))
    lon1 = lon0 + np.arctan2(math.sin(azr) * sd * cl0, cd - sl0 * np.sin(lat1))
    return lat1, lon1


def _site_h0(site, loaded, ant_height=15.0):
    """台站相位中心海拔 = 实测台站海拔(site['height']) + 天线轴高(--ant-height)。
    另返回 DEM 台站像素高程(仅作参考输出, 可能为 None)。"""
    hs = []
    for (z, tr, crs, bnd, nm) in loaded:
        hh = _bilinear(z, tr, crs, np.array([site["lon"]]), np.array([site["lat"]]))[0]
        if not np.isnan(hh):
            hs.append(float(hh))
    hdem = float(np.mean(hs)) if hs else None
    h0 = float(site.get("height", 0.0)) + float(ant_height)
    return h0, hdem


def dem_obstruction_profiles(dem_arg, key, args):
    """对台站逐方位(0-359°)计算地形遮挡角(几何, ≥0, 不含波束/均匀下限)。
    返回 dict; 无任何DEM可用时返回 None。
      terrain : 几何遮挡角(度, 含曲率与4/3折射; 无数据方位=NaN)
      withbeam: terrain + args.beam_fwhm/2 (NaN 同 terrain)
      dist_km/npts: 每方位视线采样最大距离(km)/有效点数
    """
    site = SITES[key]
    auto_mode = dem_arg is None
    if auto_mode:
        files = _auto_dem_files(site)
    else:
        if str(dem_arg).strip().lower() in ("none", "off", "0"):
            return None
        files = []
        for part in str(dem_arg).replace(",", ";").split(";"):
            part = part.strip()
            if not part:
                continue
            pp = _ensure_dem_file(Path(part))
            if pp:
                files.append(pp)
            else:
                print(f"  [警告] 找不到DEM文件: {part}")
    if not files:
        return None
    loaded = []
    for fp in files:
        try:
            loaded.append(_load_dem(fp))
        except Exception as e:
            print(f"  [警告] 读取DEM失败 {fp.name}: {e}")
    if not loaded:
        return None

    h0, hdem = _site_h0(site, loaded, args.ant_height)
    d0_m = float(getattr(args, "dem_d0", DEM_D0_M))
    rmax_km = float(getattr(args, "dem_rmax", DEM_RMAX_KM))
    ds = np.arange(d0_m, rmax_km * 1000.0, DEM_STEP_M)
    drop = ds * ds / (2.0 * REFR_K * R_EARTH_M)   # 地球曲率视落差(含折射)
    obs = np.full(360, np.nan)
    dist = np.zeros(360)
    npts = np.zeros(360, dtype=int)
    for (z, tr, crs, bnd, nm) in loaded:
        for az in range(360):
            la, lo = _geo_forward(site["lat"], site["lon"], az + 0.0, ds)
            zs = _bilinear(z, tr, crs, np.degrees(lo), np.degrees(la))
            good = ~np.isnan(zs)
            if not good.any():
                continue
            el = np.degrees(np.arctan((zs - h0 - drop) / ds))
            elv = el[good]
            obs[az] = np.fmax(obs[az], float(elv.max()))
            dist[az] = np.fmax(dist[az], float(ds[good].max() / 1000.0))
            npts[az] += int(good.sum())
    obs = np.where(np.isnan(obs), np.nan, np.maximum(obs, 0.0))
    withbeam = np.where(np.isnan(obs), np.nan, obs + args.beam_fwhm / 2.0)
    missing = []
    if auto_mode:
        la0, lo0 = int(math.floor(site["lat"])), int(math.floor(site["lon"]))
        need = {f"ASTGTM_N{la0 + dla:02d}E{lo0 + dlo:03d}"
                for dla in (-1, 0, 1) for dlo in (-1, 0, 1)}
        have = set()
        for p in files:
            m = re.match(r"(ASTGTM_N\d{2}E\d{3})", p.name)
            if m:
                have.add(m.group(1))
        missing = sorted(need - have)
    return {"site": key, "auto": auto_mode, "missing": missing,
            "names": [nm for (_, _, _, _, nm) in loaded],
            "h_site": hdem, "h0": h0,
            "terrain": obs, "withbeam": withbeam,
            "dist_km": dist, "npts": npts}


def contiguous_runs(mask):
    """返回连续True区间的 (起点, 终点) 列表(索引)"""
    runs, i, T = [], 0, len(mask)
    while i < T:
        if mask[i]:
            j = i
            while j + 1 < T and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def greedy_schedule(common, run_end, strategy):
    """接力排班: 每时刻若当前脉冲星不可共视, 切换至候选中共视剩余最长的脉冲星.
    返回 schedule[t] = 脉冲星索引 或 -1(空隙)."""
    N, T = common.shape
    sched = np.full(T, -1, dtype=int)
    t = 0
    while t < T:
        cand = np.where(common[:, t])[0]
        if cand.size == 0:
            t += 1
            continue
        if strategy == "longest":
            p = int(cand[np.argmax(run_end[cand, t])])
        else:  # 严格列表顺序
            p = int(cand[0])
        e = int(run_end[p, t])
        sched[t:e + 1] = p
        t = e + 1
    return sched


def hhmm(dt_obj):
    return dt_obj.strftime("%H:%M")


# ---------------- 主程序 ----------------
def main():
    ap = argparse.ArgumentParser(description="双40米射电望远镜脉冲星共视接力观测方案")
    ap.add_argument("--pulsars", default="10-MSP.txt", help="脉冲星列表文件(每行一个J名称)")
    ap.add_argument("--date", default=dt.date.today().isoformat(),
                    help="观测日期(北京时间, YYYY-MM-DD), 默认今天")
    ap.add_argument("--step-min", type=float, default=1.0, help="时间步长(分钟), 默认1")
    ap.add_argument("--luonan-obs", type=float, default=0.0,
                    help="洛南均匀遮挡角(度), 默认7(=已知正南遮挡角)")
    ap.add_argument("--luonan-profile", default=None,
                    help="洛南遮挡剖面CSV(az,obs), 覆盖 --luonan-obs")
    ap.add_argument("--hangzhou-obs", type=float, default=0.0,
                    help="杭州假设均匀遮挡角(度), 默认0(理想地平)")
    ap.add_argument("--hangzhou-profile", default=None,
                    help="杭州遮挡剖面CSV(az,obs), 覆盖 --hangzhou-obs")
    ap.add_argument("--luonan-dem", default=None,
                    help="洛南DEM地形高程文件(ASTER GDEM .img/.tif); 缺省自动查找目录内覆盖台站1°格及8邻格的ASTGTM_*.img; 多个用;分隔; 传 none 禁用")
    ap.add_argument("--hangzhou-dem", default=None,
                    help="杭州DEM地形高程文件(同上); 缺省自动查找; 传 none 禁用")
    ap.add_argument("--beam-fwhm", type=float, default=0.3,
                    help="40米望远镜波束半高全宽(度), 默认0.3; 遮挡下限=地形角+波束半宽(FWHM/2)")
    ap.add_argument("--ant-height", type=float, default=15.0,
                    help="望远镜相位中心离地高度(m), 默认15(40米天线仰轴近似)")
    ap.add_argument("--dem-mode", choices=["max", "replace"], default="max",
                    help="DEM与均匀/CSV基准的合成: max=逐方位取较大(基准保底, 默认); "
                         "replace=以DEM曲线为准, 仅无数据方位用基准")
    ap.add_argument("--dem-d0", type=float, default=500.0,
                    help="DEM视线采样起点(m), 默认500: 跳过台站近场(近场像元含建筑/树冠,"
                         "会形成数十度假遮挡峰); 设60可恢复旧行为")
    ap.add_argument("--dem-rmax", type=float, default=100.0,
                    help="DEM视线最大采样半径(km), 默认100; 更远处地形被曲率压平")
    ap.add_argument("--min-elev", type=float, default=5.0,
                    help="望远镜最低仰角/俯仰极限(度), 两侧相同, 默认5; 设0可关闭")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="遮挡角要求余量(度), 默认0; 工程上建议≥2")
    ap.add_argument("--strategy", choices=["longest", "list"], default="longest",
                    help="切换策略: longest=剩余共视最长(覆盖最优); list=按列表顺序")
    args = ap.parse_args()

    # ---------- 1. 台站 ----------
    locs = {
        k: EarthLocation(lat=s["lat"] * u.deg, lon=s["lon"] * u.deg, height=s["height"] * u.m)
        for k, s in SITES.items()
    }

    # ---------- 2. 时间网格: 北京时间 00:00 ~ 次日 00:00 ----------
    d0 = dt.date.fromisoformat(args.date)
    start_utc = Time(f"{(d0 - dt.timedelta(days=1))} 16:00:00", scale="utc")  # = 北京当天00:00
    minutes = np.arange(0.0, 24 * 60 + 1e-6, args.step_min)
    times = start_utc + minutes * u.min
    T = len(times)
    bj_dt = Time(times + 8 * u.hour).to_datetime()  # 北京时间 datetime 数组

    # ---------- 3. 脉冲星 ----------
    txt = Path(args.pulsars).read_text(encoding="utf-8")
    names = [ln.strip() for ln in txt.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    ra_hrs = np.zeros(len(names))
    dec_deg = np.zeros(len(names))
    for i, n in enumerate(names):
        ra_hrs[i], dec_deg[i] = parse_pulsar(n)
    coord = SkyCoord(ra=ra_hrs * u.hourangle, dec=dec_deg * u.deg, frame="icrs")
    N = len(names)
    print(f"[脉冲星] 共 {N} 颗: {', '.join(names)}")

    # ---------- 4. 高度角 / 方位角 ----------
    # 逐颗脉冲星变换再堆叠, 避免 astropy 高维广播兼容问题
    alt, az = {}, {}
    for k, loc in locs.items():
        alt_k = np.empty((N, T))
        az_k = np.empty((N, T))
        for p in range(N):
            aa = coord[p].transform_to(AltAz(obstime=times, location=loc))
            alt_k[p] = aa.alt.deg
            az_k[p] = aa.az.deg
        alt[k] = alt_k
        az[k] = az_k

    # ---------- 5. 遮挡剖面: 均匀/CSV基准 与 DEM地形曲线(逐方位)取大 ----------
    base_ln = load_obs_profile(args.luonan_obs, args.luonan_profile)
    base_hz = load_obs_profile(args.hangzhou_obs, args.hangzhou_profile)
    dem_ln = dem_obstruction_profiles(args.luonan_dem, "Luonan", args)
    dem_hz = dem_obstruction_profiles(args.hangzhou_dem, "Hangzhou", args)
    if dem_ln is not None:
        dmsk = np.isnan(dem_ln["withbeam"])
        dval = dem_ln["withbeam"]
        prof_ln = np.where(dmsk, base_ln,
                           (np.maximum(base_ln, dval) if args.dem_mode == "max" else dval))
    else:
        prof_ln = base_ln
    if dem_hz is not None:
        dmsk = np.isnan(dem_hz["withbeam"])
        dval = dem_hz["withbeam"]
        prof_hz = np.where(dmsk, base_hz,
                           (np.maximum(base_hz, dval) if args.dem_mode == "max" else dval))
    else:
        prof_hz = base_hz

    # ---------- 6. 可见性与共视 ----------
    # 有效地平 = max(地形遮挡角, 望远镜俯仰下限); 源必须高于 有效地平+余量
    eff_ln = np.maximum(obs_at(prof_ln, az["Luonan"]), args.min_elev)
    eff_hz = np.maximum(obs_at(prof_hz, az["Hangzhou"]), args.min_elev)
    vis_ln = alt["Luonan"] > eff_ln + args.margin
    vis_hz = alt["Hangzhou"] > eff_hz + args.margin
    common = vis_ln & vis_hz  # (N, T) 共视

    # ---------- 7. 接力排班 ----------
    run_end = np.zeros((N, T), dtype=int)
    for p in range(N):
        e = T - 1
        for t in range(T - 1, -1, -1):
            if not common[p, t]:
                e = t - 1
            run_end[p, t] = e if common[p, t] else -1

    sched = greedy_schedule(common, run_end, args.strategy)
    covered = sched >= 0
    cov_frac = covered.mean()
    union_frac = common.any(axis=0).mean()  # 理论最大覆盖(共视窗口并集)

    # 空隙
    gaps = []
    t = 0
    while t < T:
        if not covered[t]:
            j = t
            while j + 1 < T and not covered[j + 1]:
                j += 1
            gaps.append((t, j))
            t = j + 1
        else:
            t += 1

    # 切换序列
    segments = []
    t = 0
    while t < T:
        p = sched[t]
        if p < 0:
            t += 1
            continue
        e = t
        while e + 1 < T and sched[e + 1] == p:
            e += 1
        segments.append((t, e, p))
        t = e + 1

    # ---------- 7.5 枚举多种完整接力方案(均含 J0437-4715) ----------
    def _scheme_metrics(segs):
        """方案统计: 返回 (覆盖比例, 平均高度角[仅内部排序用], 洛南最低高度角, 杭州最低高度角).
        最低高度角 = 观测段内脉冲星在**该站**实际达到的最低俯仰角, 受各自有效地平约束:
        洛南侧 ≥ 洛南遮挡角(7°), 杭州侧 ≥ 俯仰下限(杭州遮挡0°时=5°), 两站分开标注不混淆."""
        covered = np.zeros(T, dtype=bool)
        s_alt = 0.0
        s_cnt = 0
        ln_mn = 1e9
        hz_mn = 1e9
        for (a, b, p) in segs:
            covered[a:b + 1] = True
            altm = np.minimum(alt["Luonan"][p, a:b + 1],
                              alt["Hangzhou"][p, a:b + 1])
            s_alt += float(altm.sum())
            s_cnt += altm.size
            m_ln = float(alt["Luonan"][p, a:b + 1].min())
            m_hz = float(alt["Hangzhou"][p, a:b + 1].min())
            if m_ln < ln_mn:
                ln_mn = m_ln
            if m_hz < hz_mn:
                hz_mn = m_hz
        return (float(covered.mean()),
                (s_alt / s_cnt if s_cnt else 0.0),
                (ln_mn if s_cnt else 0.0),
                (hz_mn if s_cnt else 0.0))

    def _enumerate_schedules():
        """枚举所有覆盖=理论最大覆盖(共视窗口并集)的接力方案,
        返回按覆盖降序、平均高度角降序排列的 [(segments, cov, mean_alt, min_alt), ...]."""
        found = {}

        def dfs(t, cur, segs):
            if t >= T:
                sig = tuple(segs)
                if sig not in found:
                    found[sig] = (list(segs),) + _scheme_metrics(segs)
                return
            if cur >= 0:  # 当前脉冲星观测到其共视run结束
                e = run_end[cur, t]
                if e < t:
                    return
                dfs(min(e + 1, T), -1, segs + [(t, min(e, T - 1), cur)])
                return
            # 需要选下一颗: 当前时刻共视且run可延伸的源
            cands = [p for p in range(N) if common[p, t] and run_end[p, t] > t]
            if not cands:  # 无源可用: 跳到下一共视时刻(空隙)
                nxt = None
                for tt in range(t + 1, T):
                    if any(common[p, tt] and run_end[p, tt] > tt for p in range(N)):
                        nxt = tt
                        break
                dfs(T if nxt is None else nxt, -1, segs)
                return
            for p in cands:
                dfs(t, p, segs)

        dfs(0, -1, [])
        best = max((v[1] for v in found.values()), default=0.0)
        keep = [v for v in found.values() if v[1] >= best - 2.0 / T]
        keep.sort(key=lambda v: (-v[1], -v[2]))
        return keep, best

    all_schemes, best_union_cov = _enumerate_schedules()
    cur_cov, cur_malt, cur_lnm, cur_hzm = _scheme_metrics(segments)
    j0437 = next((i for i, n in enumerate(names) if n.startswith("J0437")), None)

    def _chain_key(segs):
        last = segs[-1][2]
        return (frozenset(p for (a, b, p) in segs) - {last}, last)

    # 展示的方案均要求包含 J0437-4715(用户要求); 当前排班若不包含,
    # 则方案1改用"含J0437-4715的完整方案中平均高度角最高者"
    pool = ([s for s in all_schemes if any(p == j0437 for (a, b, p) in s[0])]
            if j0437 is not None else list(all_schemes))
    cur_has = (j0437 is None) or any(p == j0437 for (a, b, p) in segments)
    if cur_has:
        ordered = [("当前", segments, cur_cov, cur_lnm, cur_hzm)]
    else:
        s0 = pool[0] if pool else None
        ordered = ([("含J0437最优", s0[0], s0[1], s0[3], s0[4])] if s0 is not None
                   else [("当前", segments, cur_cov, cur_lnm, cur_hzm)])
    rest = [s for s in pool if tuple(s[0]) != tuple(ordered[0][1])]
    picked = {_chain_key(ordered[0][1])}
    for s in rest:  # pool 已按(覆盖, 平均高度角)降序; 平均高度角仅用于排序选方案, 不对外展示
        if len(ordered) >= 3:
            break
        ck = _chain_key(s[0])
        if ck not in picked:
            ordered.append((f"备选{len(ordered)}", s[0], s[1], s[3], s[4]))
            picked.add(ck)
    for s in rest:  # 仍不足3种时直接补足
        if len(ordered) >= 3:
            break
        if all(tuple(x[1]) != tuple(s[0]) for x in ordered):
            ordered.append((f"备选{len(ordered)}", s[0], s[1], s[3], s[4]))
    n_schemes = len(all_schemes)
    n_with0437 = len(pool)

    # ---------- 8. 杭州各方位遮挡角上限 ----------
    req = np.full(360, np.nan)
    req_src = np.full(360, -1, dtype=int)  # 各方位上限由哪颗脉冲星决定(索引)
    for t in range(T):
        p = sched[t]
        if p < 0:
            continue
        a = int(round(az["Hangzhou"][p, t])) % 360
        v = alt["Hangzhou"][p, t] - args.margin
        if np.isnan(req[a]) or v < req[a]:
            req[a] = v
            req_src[a] = p
    azg = np.arange(360)
    valid = ~np.isnan(req)

    # ---------- 9. 灵敏度: 杭州均匀遮挡角 -> 最大覆盖 ----------
    hs = np.arange(0.0, 30.01, 0.5)
    covs = np.array([(vis_ln & (alt["Hangzhou"] > np.maximum(h, args.min_elev) + args.margin)).any(axis=0).mean()
                     for h in hs])
    ok24 = hs[covs >= 1.0 - 1e-9]
    h24 = float(ok24.max()) if ok24.size else None

    # ================= 文本摘要 =================
    print("\n" + "=" * 78)
    print("双40米射电望远镜 脉冲星共视接力观测方案")
    print("=" * 78)
    for k, s in SITES.items():
        print(f"  {s['cn']:<14s} 经度 {s['lon']:.5f}°E  纬度 {s['lat']:.5f}°N  海拔 {s['height']:.0f} m")
    print(f"  观测日期(北京时间): {args.date} 00:00 ~ {args.date} 23:59   "
          f"(步长 {args.step_min:g} min, 共 {T} 点)")
    print(f"  洛南遮挡剖面: " + ("自定义CSV" if args.luonan_profile else f"均匀基准 {args.luonan_obs:.1f}°")
          + (f" + DEM地形(逐方位, {dem_ln['names'][0]})" if dem_ln is not None else ""))
    print(f"  杭州遮挡剖面: " + ("自定义CSV" if args.hangzhou_profile else f"均匀基准 {args.hangzhou_obs:.1f}°")
          + (f" + DEM地形(逐方位, {dem_hz['names'][0]})" if dem_hz is not None else ""))
    print(f"  望远镜俯仰下限(最低仰角): {args.min_elev:.1f}° (两侧相同, 有效地平=max(遮挡,俯仰下限))")
    if dem_ln is not None or dem_hz is not None:
        print(f"  DEM地形遮挡: 几何视线含地球曲率+4/3折射; 波束FWHM {args.beam_fwhm:.2f}° -> "
              f"下限=地形角+FWHM/2={args.beam_fwhm / 2:.2f}°; 相位中心=台站高程+天线轴高{args.ant_height:.0f}m")
        for key, dem, base, prof in (("Luonan", dem_ln, base_ln, prof_ln),
                                     ("Hangzhou", dem_hz, base_hz, prof_hz)):
            nm = SITES[key]["cn"]
            if dem is None:
                print(f"    {nm}: 未找到DEM -> 均匀/CSV基准 "
                      f"{np.nanmin(base):.1f}°~{np.nanmax(base):.1f}°")
                continue
            t = dem["terrain"]
            nn = int(np.isnan(t).sum())
            htxt = (f"台站处DEM高程 {dem['h_site']:.0f}m" if dem["h_site"] is not None
                    else "台站不在DEM覆盖内")
            if nn >= 360:
                print(f"    {nm}: DEM {'+'.join(dem['names'])} 无任何有效数据, "
                      f"全部方位取均匀基准 {np.nanmin(base):.1f}°; {htxt}")
                continue
            imx = int(np.nanargmax(t))
            imn = int(np.nanargmin(t))
            wa = int(np.nanargmax(prof))
            wb = int(np.nanargmin(prof))
            print(f"    {nm}: DEM {'+'.join(dem['names'])}; {htxt} -> 相位中心 {dem['h0']:.0f}m")
            print(f"      地形几何角: {t[imn]:.1f}°@方位{imn}°(0=北) ~ {t[imx]:.1f}°@方位{imx}°; "
                  f"最终下限(∨基准{np.nanmin(base):.1f}°): {prof[wb]:.1f}°@{wb}° ~ {prof[wa]:.1f}°@{wa}°"
                  + (f"; 无DEM方位 {nn} 个(取基准)" if nn else ""))
            if dem.get("missing"):
                print(f"      提示: 台站附近缺少 {', '.join(dem['missing'])} 等邻格DEM, "
                      f"对应方向无数据部分已按均匀基准处理")
    print(f"  切换策略: {args.strategy}")

    print("\n[每颗脉冲星 升起/落下/共视概况]")
    print(f"  {'名称':<11s} {'赤经':<7s} {'赤纬':<8s} {'洛南可见(h)':>10s} {'杭州可见(h)':>10s} {'共视(h)':>8s}  {'洛南窗口(北京)'}")
    for p, n in enumerate(names):
        r1 = contiguous_runs(vis_ln[p])
        r2 = contiguous_runs(vis_hz[p])
        rc = contiguous_runs(common[p])
        h1 = sum(j - i + 1 for i, j in r1) * args.step_min / 60.0
        h2 = sum(j - i + 1 for i, j in r2) * args.step_min / 60.0
        hc = sum(j - i + 1 for i, j in rc) * args.step_min / 60.0
        if r1:
            w1 = f"{hhmm(bj_dt[r1[0][0]])}~{hhmm(bj_dt[r1[-1][1]])}"
            w1 += f"({len(r1)}段)" if len(r1) > 1 else ""
        else:
            w1 = "不可见"
        print(f"  {n:<11s} {ra_hrs[p]:5.2f}h {dec_deg[p]:+7.2f}° {h1:10.2f} {h2:10.2f} {hc:8.2f}  {w1}")

    print("\n[接力覆盖结果]")
    print(f"  理论最大覆盖(共视窗口并集): {union_frac * 24:6.2f} h  ({union_frac * 100:5.1f}%)")
    print(f"  本方案实际覆盖({args.strategy}策略): {cov_frac * 24:6.2f} h  ({cov_frac * 100:5.1f}%)")
    if abs(cov_frac - union_frac) < 1e-9:
        print("  (已达成理论最大覆盖)")
    if cov_frac >= 1.0 - 1e-9:
        print("  >>> 24小时不间断共视接力观测 可行! <<<")
    else:
        tot_gap = sum(j - i + 1 for i, j in gaps) * args.step_min / 60.0
        longest = max(e - s + 1 for s, e, p in segments) * args.step_min / 60.0 if segments else 0
        print(f"  >>> 无法做到24h不间断; 最长覆盖 {cov_frac * 24:.2f} h <<<")
        print(f"  空隙: {len(gaps)} 段, 共 {tot_gap:.2f} h; 最长连续观测 {longest:.2f} h")

    if gaps:
        print("\n[空隙列表(北京时间为无共视脉冲星时段)]")
        for i, (a, b) in enumerate(gaps):
            print(f"  {i + 1:2d}. {hhmm(bj_dt[a])} ~ {hhmm(bj_dt[b])}  "
                  f"({(b - a + 1) * args.step_min / 60.0:.2f} h)")

    print("\n[接力切换序列]")
    for i, (a, b, p) in enumerate(segments):
        mark = " <<" if i + 1 == len(segments) else ""
        print(f"  {i + 1:2d}. {hhmm(bj_dt[a])} ~ {hhmm(bj_dt[b])}  {names[p]}{mark}")

    print("\n[接力方案列表] 共枚举出 %d 种完整方案(覆盖=理论最大); 其中含 J0437-4715 的方案 %d 种,"
          " 展示前 %d 种(均含 J0437-4715):" % (n_schemes, n_with0437, len(ordered)))
    for k, (lab, segs, cov, lnm, hzm) in enumerate(ordered, 1):
        seg_txt = " → ".join(f"{names[p]} {hhmm(bj_dt[a])}~{hhmm(bj_dt[b])}"
                              for (a, b, p) in segs)
        tag = {"当前": " (当前排班)", "含J0437最优": " (含J0437-4715的最优方案)"}.get(lab, "")
        print(f"  方案{k}{tag}: {seg_txt}")
        print(f"        覆盖 {cov * 24:5.2f} h | 最低高度角(两站分别实际可达): "
              f"洛南 {lnm:4.1f}° / 杭州 {hzm:4.1f}°")

    print("\n[杭州各方位遮挡角上限(满足本接力方案, 每15°一档, 取档内最严)]")
    print(f"  {'方位角':>6s}  {'上限(°)':>9s}    {'方位角':>6s}  {'上限(°)':>9s}")
    for a in range(0, 180, 15):
        d1 = np.minimum((azg - a) % 360, (a - azg) % 360)
        d2 = np.minimum((azg - (a + 180)) % 360, ((a + 180) - azg) % 360)
        v1 = req[d1 <= 7]
        v2 = req[d2 <= 7]
        t1 = f"{np.nanmin(v1):5.1f}" if np.any(~np.isnan(v1)) else " 无要求"
        t2 = f"{np.nanmin(v2):5.1f}" if np.any(~np.isnan(v2)) else " 无要求"
        print(f"  {a:4d}°  {t1:>8s}    {a + 180:4d}°  {t2:>8s}")
    if valid.any():
        ia = int(np.nanargmin(req))
        print(f"  最严方向: 方位 {ia}°(即{ia}°方位角), 遮挡角上限 {req[ia]:.1f}°"
              f"(该方位共视观测时杭州侧最低高度角{req[ia] + args.margin:.1f}° - 余量{args.margin:.1f}°)")
    else:
        print("  无任何共视观测时段(请检查参数)")

    if h24 is not None:
        print(f"\n[灵敏度参考] 若杭州各方位遮挡角均匀升高, 24h覆盖可保持至均匀遮挡角约 {h24:.1f}°;"
              f" 超过后覆盖时长逐步下降(见图4)")
    else:
        print("\n[灵敏度参考] 即使杭州遮挡角为0°, 也无法达到24h覆盖(见图4)")

    # 洛南均匀遮挡角灵敏度(杭州遮挡=0)
    print("\n[灵敏度] 洛南均匀遮挡角 vs 最大覆盖 (杭州遮挡=0°, 俯仰下限=%.1f°, 余量=%.1f°):"
          % (args.min_elev, args.margin))
    for h in [0, 2, 4, 5, 6, 7, 8, 10, 12]:
        c = ((alt["Luonan"] > np.maximum(h, args.min_elev) + args.margin)
             & (alt["Hangzhou"] > np.maximum(0, args.min_elev) + args.margin)).any(axis=0).mean()
        print(f"    洛南遮挡 {h:3d}° -> 最大覆盖 {c * 24:5.2f} h ({c * 100:5.1f}%)")

    # 余量灵敏度(洛南7°, 杭州0°)
    print("\n[灵敏度] 余量(两侧均需高于有效地平+余量) vs 最大覆盖 (洛南=%.1f°, 杭州=%.1f°, 俯仰下限=%.1f°):"
          % (args.luonan_obs ,args.hangzhou_obs ,args.min_elev))
    for m in [0, 1, 2, 3, 5]:
        c = ((alt["Luonan"] > np.maximum(7, args.min_elev) + m)
             & (alt["Hangzhou"] > np.maximum(0, args.min_elev) + m)).any(axis=0).mean()
        print(f"    余量 {m:3d}° -> 最大覆盖 {c * 24:5.2f} h ({c * 100:5.1f}%)")

    # 检查杭州假设剖面是否已超上限
    over = np.where(valid & (prof_hz > req))[0]
    if over.size:
        print(f"\n[警告] 杭州假设遮挡剖面在 {over.size} 个方位角上高于本方案上限, "
              f"例如方位 {over[0]}°: 假设 {prof_hz[over[0]]:.1f}° > 上限 {req[over[0]]:.1f}°")

    # ================= 输出文件 =================
    # relay_schedule.csv
    csv_path = OUT_DIR / "relay_schedule.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["北京时间", "UTC", "脉冲星", "洛南高度角(deg)", "洛南方位角(deg)",
                    "杭州高度角(deg)", "杭州方位角(deg)",
                    "洛南有效地平(该方位,deg)", "杭州有效地平(该方位,deg)"])
        for t in range(T):
            p = sched[t]
            pn = names[p] if p >= 0 else ""
            w.writerow([bj_dt[t].strftime("%Y-%m-%d %H:%M:%S"),
                        times[t].iso, pn,
                        f"{alt['Luonan'][p, t]:.2f}" if p >= 0 else "",
                        f"{az['Luonan'][p, t]:.2f}" if p >= 0 else "",
                        f"{alt['Hangzhou'][p, t]:.2f}" if p >= 0 else "",
                        f"{az['Hangzhou'][p, t]:.2f}" if p >= 0 else "",
                        f"{eff_ln[p, t]:.2f}" if p >= 0 else "",
                        f"{eff_hz[p, t]:.2f}" if p >= 0 else ""])

    # hangzhou_obstruction_req.csv
    req_csv = OUT_DIR / "hangzhou_obstruction_req.csv"
    with open(req_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["方位角(deg, 0=北)", "杭州遮挡角上限(deg)", "是否有限制", "来源脉冲星"])
        for a in range(360):
            if not np.isnan(req[a]):
                w.writerow([a, f"{req[a]:.2f}", "是", names[req_src[a]] if req_src[a] >= 0 else ""])
            else:
                w.writerow([a, "", "无要求", ""])

    # relay_plans.csv 多种接力方案(每段一行)
    relay_csv = OUT_DIR / "relay_plans.csv"
    with open(relay_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["方案", "段序号", "脉冲星", "开始(北京)", "结束(北京)",
                    "方案覆盖(h)", "该段洛南最低高度角(deg)", "该段杭州最低高度角(deg)"])
        for k, (lab, segs, cov, lnm, hzm) in enumerate(ordered, 1):
            for i, (a, b, p) in enumerate(segs, 1):
                w.writerow([k, i, names[p], bj_dt[a].strftime("%H:%M"),
                            bj_dt[b].strftime("%H:%M"),
                            f"{cov * 24:.2f}",
                            f"{alt['Luonan'][p, a:b + 1].min():.1f}",
                            f"{alt['Hangzhou'][p, a:b + 1].min():.1f}"])

    # DEM 地形遮挡角曲线 csv(每方位一行; 仅当该站启用了DEM)
    dem_csvs = []
    for key, dem, base, prof in (("Luonan", dem_ln, base_ln, prof_ln),
                                 ("Hangzhou", dem_hz, base_hz, prof_hz)):
        if dem is None:
            continue
        dem_csv = OUT_DIR / f"{key.lower()}_dem_obstruction.csv"
        with open(dem_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow([f"# 台站: {SITES[key]['cn']} ({SITES[key]['lat']:.5f}N, "
                        f"{SITES[key]['lon']:.5f}E, 海拔{SITES[key]['height']:.0f}m)"])
            w.writerow([f"# DEM文件: {', '.join(dem['names'])}"])
            if dem["h_site"] is not None:
                w.writerow([f"# 台站处DEM高程: {dem['h_site']:.1f} m"])
            else:
                w.writerow(["# 台站不在DEM覆盖内(相位中心取台站参数海拔)"])
            w.writerow([f"# 相位中心海拔(台站高程+天线轴高{args.ant_height:.0f}m): {dem['h0']:.1f} m"])
            w.writerow([f"# 几何: 视线仰角含地球曲率与4/3大气折射; 判定: 脉冲星在波束中央"
                        f"(FWHM={args.beam_fwhm:.2f}°), 波束下边缘与地形最高点相切 -> "
                        f"下限=地形遮挡角+FWHM/2({args.beam_fwhm / 2:.2f}°)"])
            w.writerow(["# 最终遮挡下限 = max(均匀/CSV基准, DEM下限); 无DEM数据方位取均匀/CSV基准; "
                        "不含望远镜俯仰下限"])
            w.writerow(["方位角(deg)", "地形遮挡角几何(deg)", "波束半宽FWHM/2(deg)",
                        "DEM遮挡下限(deg)", "均匀/CSV基准(deg)", "最终遮挡下限(deg)",
                        "视线最长距离(km)", "有效采样点数"])
            for a in range(360):
                tv = dem["terrain"][a]
                wb = dem["withbeam"][a]
                w.writerow([a,
                            f"{tv:.2f}" if not np.isnan(tv) else "",
                            f"{args.beam_fwhm / 2:.2f}",
                            f"{wb:.2f}" if not np.isnan(wb) else "",
                            f"{base[a]:.2f}",
                            f"{prof[a]:.2f}",
                            f"{dem['dist_km'][a]:.1f}" if dem["dist_km"][a] > 0 else "",
                            int(dem["npts"][a])])
        dem_csvs.append(dem_csv)

    # ================= 图形 =================
    cmap = plt.cm.tab20
    colors = {n: cmap(i % 20) for i, n in enumerate(names)}

    # ---- 图1: 升起落下曲线(浅色全天轨迹 + 深色可见段, 体现逐方位地形遮挡) ----
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9), sharex=True)
    for ax, k in zip(axes, ["Luonan", "Hangzhou"]):
        prof = prof_ln if k == "Luonan" else prof_hz
        visk = vis_ln if k == "Luonan" else vis_hz
        dem = dem_ln if k == "Luonan" else dem_hz
        effv = np.maximum(prof, args.min_elev)
        for p, n in enumerate(names):
            altk = alt[k][p]
            ax.plot(bj_dt, altk, lw=0.9, color=colors[n], alpha=0.30)  # 全天轨迹(浅)
            first = True
            i = 0
            while i < T:  # 可见段: 高于该方位有效地平+余量
                if visk[p, i]:
                    j = i
                    while j + 1 < T and visk[p, j + 1]:
                        j += 1
                    ax.plot(bj_dt[i:j + 1], altk[i:j + 1], lw=2.0, color=colors[n],
                            label=n if first else None)
                    first = False
                    i = j + 1
                else:
                    i += 1
        if dem is not None:
            nn = int(np.isnan(dem["terrain"]).sum())
            txt = (f"遮挡剖面: DEM地形({'+'.join(dem['names'])}), 与均匀/CSV基准逐方位取大\n"
                   f"有效地平(∨俯仰下限{args.min_elev:.0f}°): "
                   f"{effv.min():.1f}°~{effv.max():.1f}°"
                   + (f"\n无DEM数据方位 {nn} 个(取均匀基准)" if nn else ""))
        else:
            txt = (f"遮挡剖面: 均匀{prof[0]:.1f}°\n"
                   f"有效地平(∨俯仰下限{args.min_elev:.0f}°): "
                   f"{effv.min():.1f}°~{effv.max():.1f}°")
        ax.text(0.005, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.62))
        ax.set_ylabel("高度角 (°)")
        ax.set_ylim(-5, 90)
        ax.grid(alpha=0.3)
        ax.legend(ncol=5, fontsize=8, loc="upper right")
        ax.set_title(f"{SITES[k]['cn']}  脉冲星升起-落下曲线  ({args.date} 北京时间)\n"
                     f"浅色=全天轨迹, 深色粗线=可见段(高于有效地平; 遮挡随脉冲星方位变化)",
                     fontsize=10.5)
    axes[1].set_xlabel("北京时间")
    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    f1 = OUT_DIR / "fig1_riseset_curves.png"
    fig.savefig(f1, bbox_inches="tight")
    plt.close(fig)

    # ---- 图2: 接力规划图 (Gantt): 共视窗口 + 多种接力方案 ----
    K = len(ordered)
    fig, axes = plt.subplots(1 + K, 1, figsize=(14.5, 3.8 + 1.30 * K), sharex=True)
    ax1 = axes[0]
    ax1.set_yticks(range(N))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_ylim(-0.7, N - 0.3)
    for p, n in enumerate(names):
        for (i, j) in contiguous_runs(common[p]):
            ax1.barh(p, (j - i + 1) * args.step_min / 60.0 / 24.0,
                     left=bj_dt[i], height=0.65, color=colors[n], edgecolor="none")
    ax1.set_title(f"① 各脉冲星 两地共视窗口 (洛南∩杭州 同时可见; 共枚举出 {n_schemes} 种完整方案,"
                  f" 其中含J0437-4715的 {n_with0437} 种; 下图展示 {K} 种, 均含J0437-4715)\n"
                  f"遮挡剖面有效地平范围(∨俯仰下限{args.min_elev:.0f}°): "
                  f"洛南 {np.maximum(prof_ln, args.min_elev).min():.1f}~{np.maximum(prof_ln, args.min_elev).max():.1f}°, "
                  f"杭州 {np.maximum(prof_hz, args.min_elev).min():.1f}~{np.maximum(prof_hz, args.min_elev).max():.1f}°",
                  fontsize=10)
    ax1.grid(axis="x", alpha=0.3)

    for k, (lab, segs, cov, malt, mn) in enumerate(ordered, start=1):
        ax = axes[k]
        ax.set_yticks(range(N))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_ylim(-0.7, N - 0.3)
        # 空隙(红区)
        covm = np.zeros(T, dtype=bool)
        for (a, b, p) in segs:
            covm[a:b + 1] = True
        tt = 0
        while tt < T:
            if not covm[tt]:
                jj = tt
                while jj + 1 < T and not covm[jj + 1]:
                    jj += 1
                ax.axvspan(bj_dt[tt], bj_dt[jj], color="red", alpha=0.30)
                tt = jj + 1
            else:
                tt += 1
        for (a, b, p) in segs:
            ax.barh(p, (b - a + 1) * args.step_min / 60.0 / 24.0,
                    left=bj_dt[a], height=0.65, color=colors[names[p]], edgecolor="none")
        seg_txt = " → ".join(f"{names[p]} {hhmm(bj_dt[a])}~{hhmm(bj_dt[b])}"
                              for (a, b, p) in segs)
        tag = {"当前": " (当前排班)", "含J0437最优": " (含J0437-4715的最优方案)"}.get(lab, "")
        ax.set_title(f"方案{k}{tag}: 覆盖 {cov * 24:.2f} h | 最低高度角(两站实际可达): "
                     f"洛南 {lnm:.1f}° / 杭州 {hzm:.1f}°\n{seg_txt}",
                     fontsize=8.5, loc="left")
        ax.grid(axis="x", alpha=0.3)
    axes[-1].set_xlabel("北京时间")
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    f2 = OUT_DIR / "fig2_relay_plan.png"
    fig.savefig(f2, bbox_inches="tight")
    plt.close(fig)

    # ---- 图3: 两地 DEM 地形遮挡剖面 (极坐标 + 直角坐标; 不含遮挡角上限) ----
    fig = plt.figure(figsize=(15, 6.6))
    axp = fig.add_subplot(1, 2, 1, projection="polar")
    axp.set_theta_zero_location("N")
    axp.set_theta_direction(-1)
    axp.set_rlabel_position(45)
    axp.plot(np.deg2rad(azg), prof_ln, "-", color="tab:blue", lw=1.5,
             label=f"洛南遮挡下限(最终) {prof_ln.min():.1f}~{prof_ln.max():.1f}°")
    axp.plot(np.deg2rad(azg), prof_hz, "-", color="k", lw=1.5,
             label=f"杭州遮挡下限(最终) {prof_hz.min():.1f}~{prof_hz.max():.1f}°")
    if dem_ln is not None:
        m = np.ma.masked_invalid(dem_ln["terrain"])
        axp.plot(np.deg2rad(azg), m, ":", color="tab:blue", lw=1.0,
                 label="洛南地形遮挡角(DEM几何)")
    if dem_hz is not None:
        m = np.ma.masked_invalid(dem_hz["terrain"])
        axp.plot(np.deg2rad(azg), m, ":", color="gray", lw=1.0,
                 label="杭州地形遮挡角(DEM几何)")
    axp.set_title("两地 地形遮挡角下限 (极坐标)\n0°=北, 90°=东, 180°=南, 270°=西",
                  fontsize=10)
    axp.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=8)

    axc = fig.add_subplot(1, 2, 2)
    axc.plot(azg, prof_ln, "-", color="tab:blue", lw=1.5,
             label=f"洛南遮挡下限(最终) {prof_ln.min():.1f}~{prof_ln.max():.1f}°")
    axc.plot(azg, prof_hz, "-", color="k", lw=1.5,
             label=f"杭州遮挡下限(最终) {prof_hz.min():.1f}~{prof_hz.max():.1f}°")
    if dem_ln is not None:
        m = np.ma.masked_invalid(dem_ln["terrain"])
        axc.plot(azg, m, ":", color="tab:blue", lw=1.0,
                 label="洛南地形遮挡角(DEM几何)")
    if dem_hz is not None:
        m = np.ma.masked_invalid(dem_hz["terrain"])
        axc.plot(azg, m, ":", color="gray", lw=1.0,
                 label="杭州地形遮挡角(DEM几何)")
    axc.set_xlabel("方位角 (°, 0=北, 90=东, 180=南, 270=西)")
    axc.set_ylabel("遮挡角下限 (°)")
    axc.set_xticks(range(0, 360, 30))
    axc.grid(alpha=0.3)
    ymin = 0.0
    for arr in [prof_ln, prof_hz] + ([dem_ln["terrain"]] if dem_ln is not None else []) \
            + ([dem_hz["terrain"]] if dem_hz is not None else []):
        a = arr[~np.isnan(arr)]
        if a.size:
            ymin = min(ymin, float(a.min()) - 1.0)
    axc.set_ylim(bottom=ymin)
    dem_txt = []
    for k2, d2 in (("Luonan", dem_ln), ("Hangzhou", dem_hz)):
        if d2 is None:
            dem_txt.append(f"{k2}:无DEM")
        else:
            nn = int(np.isnan(d2["terrain"]).sum())
            dem_txt.append(f"{k2}:{d2['names'][0]}" + (f"(无数据方位{nn})" if nn else ""))
    axc.set_title("两地 地形遮挡角下限 (直角坐标)\n" + "; ".join(dem_txt) + "\n"
                  "实线=最终下限(DEM∨均匀基准), 点线=DEM地形几何角(含4/3折射)",
                  fontsize=9)
    axc.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    f3 = OUT_DIR / "fig3_obstruction_req.png"
    fig.savefig(f3, bbox_inches="tight")
    plt.close(fig)

    # ---- 图4: 覆盖时长 vs 杭州均匀遮挡角 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hs, covs * 24, "o-", ms=3.5, lw=1.2, color="tab:blue",
            label="最大覆盖时长(共视窗口并集)")
    ax.axhline(24, color="gray", ls="--", lw=1)
    if h24 is not None:
        ax.axvline(h24, color="red", ls="--", lw=1.2)
        ax.annotate(f"24h 均匀遮挡上限 ≈ {h24:.1f}°",
                    xy=(h24, 24), xytext=(h24 + 1, 22.5),
                    arrowprops=dict(arrowstyle="->"), fontsize=9, color="red")
    ax.set_xlabel("杭州均匀遮挡角 (°)")
    ax.set_ylabel("最大覆盖时长 (h)")
    ax.set_title("最大覆盖时长 vs 杭州均匀遮挡角 (灵敏度参考)")
    ax.set_ylim(0, 25)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    f4 = OUT_DIR / "fig4_coverage_vs_obs.png"
    fig.savefig(f4, bbox_inches="tight")
    plt.close(fig)

    # ---- 图5: 所有脉冲星的方位-俯仰(alt-az)轨迹图 ----
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), sharey=True)
    azg5 = np.arange(360.0)
    for ax, k in zip(axes, ["Luonan", "Hangzhou"]):
        prof = prof_ln if k == "Luonan" else prof_hz
        # 不可观测区: 低于 有效地平(遮挡∨俯仰下限)+余量
        hor = np.maximum(obs_at(prof, azg5), args.min_elev) + args.margin
        ax.fill_between(azg5, 0, hor, color="0.75", alpha=0.45,
                        label="不可观测区(遮挡∨俯仰下限+余量)")
        vism = vis_ln if k == "Luonan" else vis_hz
        for p, n in enumerate(names):
            azs = az[k][p]
            alts = alt[k][p]
            visp = vism[p]
            # 方位角绕北跳变处断开(如 359°→1°)
            brk = np.where(np.abs(np.diff(azs)) > 180.0)[0] + 1
            cuts = np.concatenate(([0], brk, [T]))
            for s, e in zip(cuts[:-1], cuts[1:]):
                ax.plot(azs[s:e], alts[s:e], color="0.82", lw=0.7)  # 全天轨迹(浅灰)
                i = s
                while i < e:  # 可见段加粗着色, 起点=升起, 终点=落下
                    if visp[i]:
                        j = i
                        while j + 1 < e and visp[j + 1]:
                            j += 1
                        ax.plot(azs[i:j + 1], alts[i:j + 1], color=colors[n], lw=1.9)
                        ax.plot(azs[i], alts[i], ".", color=colors[n], ms=6)
                        ax.plot(azs[j], alts[j], "x", color=colors[n], ms=6)
                        i = j + 1
                    else:
                        i += 1
        ax.set_xlim(0, 360)
        ax.set_xticks(range(0, 360, 30))
        ax.set_xlabel("方位角 (°, 0=北, 90=东, 180=南, 270=西)")
        ax.set_ylim(0, 90)
        ax.grid(alpha=0.3)
        ax.set_title(f"{SITES[k]['cn']}  脉冲星方位-俯仰(alt-az)轨迹\n"
                     f"灰线=全天轨迹, 彩色粗线=可见段(•升起 ×落下), 阴影=不可观测区",
                     fontsize=10)
    axes[0].set_ylabel("俯仰角 / 高度角 (°)")
    handles, labels = axes[0].get_legend_handles_labels()
    h2 = [plt.Line2D([0], [0], color=colors[n], lw=2.2) for n in names]
    fig.legend(handles=handles + h2, labels=labels + list(names), ncol=6,
               fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    f5 = OUT_DIR / "fig5_altaz_tracks.png"
    fig.savefig(f5, bbox_inches="tight")
    plt.close(fig)

    # ---- 图6: 每台站一张复合图: 上子图=全部脉冲星 俯仰-时间, 下子图=全部脉冲星 方位-时间 ----
    # 文件: fig6_azel_time_Luonan.png , fig6_azel_time_Hangzhou.png
    fig6_files = []
    for k in ["Luonan", "Hangzhou"]:
        prof = prof_ln if k == "Luonan" else prof_hz
        vism = vis_ln if k == "Luonan" else vis_hz
        legacy_dir = OUT_DIR / f"fig6_azel_time_{k}"
        if legacy_dir.is_dir():  # 清理旧版"每颗星一张"目录, 避免残留混淆
            shutil.rmtree(legacy_dir)

        fig, (ax_el, ax_az) = plt.subplots(
            2, 1, figsize=(13.5, 8.6), sharex=True,
            gridspec_kw=dict(height_ratios=[1, 1], hspace=0.06))
        for p, n in enumerate(names):
            azs = az[k][p]
            alts = alt[k][p]
            visp = vism[p]
            # 方位角绕北跳变处断开(如 359°→1°)
            brk = np.where(np.abs(np.diff(azs)) > 180.0)[0] + 1
            cuts = np.concatenate(([0], brk, [T]))
            # 全天轨迹(浅灰): 上=俯仰, 下=方位
            for s, e in zip(cuts[:-1], cuts[1:]):
                ax_el.plot(bj_dt[s:e], alts[s:e], color="0.82", lw=0.7)
                ax_az.plot(bj_dt[s:e], azs[s:e], color="0.82", lw=0.7)
            # 可见段加粗着色, 起点=升起(•), 终点=落下(×)
            first = True
            i = 0
            while i < T:
                if visp[i]:
                    j = i
                    while j + 1 < T and visp[j + 1]:
                        j += 1
                    ax_el.plot(bj_dt[i:j + 1], alts[i:j + 1], color=colors[n], lw=1.6,
                               label=n if first else None)
                    ax_az.plot(bj_dt[i:j + 1], azs[i:j + 1], color=colors[n], lw=1.4)
                    ax_el.plot(bj_dt[i], alts[i], ".", color=colors[n], ms=4)
                    ax_el.plot(bj_dt[j], alts[j], "x", color=colors[n], ms=4)
                    first = False
                    i = j + 1
                else:
                    i += 1
        # 有效地平参考线: 剖面随方位变化 -> 画上/下限两条参考
        eff_hi = max(float(prof.max()), args.min_elev)
        eff_lo = max(float(prof.min()), args.min_elev)
        ax_el.axhline(eff_hi, color="r", ls="--", lw=1.0,
                      label=f"有效地平(遮挡∨俯仰) {eff_lo:.1f}°~{eff_hi:.1f}°")
        ax_el.axhline(eff_lo, color="r", ls=":", lw=0.8)
        ax_el.set_ylim(-5, 90)
        ax_el.set_ylabel("俯仰角 / 高度角 (°)")
        ax_el.grid(alpha=0.3)
        ax_el.legend(ncol=3, fontsize=7.5, loc="upper right")

        for a0 in (0, 90, 180, 270):
            ax_az.axhline(a0, color="k", lw=0.5)
        ax_az.set_ylim(0, 360)
        ax_az.set_yticks([0, 90, 180, 270, 360])
        ax_az.set_yticklabels(["0°北", "90°东", "180°南", "270°西", "360°北"])
        ax_az.set_xlim(bj_dt[0], bj_dt[-1])
        ax_az.set_xlabel("北京时间")
        ax_az.set_ylabel("方位角 (°)")
        ax_az.grid(alpha=0.3)
        ax_az.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax_az.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate()
        fig.suptitle(f"{SITES[k]['cn']}  全部脉冲星  俯仰-时间(上) / 方位-时间(下)  ({args.date} 北京时间)\n"
                     f"灰线=全天轨迹, 彩色粗线=可见段(•升起 ×落下), 红线=有效地平",
                     fontsize=11.5)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fp = OUT_DIR / f"fig6_azel_time_{k}.png"
        fig.savefig(fp, bbox_inches="tight")
        plt.close(fig)
        fig6_files.append(fp)

    print("\n[输出文件]")
    for f in [f1, f2, f3, f4, f5, csv_path, req_csv, relay_csv] + fig6_files + dem_csvs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
