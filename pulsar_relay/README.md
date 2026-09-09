# 双40米射电望远镜 · 脉冲星共视接力观测方案

洛南站（陕西省商洛市洛南县，34.170851°N, 109.951457°E）与杭州站（29.934176°N, 119.829719°E）两台 40 m 级射电望远镜的脉冲星**共视接力 24 小时不间断观测**规划脚本。

## 功能

- 用 astropy 计算脉冲星在两地的升起/落下（高度角-时间）曲线；
- 计算共视接力方案：任一时刻要求两站同时可见同一颗脉冲星（高度角 ≥ 该方位遮挡下限 + 余量），目标 24 h 不间断；枚举多种完整方案并展示前 3 种；
- 可选 ASTER GDEM 地形遮挡：逐方位沿视线采样地形（含地球曲率与 4/3 折射、波束 FWHM 半宽），合成各方位遮挡下限；
- 输出图与表（见下）。

## 依赖

```
python>=3.9
numpy  matplotlib  astropy
rasterio   # 仅当使用 DEM 地形遮挡时需要
```

## 数据文件

- `10-MSP.txt`：脉冲星列表（每行一个 J 名称）；
- `ASTGTM_N34E109.img.zip` / `ASTGTM_N29E119.img.zip`：ASTER GDEM 高程数据（需**先解压出同名 `.img`** 放到脚本同目录；脚本也会尝试自动解压）。坐标已更新后两站均落在上述 1° 格内。

## 用法

```bash
# 基本：洛南均匀遮挡 7°、杭州均匀遮挡 15°
python pulsar_relay_plan.py --pulsars 10-MSP.txt --luonan-obs 7 --hangzhou-obs 15

# 杭州遮挡以 DEM 曲线为准（南侧地形低，J0437 等南天源可共视）
python pulsar_relay_plan.py --pulsars 10-MSP.txt --date 2026-09-09 --luonan-obs 0.0 --hangzhou-obs 0.0 --min-elev 5.0 --margin 1.0 --strategy longest
```

常用参数（`python pulsar_relay_plan.py --help` 查看全部）：

| 参数 | 含义 | 默认 |
|---|---|---|
| `--pulsars` | 脉冲星列表文件 | `10-MSP.txt` |
| `--date` | 观测日期（北京时间，默认今天） | — |
| `--luonan-obs` / `--hangzhou-obs` | 均匀遮挡角基准 | 7 / 0 |
| `--min-elev` | 望远镜俯仰下限 | 5.0° |
| `--margin` | 余量 | 0.0° |
| `--strategy` | `longest` / `list` | `longest` |
| `--dem-mode` | `max`（基准保底）/ `replace`（DEM 为主） | `max` |
| `--beam-fwhm` / `--ant-height` | 波束宽度 / 天线轴高 | 0.3° / 15 m |

## 输出

| 文件 | 内容 |
|---|---|
| `fig1_riseset_curves.png` | 两地脉冲星升起/落下曲线 |
| `fig2_relay_plan.png` | 共视接力规划（Gantt，含 J0437 的前 3 种方案） |
| `fig3_obstruction_req.png` | 两地 DEM 地形遮挡角下限（极坐标 + 直角坐标） |
| `fig4_coverage_vs_obs.png` | 最大覆盖时长 vs 杭州均匀遮挡角（灵敏度辅助） |
| `fig5_altaz_tracks.png` | 两地全部脉冲星 alt-az 轨迹 |
| `fig6_azel_time_*.png` | 洛南 / 杭州俯仰-时间、方位-时间复合图 |
| `relay_schedule.csv` | 接力排班表（北京/UTC 时间、星名、高度角、方位角） |
| `relay_plans.csv` | 多种接力方案一览 |
| `hangzhou_obstruction_req.csv` | 杭州各方位遮挡角上限（满足排班的工程要求曲线） |
| `luonan_dem_obstruction.csv` / `hangzhou_dem_obstruction.csv` | 两站 DEM 逐方位遮挡角曲线 |
