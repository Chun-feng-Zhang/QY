# 方向 B · Duffing 振子单脉冲信号增强（相干单脉冲实验）

> 代码源：本地工程 `coherent_duffing_singlepulse`（2026-09 状态拷贝入库）。

## 概述

观测对象为毫秒脉冲星 **B1937+21**（DM = 71.0237 pc·cm⁻³），2024-11-22 基带观测（500 MHz 带宽，中心 1250 MHz）。研究链路：

1. **相干消色散**：DADA 基带 → 复基带 → Hankins–Rickett 消色散滤波 → 降采样功率序列（2 MHz）；
2. **单脉冲/巨脉冲分析**：匹配滤波检测、相位标定、2D 折叠动态谱、巨脉冲时频图；
3. **Duffing 振子增强**：强耦合 Duffing 双振子（v5，参考 5-diffuse 最优参数）在脉冲重复频率上谐振，输出 `w·(y1−y2)`，可在**不折叠**情况下重建/增强单脉冲时域波形并提升 SNR —— 区别于经典 Holmes–Duffing 仅能判别周期成分存在性。

## 文档

| 文件 | 说明 |
|---|---|
| `Duffing振子检测实验总结_20260905.docx` | 实验总结报告（由 `extract_report_data.py` + `build_report_docx.py` 自动生成） |

## 脚本分组

| 阶段 | 脚本 | 说明 |
|---|---|---|
| 相干消色散 | `coherent_dedisp.py` | 读 8-bit 实数基带（fs=1 GHz）→ Hankins–Rickett 滤波（N_fft=2²⁸，overlap-save）→ 2 MHz 功率序列 |
| 折叠 / 动态谱 | `make_fold2d_raw.py` | 2D 折叠（频域切 32 子带，15.625 MHz/带） |
| | `make_singlepulse_giant.py` | 巨脉冲单脉冲动态谱（t_giant=16.9302305 s ±1 ms，不折叠） |
| | `compose_fold_final.py`, `align_singlepulse.py` | 折叠图合成、单脉冲对齐 |
| | `check_fold2d_t10.py`, `verify_fold2d.py`, `scan_fold_full.py` | 折叠结果校验/扫描 |
| Duffing 增强 | `duffing_singlepulse_30ms.py` | 主流程：无折叠单脉冲 SNR 增强，30 ms 分段标注与 SNR 统计 |
| | `duffing_mark_20ms.py` / `duffing_mark_t10_20ms.py` | 20 ms 窗标注图（锚定脉冲 + 主/中间脉冲网格） |
| | `duffing_mark_fold_t10_20ms.py` | 20 ms 窗标注 + 整段 2s 2D 折叠图 |
| 扫描 | `scan_snr_probe*.py` | 参数/SNR 扫描（过程性） |
| 报告 | `extract_report_data.py`, `build_report_docx.py` | 汇总 `out_*.json` → 生成 docx 报告 |
| 调试 | `debug_*.py`, `probe_*.py`, `check*.py` | 过程性调试脚本（未整理，保留原始状态） |

## 数据与运行

- **原始数据不入库**：`bcmin1937+21.dada`（数百 GB 级）及 `B1937+21.par` 位于外部磁盘 `D:\VBshare\fast_1937+21\`，脚本内为绝对路径；
- **中间大文件不入库**：`out_coh_*.npy`（260 MB 级功率序列）由 `coherent_dedisp.py` 产出；`*.npz`、`*.png`、`*.log`、`__pycache__` 均不入库（重跑可再生成）；
- 运行示例：

```bash
python coherent_dedisp.py --dur 0.6 --plot          # 自检/短段处理
python duffing_singlepulse_30ms.py --series out_coh_t10_2s.npy --meta out_coh_t10_2s.json \
    --par D:\VBshare\fast_1937+21\B1937+21.par --t0 10.0 --seg-ms 30 --n-seg 4
```
