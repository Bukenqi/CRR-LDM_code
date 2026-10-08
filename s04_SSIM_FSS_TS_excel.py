# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:41:35 2026

@author: 59278
"""

# -*- coding: utf-8 -*-
"""逐成员 SSIM / FSS / TS 统计表。

行为生成次数，列为阈值（每个阈值一对 Mean / SD 列）。
每个指标一个 sheet，页内 Full 与 IR 上下排列。
"""

import os

import numpy as np
import pandas as pd

import FUNC_read_data as read
import FUNC_analyse_data as analyse


MODELS = {"Full": "CRR-LDM-Full", "IR": "CRR-LDM-IR"}
N_MEMBERS = 10
BOUNDARY = [-25, -20, -15, -10, -5, 0, 5, 10, 15]
FILL_NAN = -35.0
OUT_DIR = "/public/home/Xiongqq/CRR-LDM_result"
TAG = "metric_per_member"


def mean_std(values):
    """返回样本间的均值和标准差。"""
    return np.nanmean(values), np.nanstd(values)


def threshold_key(name):
    """阈值排序用。"""
    text = str(name).replace(">=", "").strip()
    try:
        return float(text)
    except ValueError:
        return np.inf


# store[metric][model][member][threshold] = (mean, std)
store = {metric: {label: {} for label in MODELS.values()} for metric in ("SSIM", "FSS", "TS")}

for model_tag, model_label in MODELS.items():

    data_path = (f"{OUT_DIR}/CRRLDM-{model_tag}_test_2020_original/CRRLDM-{model_tag}_49time.nc")
    data = read.read_data_nc(data_path,["Gen_result", "Radar_Reflectivity"],dtype=np.float32,)

    true = data["Radar_Reflectivity"]
    true[np.isnan(true)] = FILL_NAN
    gen = data["Gen_result"]

    n_use = min(N_MEMBERS, gen.shape[-1])
    print(f"\n{model_label}: true {true.shape}  gen {gen.shape}  使用 {n_use} 个成员")

    for m in range(n_use):

        pred = gen[:, :, :, m]

        ssim = analyse.Evaluation_sort("SSIM", true, pred)["All"]
        fss = analyse.Evaluation_sort("FSS", true, pred, boundary=BOUNDARY)
        ts = analyse.Evaluation_sort("TS", true, pred, boundary=BOUNDARY)

        member = m + 1

        store["SSIM"][model_label][member] = {"All": mean_std(ssim)}
        store["FSS"][model_label][member] = {str(k): mean_std(v) for k, v in fss.items()}
        store["TS"][model_label][member] = {str(k): mean_std(v) for k, v in ts.items()}
        print(f"  member {member}/{n_use} done")


def build_sheet(metric_store):
    """行=生成次数，列=阈值的 Mean / SD；两个模型上下排列。"""
    blocks = []

    for model_label in MODELS.values():
        per_member = metric_store[model_label]
        if not per_member:
            continue

        members = sorted(per_member)
        thresholds = sorted(per_member[members[0]], key=threshold_key)

        rows = []
        for m in members:
            row = {"Model": model_label, "Member": str(m)}
            for th in thresholds:
                mean_v, std_v = per_member[m][th]
                row[f"{th} Mean"] = round(mean_v, 4)
                row[f"{th} SD"] = round(std_v, 4)
            rows.append(row)

        df = pd.DataFrame(rows)

        # 成员间统计：对上面各成员的 Mean 再求均值和标准差
        stat = []
        for name, func in (("Mean over members", np.nanmean),
                           ("SD over members", np.nanstd)):
            row = {"Model": model_label, "Member": name}
            for th in thresholds:
                row[f"{th} Mean"] = round(func(df[f"{th} Mean"].values), 4)
            stat.append(row)

        blocks.append(pd.concat([df, pd.DataFrame(stat)], ignore_index=True))
        blocks.append(pd.DataFrame([{}]))

    return pd.concat(blocks, ignore_index=True)


sheets = {metric: build_sheet(store[metric]) for metric in ("SSIM", "FSS", "TS")}

xlsx_path = os.path.join(OUT_DIR, f"{TAG}.xlsx")

try:
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for metric in ("SSIM", "FSS", "TS"):
            sheets[metric].to_excel(writer, sheet_name=metric, index=False)
            writer.sheets[metric].freeze_panes = "C2"
    print(f"\n已保存: {xlsx_path}")

except ModuleNotFoundError:
    for metric in ("SSIM", "FSS", "TS"):
        path = os.path.join(OUT_DIR, f"{TAG}_{metric}.csv")
        sheets[metric].to_csv(path, index=False)
        print(f"已保存: {path}")
    print("提示: 未安装 openpyxl，已改存为 CSV。")

for metric in ("SSIM", "FSS", "TS"):
    print(f"\n===== {metric} =====")
    print(sheets[metric].to_string(index=False))