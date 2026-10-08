# -*- coding: utf-8 -*-
"""s01_ensemble.py —— 分步计算：CRPS / spread-RMSE / 生成一致性 / 出图

Step 1  读取已算好的逐样本 CRPS，聚合成成员数扫描
Step 2  读取 50 成员生成结果，算 spread、RMSE 与按云类统计
Step 3  同一份数据画生成一致性图，随后释放内存
Step 4  出图（双面板）
"""
import os
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '1'

import gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import FUNC_read_data as read
import FUNC_plot_image as plot
from s00_plot_style import (FACE, LINE, EDGE, RC, NO_ECHO, MODELS, DATA_PATH,
                            IMG_SAVE, OR_DATA, CLASS_ID, DROP_CLASS,
                            CLASS_ORDER, SAMPLES)

plt.rcParams.update(RC)

# 统一字号（取自 2a_F2 标准）
FS_TICK  = 15 #刻度数字
FS_LABEL = 16.5 #轴标题(dBZ、CRPS、成员数)
FS_TITLE = 17.5 #面板标题(CRR-LDM-Full/IR)
FS_LEG   = 12 #图例
FS_ANNO  = 9 #折线/柱上的数值标注

MEMBERS   = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
REF_M     = 30
N_GEN     = 5                      # 一致性图展示的成员数
SUB_BLOCK = 32                     # 每个进程内部的分块大小
N_WORKER  = 16
N_BOOT    = 1000
MIN_PIX   = 10000
OUT_XLSX  = f'{DATA_PATH}/ensemble_metrics.xlsx'

F2_YLIM, F2_YTICK = (0, 15), 3     # spread / RMSE vs members
F5_YLIM, F5_YTICK = (0, 30), 5     # by cloud class

TRUTH = PRED = CLS = None          # fork 共享，由 load_model 填充


# ===================================================================
# Step 1  CRPS
# ===================================================================
def _load_crps(args):
    model, m, pos = args
    f = (Path(f'{DATA_PATH}/CRRLDM-{model}_test_2020_original_CRPS')
         / f'CRRLDM-{model}_{m}time.nc')
    bar = tqdm(total=1, desc=f'{model[:2]}:m={m:<3}', position=pos,
               leave=False, ncols=70)
    if not f.exists():
        bar.close()
        return model, m, None
    v = read.read_data_nc(str(f), ['sample_crps'],
                          dtype=np.float32)['sample_crps']
    bar.update(1)
    bar.close()
    return model, m, np.asarray(v, dtype=np.float64)


def step1_crps():
    """返回 {model: {m: 逐样本 CRPS}}。"""
    print('=' * 62)
    print('Step 1  reading pre-computed CRPS')
    print('=' * 62)

    pairs = [(mo, m) for mo in MODELS for m in MEMBERS]
    tasks = [(mo, m, i % N_WORKER) for i, (mo, m) in enumerate(pairs)]
    out = {mo: {} for mo in MODELS}

    ctx = mp.get_context('fork')
    lock = ctx.RLock()
    tqdm.set_lock(lock)
    with ProcessPoolExecutor(max_workers=N_WORKER, mp_context=ctx,
                             initializer=tqdm.set_lock,
                             initargs=(lock,)) as ex:
        for fut in as_completed([ex.submit(_load_crps, t) for t in tasks]):
            model, m, v = fut.result()
            if v is None:
                tqdm.write(f'  missing CRPS: {model} m={m}')
                continue
            out[model][m] = v

    print('\n' * min(N_WORKER, len(tasks)))
    for mo in MODELS:
        print(f'  {mo:5s} loaded members: {sorted(out[mo])}')
    return out


# ===================================================================
# Step 2  spread / RMSE / 按云类
# ===================================================================
def load_model(model):
    """载入真值、云类与 50 成员生成结果（全像元，NaN → −35）。"""
    global TRUTH, PRED, CLS
    f = f'{DATA_PATH}/CRRLDM-{model}_test_2020_original/CRRLDM-{model}_49time.nc'
    print(f'\n[{model}] loading generation result ...')
    PRED = np.nan_to_num(
        read.read_data_nc(f, ['Gen_result'], dtype=np.float32)['Gen_result'],
        nan=NO_ECHO)

    od = read.read_data_nc(OR_DATA, ['cloud_scenario', 'Radar_Reflectivity'],
                           dtype=np.float32)
    ref = PRED.shape[:3]

    def align(a):
        if a.shape[:3] == ref:
            return a
        if a.shape[0] == ref[0] and a.shape[1:3] == ref[1:3][::-1]:
            return np.swapaxes(a, 1, 2)
        raise ValueError(f'shape mismatch {a.shape} vs {ref}')

    TRUTH = np.nan_to_num(align(od['Radar_Reflectivity']), nan=NO_ECHO)
    CLS = align(od['cloud_scenario']).astype(np.int16)
    del od
    gc.collect()
    print(f'[{model}] samples={PRED.shape[0]}, members={PRED.shape[-1]}')


def free_model():
    global TRUTH, PRED, CLS
    TRUTH = PRED = CLS = None
    gc.collect()


def _worker(args):
    """一个连续样本区间；进程内部分块并显示自己的进度条。"""
    wid, s, e = args
    n = e - s
    var = {m: np.empty(n) for m in MEMBERS}
    mse = {m: np.empty(n) for m in MEMBERS}
    acc = {c: np.zeros(3) for c in CLASS_ID}

    for b in tqdm(range(0, n, SUB_BLOCK), desc=f'worker {wid:02d} [{s}:{e}]',
                  position=wid, leave=False, ncols=70):
        a, z = b, min(b + SUB_BLOCK, n)
        truth = TRUTH[s + a:s + z].astype(np.float64)
        ens_a = PRED[s + a:s + z].astype(np.float64)
        cls   = CLS[s + a:s + z]

        for m in MEMBERS:
            ens  = ens_a[..., :m]
            mean = ens.mean(axis=-1)
            v    = ens.var(axis=-1, ddof=1) if m > 1 else np.zeros_like(mean)
            se   = (mean - truth) ** 2
            var[m][a:z] = v.mean(axis=(1, 2))
            mse[m][a:z] = se.mean(axis=(1, 2))
            if m == REF_M:
                for c in CLASS_ID:
                    sel = cls == c
                    if sel.any():
                        acc[c] += np.array([v[sel].sum(), se[sel].sum(),
                                            int(sel.sum())])
    return s, e, var, mse, acc


def step2_spread_rmse(model, crps_store):
    """返回 (sweep_rows, class_rows)。"""
    print('=' * 62)
    print(f'Step 2  spread / RMSE  —  CRR-LDM-{model}')
    print('=' * 62)

    n = TRUTH.shape[0]
    bounds = np.linspace(0, n, N_WORKER + 1).astype(int)
    tasks = [(i, bounds[i], bounds[i + 1]) for i in range(N_WORKER)
             if bounds[i + 1] > bounds[i]]

    var_s = {m: np.empty(n) for m in MEMBERS}
    mse_s = {m: np.empty(n) for m in MEMBERS}
    acc   = {c: np.zeros(3) for c in CLASS_ID}

    ctx = mp.get_context('fork')
    lock = ctx.RLock()
    tqdm.set_lock(lock)
    with ProcessPoolExecutor(max_workers=N_WORKER, mp_context=ctx,
                             initializer=tqdm.set_lock,
                             initargs=(lock,)) as ex:
        for fut in as_completed([ex.submit(_worker, t) for t in tasks]):
            s, e, var, mse, a = fut.result()
            for m in MEMBERS:
                var_s[m][s:e] = var[m]
                mse_s[m][s:e] = mse[m]
            for c in CLASS_ID:
                acc[c] += a[c]
    print('\n' * len(tasks))

    # ---- 成员数扫描 ----
    rng  = np.random.default_rng(0)
    boot = rng.integers(0, n, size=(N_BOOT, n))
    sweep = []
    print(f'CRR-LDM-{model}  {"N":>3} {"spread":>8} {"rmse":>8} '
          f'{"ssrat":>7} {"crps":>8}')
    for m in MEMBERS:
        sp = np.sqrt(var_s[m].mean() * (m + 1) / m) if m > 1 else np.nan
        rm = np.sqrt(mse_s[m].mean())
        bs = np.sqrt(mse_s[m][boot].mean(axis=1))
        lo, hi = np.percentile(bs, [2.5, 97.5])

        cv = crps_store.get(m)
        if cv is not None:
            cr = float(cv.mean())
            cb = cv[boot].mean(axis=1)
            clo, chi = np.percentile(cb, [2.5, 97.5])
        else:
            cr = clo = chi = np.nan

        sweep.append([f'CRR-LDM-{model}', m, sp, rm, sp / rm, cr,
                      lo, hi, clo, chi])
        print(f'{"":>13} {m:>3} {sp:>8.4f} {rm:>8.4f} '
              f'{sp / rm:>7.3f} {cr:>8.4f}')

    r1, r30 = np.sqrt(mse_s[1].mean()), np.sqrt(mse_s[REF_M].mean())
    print(f'[{model}] RMSE  m=1 -> {REF_M}: {(r1 - r30) / r1 * 100:.2f}%')
    if 50 in MEMBERS:
        r50 = np.sqrt(mse_s[50].mean())
        print(f'[{model}] RMSE  m={REF_M} -> 50: '
              f'{(r30 - r50) / r30 * 100:.3f}%')
    if 1 in crps_store and REF_M in crps_store:
        c1, c30 = crps_store[1].mean(), crps_store[REF_M].mean()
        print(f'[{model}] CRPS  m=1 -> {REF_M}: {(c1 - c30) / c1 * 100:.2f}%')
        if 50 in crps_store:
            c50 = crps_store[50].mean()
            print(f'[{model}] CRPS  m={REF_M} -> 50: '
                  f'{(c30 - c50) / c30 * 100:.3f}%')

    # ---- 按云类 ----
    tot = sum(v[2] for v in acc.values())
    cls_rows = []
    print(f'\n{"class":>16} {"n_pixel":>11} {"frac%":>7} '
          f'{"spread":>8} {"rmse":>8} {"ssrat":>7}  note')
    for c, name in CLASS_ID.items():
        sv, se, npx = acc[c]
        if npx == 0:
            continue
        sp = np.sqrt(sv / npx * (REF_M + 1) / REF_M)
        rm = np.sqrt(se / npx)
        keep = npx >= MIN_PIX and name not in DROP_CLASS
        if keep:
            cls_rows.append([f'CRR-LDM-{model}', name, int(npx),
                             npx / tot * 100, sp, rm, sp / rm])
        print(f'{name:>16} {int(npx):>11} {npx / tot * 100:>7.2f} '
              f'{sp:>8.4f} {rm:>8.4f} {sp / rm:>7.3f}'
              f'  {"" if keep else "excluded"}')
    return sweep, cls_rows


# ===================================================================
# Step 3  生成一致性图
# ===================================================================
def step3_consistency(model):
    print('=' * 62)
    print(f'Step 3  generation consistency  —  CRR-LDM-{model}')
    print('=' * 62)

    idx   = list(SAMPLES.values())
    names = list(SAMPLES.keys())
    sel = np.random.default_rng().choice(PRED.shape[-1], N_GEN, replace=False)
    print(f'[{model}] members used: {sorted(sel + 1)}')

    sub = PRED[idx]                       # (7, 64, 128, 50)
    dl  = [TRUTH[idx]]
    dn  = ['CloudSat CPR_Reflectgrey']
    hl  = [None]
    for k, j in enumerate(sel, 1):
        dl.append(sub[..., j])
        dn.append(f'Gen {k}_Reflectgrey')
        hl.append(None)

    stack   = sub[..., sel]
    rng_map = stack.max(axis=-1) - stack.min(axis=-1)
    print(f'[{model}] pixel-wise range over {N_GEN} gens: '
          f'mean {rng_map.mean():.3f}, '
          f'p95 {np.percentile(rng_map, 95):.3f} dBZ')

    tag = 'F3' if model == 'Full' else 'F4'
    P = plot.Comparison(IMG_SAVE, resolution=1.1, samples=len(idx))
    P.images(dl, dn, hl, sampletitle=names,
             save_name=f'2b_{tag}_{model}_generation_consistency')
    print(f'[{model}] saved 2b_{tag}_{model}_generation_consistency')


# ===================================================================
# Step 4  出图
# ===================================================================
def step4_figures(sweep, cls_df):
    print('=' * 62)
    print('Step 4  figures')
    print('=' * 62)

    # ---- 2a_F1  CRPS：单面板双曲线 ----
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    OFF = {'Full': (0, -17), 'IR': (0, 10)}
    cmax = sweep['CRPS (dBZ)'].max()
    for model in MODELS:
        d = (sweep[sweep.Model == f'CRR-LDM-{model}']
             .dropna(subset=['CRPS (dBZ)']))
        if d.empty:
            continue
        ax.plot(d.Members, d['CRPS (dBZ)'], '-o', ms=6, lw=1.8,
                color=LINE[model], mfc=FACE[model], mec=EDGE, mew=.8,
                label=f'CRR-LDM-{model}')
        ax.fill_between(d.Members, d.crps_lo, d.crps_hi,
                        color=FACE[model], alpha=.45, lw=0)
        for xx, yy in zip(d.Members, d['CRPS (dBZ)']):
            ax.annotate(f'{yy:.3f}', (xx, yy), textcoords='offset points',
                        xytext=OFF[model], ha='center', fontsize=FS_ANNO,
                        color=LINE[model])
    ax.set_xticks(MEMBERS)
    ax.set_ylim(0, cmax * 1.45)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.set_xlabel('Number of ensemble members', fontsize=FS_LABEL)
    ax.set_ylabel('CRPS (dBZ)', fontsize=FS_LABEL)
    ax.grid(alpha=.25, lw=.7)
    ax.legend(frameon=False, loc='upper right', fontsize=FS_LEG)
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE)
        sp.set_linewidth(.9)
    fig.tight_layout()
    fig.savefig(f'{IMG_SAVE}/2a_F1_crps_vs_members.png', dpi=300)
    plt.close(fig)

    # ---- 2a_F2  spread / RMSE + SSRAT 右轴：双面板 ----
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2),
                             sharex=True, sharey=True)
    ytop = 18
    ss_max = sweep['SSRAT'].max()
    ss_top = np.ceil(ss_max * 1.6 * 10) / 10

    right_axes = []
    for ax, model in zip(axes, MODELS):
        d  = sweep[sweep.Model == f'CRR-LDM-{model}']
        dv = d.dropna(subset=['Spread (dBZ)'])
        lc, fc = LINE[model], FACE[model]

        l1, = ax.plot(d.Members, d['RMSE (dBZ)'], '-o', ms=6, lw=1.8,
                      color=lc, mfc=fc, mec=EDGE, mew=.8,
                      label='RMSE (ensemble mean)')
        l2, = ax.plot(dv.Members, dv['Spread (dBZ)'], '--s', ms=6, lw=1.5,
                      color=lc, mfc='white', mec=EDGE, mew=.8, label='Spread')

        axr = ax.twinx()
        right_axes.append(axr)
        l3, = axr.plot(dv.Members, dv['SSRAT'], ':^', ms=6, lw=1.5,
                       color='0.35', mfc='0.75', mec=EDGE, mew=.8,
                       label='SSRAT')
        axr.set_ylim(0, ss_top)
        axr.tick_params(axis='y', labelsize=FS_TICK, colors='0.30')

        for xx, yy in zip(d.Members, d['RMSE (dBZ)']):
            ax.annotate(f'{yy:.3f}', (xx, yy), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=FS_ANNO, color=lc)
        for xx, yy in zip(dv.Members, dv['Spread (dBZ)']):
            ax.annotate(f'{yy:.3f}', (xx, yy), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=FS_ANNO, color=lc)
        for xx, yy in zip(dv.Members, dv['SSRAT']):
            axr.annotate(f'{yy:.3f}', (xx, yy), textcoords='offset points',
                         xytext=(0, 8), ha='center', fontsize=FS_ANNO,
                         color='0.30')

        ax.set_ylim(0, ytop)
        ax.set_yticks(np.arange(0, ytop + .1, F2_YTICK))
        ax.set_xticks(MEMBERS)
        ax.tick_params(axis='both', labelsize=FS_TICK)
        ax.set_xlabel('Number of ensemble members', fontsize=FS_LABEL)
        ax.set_title(f'CRR-LDM-{model}', fontsize=FS_TITLE)
        ax.grid(alpha=.25, lw=.7)
        ax.legend([l1, l2, l3], [t.get_label() for t in (l1, l2, l3)],
                  frameon=False, loc='upper left', fontsize=FS_LEG)
        for sp in ax.spines.values():
            sp.set_edgecolor(EDGE)
            sp.set_linewidth(.9)

    axes[0].set_ylabel('dBZ', fontsize=FS_LABEL)
    right_axes[0].set_yticklabels([])
    right_axes[1].set_ylabel('SSRAT', fontsize=FS_LABEL, color='0.30')
    fig.tight_layout()
    fig.savefig(f'{IMG_SAVE}/2a_F2_spread_rmse_vs_members.png', dpi=300)
    plt.close(fig)

    # ---- 2c_F5  按云类：双面板 ----
    order = [c for c in CLASS_ORDER if c in set(cls_df['Cloud class'])]
    x, wd = np.arange(len(order)), 0.36
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6), sharey=True)

    for ax, model in zip(axes, MODELS):
        d = (cls_df[cls_df.Model == f'CRR-LDM-{model}']
             .set_index('Cloud class').reindex(order))
        lc, fc = LINE[model], FACE[model]

        ax.bar(x - wd / 2, d['RMSE (dBZ)'], wd, color=fc,
               edgecolor=EDGE, lw=.8, label='RMSE (ensemble mean)')
        ax.bar(x + wd / 2, d['Spread (dBZ)'], wd, color=fc,
               edgecolor=EDGE, lw=.8, alpha=.55, hatch='//', label='Spread')

        for xx, rm, sp_, ss in zip(x, d['RMSE (dBZ)'],
                                   d['Spread (dBZ)'], d['SSRAT']):
            ax.annotate(f'{rm:.2f}', (xx - wd / 2, rm),
                        textcoords='offset points', xytext=(0, 3),
                        ha='center', fontsize=FS_ANNO, color=EDGE)
            ax.annotate(f'{sp_:.2f}', (xx + wd / 2, sp_),
                        textcoords='offset points', xytext=(0, 3),
                        ha='center', fontsize=FS_ANNO, color=EDGE)
            ax.annotate(f'SSRAT={ss:.2f}', (xx, rm),
                        textcoords='offset points', xytext=(0, 16),
                        ha='center', va='bottom', fontsize=FS_ANNO, color=lc)

        ax.set_ylim(*F5_YLIM)
        ax.set_yticks(np.arange(F5_YLIM[0], F5_YLIM[1] + .1, F5_YTICK))
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha='right', fontsize=FS_TICK)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.set_title(f'CRR-LDM-{model}', fontsize=FS_TITLE)
        ax.grid(alpha=.25, axis='y', lw=.7)
        ax.legend(frameon=False, loc='upper left', fontsize=FS_LEG)
        for sp in ax.spines.values():
            sp.set_edgecolor(EDGE)
            sp.set_linewidth(.9)

    axes[0].set_ylabel('dBZ', fontsize=FS_LABEL)
    fig.tight_layout()
    fig.savefig(f'{IMG_SAVE}/2c_F5_spread_rmse_by_cloudclass.png', dpi=300)
    plt.close(fig)
    print('figures saved')


# ===================================================================
if __name__ == '__main__':
    crps_all = step1_crps()

    sweep_rows, class_rows = [], []
    for model in MODELS:
        load_model(model)
        a, b = step2_spread_rmse(model, crps_all[model])
        sweep_rows += a
        class_rows += b
        step3_consistency(model)
        free_model()
        print(f'[{model}] memory released\n')

    sweep = pd.DataFrame(sweep_rows, columns=[
        'Model', 'Members', 'Spread (dBZ)', 'RMSE (dBZ)', 'SSRAT',
        'CRPS (dBZ)', 'rmse_lo', 'rmse_hi', 'crps_lo', 'crps_hi'])
    cls_df = pd.DataFrame(class_rows, columns=[
        'Model', 'Cloud class', 'Pixels', 'Fraction (%)',
        'Spread (dBZ)', 'RMSE (dBZ)', 'SSRAT'])

    with pd.ExcelWriter(OUT_XLSX) as w:
        (sweep.drop(columns=['rmse_lo', 'rmse_hi', 'crps_lo', 'crps_hi'])
         .round({'Spread (dBZ)': 2, 'RMSE (dBZ)': 2, 'SSRAT': 3,
                 'CRPS (dBZ)': 3})
         .to_excel(w, sheet_name='member_sweep', index=False))
        (cls_df.round({'Fraction (%)': 2, 'Spread (dBZ)': 2,
                       'RMSE (dBZ)': 2, 'SSRAT': 3})
         .to_excel(w, sheet_name=f'by_cloud_class_m{REF_M}', index=False))
    print(f'saved: {OUT_XLSX}')

    step4_figures(sweep, cls_df)
