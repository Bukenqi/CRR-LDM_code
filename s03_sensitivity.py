# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:43:07 2026

@author: 59278
"""

# -*- coding: utf-8 -*-
"""s03_sensitivity.py —— 单通道 ±σ 扰动：热图 + 剖面图 + 汇总表"""
import os
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '1'

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import FUNC_read_data as read
import FUNC_plot_image as plot
from s00_plot_style import (RC, NO_ECHO, MODELS, DATA_PATH, IMG_SAVE, OR_DATA,
                            CHANNEL_ORDER, SIGMAS, SIG_LAB, SAMPLES, EDGE)

plt.rcParams.update(RC)

OUT_XLSX   = f'{DATA_PATH}/sensitivity_rmse.xlsx'
BLOCK      = 400
N_WORKER   = 8
VMIN, VMAX = 11, 23

VARS = {'Full': list(CHANNEL_ORDER),
        'IR': [c for c in CHANNEL_ORDER
               if (c.startswith('tbb_') and c != 'tbb_07')
               or c in ('z_tbb_08~10', 'z_tbb_13~15')]}

CMAP = {'Full': LinearSegmentedColormap.from_list(
            'full', ['#ffffff', 'lightpink', '#c9506b', '#7d2b3f']),
        'IR': LinearSegmentedColormap.from_list(
            'ir', ['#ffffff', 'skyblue', '#2f6f96', '#1a4560'])}

IDX = list(SAMPLES.values())          # 剖面图用的样本下标

_SLOT = None
TRUTH = None                          # fork 共享（float32）


def pretty(c):
    """albedo_03 → albedo 03；z_tbb_13~15 → tbb 13~15"""
    return c.removeprefix('z_').replace('_', ' ')


def _init(slot_queue, lock, truth):
    global _SLOT, TRUTH
    _SLOT = slot_queue.get()
    TRUTH = truth
    tqdm.set_lock(lock)


def load_truth():
    a = read.read_data_nc(OR_DATA, ['Radar_Reflectivity'],
                          dtype=np.float32)['Radar_Reflectivity']
    a = np.transpose(a, (0, 2, 1))
    return np.ascontiguousarray(np.nan_to_num(a, nan=NO_ECHO),
                                dtype=np.float32)


def process_var(args):
    """一个变量目录 → (model, var, 9 个 sigma 的 RMSE, 剖面切片)。"""
    root, model, var = args
    tag = f'{model[:2]}:{var}'
    try:
        f = read.search_files(root, keys=['sigma'])[0]
    except Exception as e:
        return model, var, None, None, f'no sigma file: {e}'

    res = read.read_data_nc(f, dtype=np.float32)
    n = res[SIGMAS[0]].shape[0]
    if n != TRUTH.shape[0]:
        return model, var, None, None, f'n={n} vs truth {TRUTH.shape[0]}'

    sse, npix = {s: 0.0 for s in SIGMAS}, 0
    prof = np.empty((len(SIGMAS), len(IDX)) + TRUTH.shape[1:], np.float32)

    for s_i, s in enumerate(tqdm(SIGMAS, desc=f'{tag:22s}', position=_SLOT,
                                 leave=False, ncols=78)):
        arr = res[s]
        prof[s_i] = np.nan_to_num(arr[IDX], nan=NO_ECHO)
        tot = 0
        for st in range(0, n, BLOCK):
            sl = slice(st, min(st + BLOCK, n))
            d = np.nan_to_num(arr[sl], nan=NO_ECHO) - TRUTH[sl]
            np.square(d, out=d)
            sse[s] += float(d.sum(dtype=np.float64))
            tot += d.size
        npix = tot
        res[s] = None                       # 及时释放

    rmse = [float(np.sqrt(sse[s] / npix)) for s in SIGMAS]
    return model, var, rmse, prof, None


def collect_tasks():
    tasks = []
    for model in MODELS:
        roots = read.search_files(DATA_PATH, keys=[model])
        for var in VARS[model]:
            p = next((r for r in roots
                      if Path(r).name.endswith(f'_test_2020_{var}')), None)
            if p is None:
                print(f'  missing dir: {model} {var}')
                continue
            tasks.append((p, model, var))
    return tasks


# ===================================================================
if __name__ == '__main__':
    truth = load_truth()
    print(f'truth: {truth.shape}, {truth.nbytes / 1e9:.2f} GB')

    tasks = collect_tasks()
    print(f'{len(tasks)} variable directories, {N_WORKER} processes\n')

    ctx = mp.get_context('fork')            # fork 共享 truth
    slots = ctx.Queue()
    for i in range(N_WORKER):
        slots.put(i)
    lock = ctx.RLock()

    rmse_all = {m: {} for m in MODELS}      # var → 9 个 RMSE
    prof_all = {m: {} for m in MODELS}      # var → (9, 7, H, W)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKER, mp_context=ctx,
                             initializer=_init,
                             initargs=(slots, lock, truth)) as ex:
        futs = {ex.submit(process_var, t): t for t in tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            model, var, rmse, prof, err = fut.result()
            if err:
                tqdm.write(f'  skip {model} {var}: {err}')
                continue
            rmse_all[model][var] = rmse
            prof_all[model][var] = prof
            el = time.time() - t0
            tqdm.write(f'[{k:2d}/{len(tasks)}] {model:5s} {var:16s} '
                       f'0σ={rmse[len(SIGMAS) // 2]:.4f}  '
                       f'elapsed {el / 60:.1f} min, '
                       f'eta {el / k * (len(tasks) - k) / 60:.1f} min')

    print('\n' * N_WORKER)

    # ---------------- 汇总表 ----------------
    rows = []
    for model in MODELS:
        for var in VARS[model]:
            if var in rmse_all[model]:
                rows.append([f'CRR-LDM-{model}', var] +
                            [round(v, 4) for v in rmse_all[model][var]])
    df = pd.DataFrame(rows, columns=['Model', 'Channel'] + SIG_LAB)
    df.to_excel(OUT_XLSX, index=False)
    print(f'saved: {OUT_XLSX}')

    for model in MODELS:
        c = len(SIGMAS) // 2
        z = [rmse_all[model][v][c] for v in rmse_all[model]]
        if z:
            print(f'[{model}] 0σ across {len(z)} runs: '
                  f'{min(z):.4f} – {max(z):.4f}, range {max(z) - min(z):.4f}')

    # ---------------- 3_F7 热图 ----------------
    grids = {m: dict(labels=[v for v in VARS[m] if v in rmse_all[m]])
             for m in MODELS}
    for m in MODELS:
        grids[m]['grid'] = np.array([rmse_all[m][v] for v in grids[m]['labels']])

    nmax = max(len(g['labels']) for g in grids.values())
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 0.32 * nmax + 2.0),
                             gridspec_kw=dict(wspace=.06))

    for ax, model in zip(axes, MODELS):
        g, arr = grids[model], grids[model]['grid']
        im = ax.imshow(arr, aspect='auto', cmap=CMAP[model],
                       vmin=VMIN, vmax=VMAX)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                sh = (arr[i, j] - VMIN) / (VMAX - VMIN)
                ax.text(j, i, f'{arr[i, j]:.1f}', ha='center', va='center',
                        fontsize=11, color='w' if sh > .58 else EDGE)

        ax.set_xticks(np.arange(-.5, len(SIG_LAB), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(g['labels']), 1), minor=True)
        ax.grid(which='minor', color=EDGE, lw=.4, alpha=.35)
        ax.tick_params(which='minor', length=0)

        ax.set_xticks(range(len(SIG_LAB)))
        ax.set_xticklabels(SIG_LAB, fontsize=11.5)
        ax.set_yticks(range(len(g['labels'])))
        ax.set_yticklabels([pretty(v) for v in g['labels']], fontsize=11.5)
        ax.set_xlabel('Perturbation amplitude', fontsize=12.5)

        # IR 面板的 y 轴标签移到右侧，两边标签分别朝外
        if model != MODELS[0]:
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_label_position('right')
            ax.tick_params(axis='y', labelleft=False, labelright=True)

        # colorbar 横放在面板上方，标签即模型名
        cax = ax.inset_axes([0, 1.015, 1, 0.024])
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        cb.set_label(f'CRR-LDM-{model}  RMSE (dBZ)', fontsize=12.5, labelpad=6)
        cb.ax.tick_params(labelsize=11, length=3, pad=2)
        cb.outline.set_edgecolor(EDGE)

    fig.tight_layout(pad=0.5)
    fig.savefig(f'{IMG_SAVE}/3_F7_sensitivity_rmse_heatmap.png', dpi=300,
                bbox_inches='tight')
    print('saved: 3_F7')



    """
    # ---------------- 剖面图 ----------------
    for model in MODELS:
        labels = [v for v in VARS[model] if v in prof_all[model]]
        tag = 'F8' if model == 'Full' else 'F9'
        for si, (cname, idx) in enumerate(SAMPLES.items()):
            dl, dn, hl = [], [], []
            for var in labels:
                p = prof_all[model][var]                 # (9, 7, H, W)
                dl.append(np.concatenate([truth[idx][None], p[:, si]], axis=0))
                dn.append(f'{pretty(var)}_Reflectgrey')
                hl.append(None)
            P = plot.Comparison(IMG_SAVE, resolution=1.1,
                                samples=len(SIGMAS) + 1)
            P.images(dl, dn, hl, sampletitle=['Observed'] + SIG_LAB,
                     save_name=f'3_{tag}_{model}_sensitivity_'
                               f'{cname.replace(" ", "")}_{idx}')
            print(f'  saved 3_{tag}_{model}_{cname}')
        prof_all[model].clear()
    """

