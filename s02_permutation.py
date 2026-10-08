# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:43:53 2026

@author: 59278
"""

"""s02_permutation.py —— 置换检验：柱状图 + 汇总表"""
import os
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '1'

from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import FUNC_read_data as read
from s00_plot_style import (FACE, LINE, EDGE, RC, NO_ECHO, MODELS,
                            DATA_PATH, IMG_SAVE, OR_DATA,
                            CHANNEL_ORDER as ORDER)

plt.rcParams.update(RC)

OUT_XLSX = f'{DATA_PATH}/permutation_importance.xlsx'
OUT_NPZ  = f'{DATA_PATH}/permutation_rmse_raw.npz'

N_RUN    = 30                     # 置换次数与基线成员数统一为 30
N_WORKER = 8                      # 受共享存储带宽限制，8 通常优于 16

RUN_RE = re.compile(r'_random_(\d+)\.nc$')

_SLOT = None
TRUTH = None                      # fork 共享的真值（float32）


# ================= 工具函数 =================
def _init(slot_queue, lock, truth):
    global _SLOT, TRUTH
    _SLOT = slot_queue.get()
    TRUTH = truth
    tqdm.set_lock(lock)


def count_rmse(gen):
    """逐样本 RMSE：全像元，与 fork 共享的 TRUTH 比较。

    差值在 float32 上算，逐样本求和时升 float64，避免整块 float64 副本。
    """
    g = np.nan_to_num(gen, nan=NO_ECHO)
    if g.dtype != np.float32:
        g = g.astype(np.float32, copy=False)
    d = g - TRUTH                                   # float32，与输入同量级
    np.square(d, out=d)                             # 原地平方，不再开新数组
    return np.sqrt(d.sum(axis=(1, 2), dtype=np.float64) / (d.shape[1] * d.shape[2]))


def run_index(p):
    m = RUN_RE.search(p.name)
    return int(m.group(1)) if m else 0


def list_runs(d):
    """取 _random_1 .. _random_30，丢弃无后缀文件与其它 nc。"""
    files = [f for f in Path(d).glob('*.nc') if RUN_RE.search(f.name)]
    return sorted(files, key=run_index)[:N_RUN]


def process_dir(args):
    """一个变量目录 → (model, channel, (N_RUN, n_sample) 矩阵)。"""
    dir_str, model, ch = args
    d = Path(dir_str)
    tag = f'{model[:2]}:{ch}'

    if ch == 'original':
        f = d / f'CRRLDM-{model}_49time.nc'
        gen = read.read_data_nc(str(f), ['Gen_result'],
                                dtype=np.float32)['Gen_result']
        rows = [count_rmse(gen[..., i])
                for i in tqdm(range(N_RUN), desc=f'{tag:22s}',
                              position=_SLOT, leave=False, ncols=78)]
        del gen
    else:
        rows = []
        for f in tqdm(list_runs(d), desc=f'{tag:22s}',
                      position=_SLOT, leave=False, ncols=78):
            g = read.read_data_nc(str(f), ['Gen_result'],
                                  dtype=np.float32)['Gen_result']
            rows.append(count_rmse(g[..., 0] if g.ndim == 4 else g))
            del g

    mat = np.stack(rows)
    assert mat.shape[0] == N_RUN and mat.shape[1] > 1000, \
        f'{model} {ch}: unexpected shape {mat.shape}'
    return model, ch, mat


def collect_tasks():
    tasks = []
    for d in sorted(Path(DATA_PATH).iterdir()):
        if not d.is_dir() or '_test_2020_' not in d.name:
            continue
        model = next((m for m in MODELS
                      if d.name.startswith(f'CRRLDM-{m}_')), None)
        if model is None:
            continue
        ch = d.name.split('_test_2020_', 1)[1]
        if ch == 'original':
            if (d / f'CRRLDM-{model}_49time.nc').exists():
                tasks.append((str(d), model, ch))
        else:
            runs = list_runs(d)
            if len(runs) == N_RUN:
                tasks.append((str(d), model, ch))
            elif runs:
                print(f'  skip {model} {ch}: only {len(runs)} runs found')
    return tasks


def load_truth(ref_shape):
    """读一次真值，对齐到生成结果的 (sample, height, along-track)。"""
    a = read.read_data_nc(OR_DATA, ['Radar_Reflectivity'],
                          dtype=np.float32)['Radar_Reflectivity']
    if a.shape[:3] != ref_shape:
        if a.shape[0] == ref_shape[0] and a.shape[1:3] == ref_shape[1:3][::-1]:
            a = np.swapaxes(a, 1, 2)
        else:
            raise ValueError(f'truth shape {a.shape} vs {ref_shape}')
    return np.ascontiguousarray(np.nan_to_num(a, nan=NO_ECHO),
                                dtype=np.float32)


# ================= 主流程 =================
if __name__ == '__main__':
    tasks = collect_tasks()
    if not tasks:
        raise SystemExit('no task found')

    # 用任一目录的首个文件确定形状，再读真值
    probe_dir, probe_model, probe_ch = tasks[0]
    pf = (Path(probe_dir) / f'CRRLDM-{probe_model}_49time.nc'
          if probe_ch == 'original' else list_runs(Path(probe_dir))[0])
    t0 = time.time()
    pg = read.read_data_nc(str(pf), ['Gen_result'],
                           dtype=np.float32)['Gen_result']
    print(f'probe read: {pg.nbytes / 1e9:.2f} GB in {time.time() - t0:.1f} s')
    ref_shape = pg.shape[:3]
    del pg

    truth = load_truth(ref_shape)
    print(f'truth loaded: {truth.shape}, {truth.nbytes / 1e9:.2f} GB')
    print(f'{len(tasks)} variable directories, {N_WORKER} processes, '
          f'{N_RUN} runs each\n')

    ctx = mp.get_context('fork')          # fork 共享 truth，不走 pickle
    slots = ctx.Queue()
    for i in range(N_WORKER):
        slots.put(i)
    lock = ctx.RLock()

    store, done, failed = {m: {} for m in MODELS}, [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKER, mp_context=ctx,
                             initializer=_init,
                             initargs=(slots, lock, truth)) as ex:
        futures = {ex.submit(process_dir, t): t for t in tasks}
        for k, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                model, ch, mat = fut.result()
            except Exception as e:
                failed.append(f'  {t[1]:5s} {t[2]:16s} FAILED: {e}')
                continue
            store[model][ch] = mat
            el = time.time() - t0
            done.append(f'  {model:5s} {ch:16s} '
                        f'rmse={np.sqrt((mat ** 2).mean()):.4f}')
            tqdm.write(f'[{k:2d}/{len(tasks)}] {model:5s} {ch:16s} '
                       f'elapsed {el / 60:.1f} min, '
                       f'eta {el / k * (len(tasks) - k) / 60:.1f} min')

    print('\n' * N_WORKER)
    print('\n'.join(sorted(done)))
    if failed:
        print('\n' + '\n'.join(failed))

    np.savez_compressed(OUT_NPZ, **{f'{m}__{c}': v
                                    for m in MODELS
                                    for c, v in store[m].items()})
    print(f'\nsaved raw: {OUT_NPZ}')

    # ---------------- 汇总表 ----------------
    rows = []
    for model in MODELS:
        if 'original' not in store[model]:
            print(f'[{model}] WARNING: baseline missing')
            continue
        base = np.sqrt((store[model]['original'] ** 2).mean())
        for ch, mat in store[model].items():
            rms = np.sqrt((mat ** 2).mean())
            rows.append([
                f'CRR-LDM-{model}',
                'baseline' if ch == 'original' else ch,
                round(rms, 4),
                np.nan if ch == 'original' else round(rms - base, 4),
                np.nan if ch == 'original' else round((rms - base) / base, 4),
                round(np.sqrt((mat ** 2).mean(axis=1)).std(ddof=1), 4),
            ])

    df = pd.DataFrame(rows, columns=[
        'Model', 'Channel', 'RMSE (dBZ)', 'ΔRMSE (dBZ)',
        'Relative change', 'SD over runs (dBZ)'])
    '''
    df['_k'] = df['ΔRMSE (dBZ)'].fillna(-1)
    df = (df.sort_values(['Model', '_k'], ascending=[True, False])
            .drop(columns='_k'))
    '''
    ORDER_KEY = {c: i for i, c in enumerate(ORDER)}
    df['_k'] = df.Channel.map(
        lambda c: -1 if c == 'baseline' else ORDER_KEY.get(c, 10 ** 6))
    df = df.sort_values(['Model', '_k']).drop(columns='_k')
    df.to_excel(OUT_XLSX, index=False)
    print(f'saved: {OUT_XLSX}\n')
    print(df.to_string(index=False))

    # ---------------- 柱状图 ----------------
    def pretty(c):
        """albedo_03 → albedo 03；z_tbb_13~15 → tbb 13~15"""
        return c.removeprefix('z_').replace('_', ' ')
    
    ch_df = df[df.Channel != 'baseline']
    channels = [c for c in ORDER if c in set(ch_df.Channel)]
    x, w = np.arange(len(channels)), 0.40
    
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    
    bases, handles = {}, []
    for k, model in enumerate(MODELS):
        d = (ch_df[ch_df.Model == f'CRR-LDM-{model}']
             .set_index('Channel').reindex(channels))
        brow = df[(df.Model == f'CRR-LDM-{model}') & (df.Channel == 'baseline')]
        if brow.empty:
            continue
        base = float(brow['RMSE (dBZ)'].iloc[0])
        bases[model] = base
    
        bar = ax.bar(x + (k - .5) * w, d['RMSE (dBZ)'], w * .92,
                     color=FACE[model], edgecolor=EDGE, lw=.8)
        ax.axhline(base, color=LINE[model], ls='--', lw=1.6, zorder=3)
    
        handles.append(bar)
        handles.append(Line2D([0], [0], color=LINE[model], ls='--', lw=1.6))
    
    vmax = ch_df['RMSE (dBZ)'].max()
    vmin = min(bases.values()) if bases else ch_df['RMSE (dBZ)'].min()
    ax.set_ylim(np.floor(vmin) - 1, vmax + (vmax - vmin) * 0.42)
    ax.set_xlim(-0.7, len(channels) - 0.3)
    
    labels = []
    for model in MODELS:
        if model in bases:
            labels.append(f'CRR-LDM-{model} (permuted)')
            labels.append(f'CRR-LDM-{model} baseline = {bases[model]:.2f} dBZ')
    
    ax.set_xticks(x)
    ax.set_xticklabels([pretty(c) for c in channels], rotation=45,
                       ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel('Ensemble RMSE (dBZ)', fontsize=13)
    ax.grid(alpha=.25, axis='y', lw=.7)
    ax.legend(handles, labels, frameon=False, loc='upper left', ncol=2,
              fontsize=11.5, handlelength=1.8, columnspacing=1.2)
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE)
        sp.set_linewidth(.9)
    
    fig.tight_layout(pad=0.4)
    fig.savefig(f'{IMG_SAVE}/3_F6_permutation_importance.png', dpi=300,
                bbox_inches='tight')
    print(f'\nsaved: {IMG_SAVE}/3_F6_permutation_importance.png')


