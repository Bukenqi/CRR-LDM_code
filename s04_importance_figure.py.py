# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:22:37 2026

@author: Admin
"""

# -*- coding: utf-8 -*-
"""s04_importance_figure.py —— 置换检验 + 敏感性分析合并图（ΔRMSE）

只读 s02 / s03 输出的 Excel，不重算。
每个模型一行，左侧窄列为置换 ΔRMSE，右侧九列为 ±σ 扰动 ΔRMSE，
两者统一以该模型的 baseline 为参照，色标按模型独立归一化。
"""
# -*- coding: utf-8 -*-
"""s04_importance_figure.py —— 置换检验 + 敏感性分析合并图（ΔRMSE）

只读 s02 / s03 输出的 Excel，不重算。
两个模型左右并排，各自内部为「置换 ΔRMSE 窄列 + 九档扰动 ΔRMSE」，
统一以该模型的 baseline 为参照，色标按模型独立归一化，不做跨模型比较。
"""
# -*- coding: utf-8 -*-
"""s04_importance_figure.py —— 置换检验 + 敏感性分析合并图（ΔRMSE）

只读 s02 / s03 输出的 Excel，不重算。
两个模型左右并排，各自内部为「置换 ΔRMSE 窄列 + 九档扰动 ΔRMSE」，
统一以该模型的 baseline 为参照，色标按模型独立归一化，不做跨模型比较。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from s00_plot_style import (RC, EDGE, MODELS, DATA_PATH, IMG_SAVE,
                            CHANNEL_ORDER, SIG_LAB)

plt.rcParams.update(RC)

PERM_XLSX = f'{DATA_PATH}/permutation_importance.xlsx'
SENS_XLSX = f'{DATA_PATH}/sensitivity_rmse.xlsx'
OUT_PNG   = f'{IMG_SAVE}/3_F6_importance_sensitivity.png'

CMAP = {'Full': LinearSegmentedColormap.from_list(
            'full', ['#ffffff', 'lightpink', '#c9506b', '#7d2b3f']),
        'IR': LinearSegmentedColormap.from_list(
            'ir', ['#ffffff', 'skyblue', '#2f6f96', '#1a4560'])}

FS_TICK, FS_LAB, FS_ANNO, FS_CB = 22, 23, 17.5, 27
ROW_H  = 0.62                     # Full 面板每行高度（英寸）
LAB_SIDE = {'Full': 'left', 'IR': 'right'}


def pretty(c):
    """albedo_03 → albedo 03；z_tbb_13~15 → tbb 13~15"""
    return c.removeprefix('z_').replace('_', ' ')


def load():
    perm = pd.read_excel(PERM_XLSX)
    sens = pd.read_excel(SENS_XLSX)
    miss = [c for c in SIG_LAB if c not in sens.columns]
    if miss:
        raise SystemExit(f'{SENS_XLSX} missing columns: {miss}')

    out = {}
    for m in MODELS:
        tag = f'CRR-LDM-{m}'
        brow = perm[(perm.Model == tag) & (perm.Channel == 'baseline')]
        if brow.empty:
            raise SystemExit(f'{tag}: baseline missing in {PERM_XLSX}')
        base = float(brow['RMSE (dBZ)'].iloc[0])

        p = (perm[(perm.Model == tag) & (perm.Channel != 'baseline')]
             .set_index('Channel'))
        s = sens[sens.Model == tag].set_index('Channel')
        chs = [c for c in CHANNEL_ORDER if c in set(p.index) | set(s.index)]
        if not chs:
            raise SystemExit(f'{tag}: no channel matched CHANNEL_ORDER')

        out[m] = dict(
            base=base,
            chs=chs,
            perm=p.reindex(chs)['ΔRMSE (dBZ)'].to_numpy(float).reshape(-1, 1),
            sens=s.reindex(chs)[SIG_LAB].to_numpy(float) - base)
    return out


def draw_block(ax, arr, cmap, vmax, xlabels, ylabels=None, yside='left'):
    im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isfinite(v):
                continue
            v = 0.0 if abs(v) < 5e-3 else v          # 避免出现 -0.00
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=FS_ANNO,
                    color='w' if v / vmax > .58 else EDGE)

    ax.set_xticks(np.arange(-.5, arr.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, arr.shape[0], 1), minor=True)
    ax.grid(which='minor', color=EDGE, lw=.4, alpha=.35)
    ax.tick_params(which='minor', length=0)

    ax.set_xticks(range(arr.shape[1]))
    ax.set_xticklabels(xlabels, fontsize=FS_TICK)
    if ylabels is None:
        ax.set_yticks([])
    else:
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=FS_TICK)
        if yside == 'right':
            ax.yaxis.tick_right()

    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE)
        sp.set_linewidth(.9)
    return im


if __name__ == '__main__':
    D = load()
    nmax = max(len(D[m]['chs']) for m in MODELS)
    nsig = len(SIG_LAB)
    
    fig = plt.figure(figsize=(10.5 * len(MODELS), ROW_H * nmax + 2.6))
    
    # 每个模型占「置换 1 列 + 扰动 nsig 列」，模型之间留空隙
    widths, gap = [], 0.1
    for i in range(len(MODELS)):
        if i:
            widths.append(gap)
        widths += [1, nsig]
    gs = fig.add_gridspec(2, len(widths),
                          width_ratios=widths, height_ratios=[.022, 1],
                          wspace=.06, hspace=.045,
                          left=.085, right=.915, top=.93, bottom=.09)

    for i, m in enumerate(MODELS):
        d = D[m]
        vmax = float(np.nanmax([np.nanmax(d['perm']), np.nanmax(d['sens'])]))
        vmax = np.ceil(vmax * 2) / 2
        c0 = i * 3                               # 该模型置换列所在的 grid 列

        ax_p = fig.add_subplot(gs[1, c0])
        ax_s = fig.add_subplot(gs[1, c0 + 1])
        side = LAB_SIDE.get(m, 'left')
        names = [pretty(c) for c in d['chs']]

        if side == 'right':
            draw_block(ax_p, d['perm'], CMAP[m], vmax, ['permuted'])
            im = draw_block(ax_s, d['sens'], CMAP[m], vmax, SIG_LAB,
                            names, 'right')
        else:
            draw_block(ax_p, d['perm'], CMAP[m], vmax, ['permuted'], names)
            im = draw_block(ax_s, d['sens'], CMAP[m], vmax, SIG_LAB)

        ax_s.set_xlabel('Perturbation amplitude', fontsize=FS_LAB, labelpad=6)

        cax = fig.add_subplot(gs[0, c0:c0 + 2])
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.tick_params(labelsize=FS_TICK, length=3, pad=2)
        cb.outline.set_edgecolor(EDGE)
        cax.set_title(f'CRR-LDM-{m}   ΔRMSE (dBZ)',
                      fontsize=FS_CB, pad=44)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    print(f'saved: {OUT_PNG}\n')

    # ---- 供核对：0σ 残差与各模型极值 ----
    zi = nsig // 2
    for m in MODELS:
        d = D[m]
        s = d['sens'][:, zi]
        print(f'[{m}] baseline={d["base"]:.4f} dBZ, {len(d["chs"])} channels')
        print(f'      {SIG_LAB[zi]} residual: '
              f'{np.nanmin(s):+.4f} .. {np.nanmax(s):+.4f} dBZ')
        print(f'      permutation ΔRMSE max {np.nanmax(d["perm"]):.4f}, '
              f'sensitivity ΔRMSE max {np.nanmax(d["sens"]):.4f}')


