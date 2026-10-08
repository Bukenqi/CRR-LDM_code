# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:50:17 2026

@author: 59278
"""

"""CRR-LDM 图表统一风格与公共常量。"""

FACE = {'Full': 'lightpink', 'IR': 'skyblue'}
LINE = {'Full': '#d1607a',   'IR': '#3f7fa8'}
EDGE = 'black'

RC = {
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'axes.linewidth': 0.9, 'axes.edgecolor': EDGE,
}

NO_ECHO = -35.0
MODELS  = ['Full', 'IR']

DATA_PATH = '/public/home/Xiongqq/CRR-LDM_result'
IMG_SAVE  = '/public/home/Xiongqq/CRR-LDM_image'
OR_DATA   = '/public/home/Xiongqq/Data_Train/2020_Himawari_cloudsat_128_cloud_SAZ.nc'

# 已剔除 Stratus（像元数不足）
CLASS_ORDER = ['Cirrus', 'Altostratus', 'Altocumulus', 'Stratocumulus',
               'Cumulus', 'Nimbostratus', 'Deep Convection']

CLASS_ID = {0: 'Clear', 1: 'Cirrus', 2: 'Altostratus', 3: 'Altocumulus',
            4: 'Stratus', 5: 'Stratocumulus', 6: 'Cumulus',
            7: 'Nimbostratus', 8: 'Deep Convection'}

DROP_CLASS = ['Stratus'] 

CHANNEL_ORDER = ['albedo_01', 'albedo_02', 'albedo_03', 'albedo_04',
                 'albedo_05', 'albedo_06', 'z_albedo_01~03',
                 'z_albedo_04~06', 'SOZ',
                 'tbb_07', 'tbb_08', 'tbb_09', 'tbb_10', 'tbb_11', 'tbb_12',
                 'tbb_13', 'tbb_14', 'tbb_15', 'tbb_16',
                 'z_tbb_08~10', 'z_tbb_13~15']

SIGMAS  = ['sigma_-2.0', 'sigma_-1.5', 'sigma_-1.0', 'sigma_-0.5', 'sigma_0.0',
           'sigma_0.5', 'sigma_1.0', 'sigma_1.5', 'sigma_2.0']
SIG_LAB = ['−2σ', '−1.5σ', '−1σ', '−0.5σ', '0', '+0.5σ', '+1σ', '+1.5σ', '+2σ']

# 各云类代表样本（未固定种子随机抽取，已剔除 Stratus）
'''
SAMPLES = {'Cirrus': 5692, 'Altostratus': 3287, 'Altocumulus': 5668,
           'Stratocumulus': 3555, 'Cumulus': 5563, 'Nimbostratus': 4867,
           'Deep Convection': 1444}
'''
SAMPLES = {'Nimbostratus': 210,'Deep Convection': 3832}

