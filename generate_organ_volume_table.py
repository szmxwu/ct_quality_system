#!/usr/bin/env python3
"""
基于stage0过滤结果生成器官体积详表
排除outliers，每行一个mask文件，包含117个器官的体积
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import signal

sys.path.insert(0, os.path.dirname(__file__))
from Seg_lables import totalsegmentator_dict
from match_data import load_mask_array

# 体素大小: x=1mm, y=1mm, z=3mm
VOXEL_VOLUME_MM3 = 1.0 * 1.0 * 3.0  # 3 mm³
VOXEL_VOLUME_ML = VOXEL_VOLUME_MM3 / 1000.0  # 0.003 ml

# 器官列表（按label值排序）
ORGAN_NAMES = [name for name, label in sorted(totalsegmentator_dict.items(), key=lambda x: x[1])]

# Label到器官名称的映射
LABEL_TO_ORGAN = {v: k for k, v in totalsegmentator_dict.items()}


def load_outlier_cases(outlier_csv: str) -> set:
    """加载需要排除的outlier cases (accession_number, series_number)"""
    df = pd.read_csv(outlier_csv)
    outlier_set = set()
    for _, row in df.iterrows():
        key = (row['accession_number'], str(row['series_number']))
        outlier_set.add(key)
    print(f"[INFO] 从 {outlier_csv} 加载了 {len(outlier_set)} 个 outlier cases")
    return outlier_set


def compute_organ_volumes_fast(mask_path: str) -> dict:
    """快速计算mask中所有器官的体积（单位：ml）
    
    使用np.bincount直接计算各label的体素数，避免创建多个二值mask
    """
    volumes = {name: 0.0 for name in ORGAN_NAMES}
    
    try:
        mask = load_mask_array(mask_path)
        # 将mask转为int64用于bincount
        mask_int = mask.astype(np.int64, copy=False)
        # 计算各label的体素数
        counts = np.bincount(mask_int.ravel())
        
        # 将计数转换为体积 (跳过背景0)
        for label in range(1, len(counts)):
            if counts[label] > 0 and label in LABEL_TO_ORGAN:
                organ_name = LABEL_TO_ORGAN[label]
                volumes[organ_name] = counts[label] * VOXEL_VOLUME_ML
    except Exception as e:
        print(f"[ERROR] 处理 {mask_path}: {e}")
    
    return volumes


def init_worker():
    """初始化worker进程，忽略SIGINT信号，让主进程处理"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_row(args):
    """处理单行数据（用于多进程）"""
    idx, row = args
    accession = row['accession_number']
    series = str(row['series_number'])
    mask_path = row['mask_path']
    ct_path = row['ct_path']
    mask_exists = row['mask_exists']
    ct_exists = row['ct_exists']
    
    result = {
        'accession_number': accession,
        'series_number': series,
        'mask_path': mask_path,
        'ct_path': ct_path,
        'mask_exists': mask_exists,
        'ct_exists': ct_exists,
    }
    
    # 如果mask存在，计算各器官体积
    if mask_exists and os.path.exists(mask_path):
        volumes = compute_organ_volumes_fast(mask_path)
        result.update(volumes)
    else:
        # mask不存在，所有器官体积为0
        for name in ORGAN_NAMES:
            result[name] = 0.0
    
    return result


def generate_volume_table(
    file_index_csv: str,
    outlier_csv: str,
    output_csv: str,
    num_workers: int = 4,
    sample_size: int = None
):
    """生成器官体积详表"""
    
    print(f"[INFO] 读取文件索引: {file_index_csv}")
    df_index = pd.read_csv(file_index_csv, low_memory=False)
    print(f"[INFO] 原始文件数: {len(df_index)}")
    
    # 加载outliers并过滤
    outlier_set = load_outlier_cases(outlier_csv)
    
    # 过滤掉outliers
    mask = df_index.apply(
        lambda row: (row['accession_number'], str(row['series_number'])) not in outlier_set,
        axis=1
    )
    df_filtered = df_index[mask].copy()
    print(f"[INFO] 过滤outliers后剩余: {len(df_filtered)}")
    
    # 如果指定了样本大小，只取前N个（用于测试）
    if sample_size is not None and sample_size > 0:
        df_filtered = df_filtered.head(sample_size)
        print(f"[INFO] 测试模式：只处理前 {sample_size} 个文件")
    
    # 准备输出列
    output_columns = [
        'accession_number', 'series_number', 
        'mask_path', 'ct_path',
        'mask_exists', 'ct_exists'
    ] + ORGAN_NAMES
    
    # 多进程处理
    print(f"[INFO] 使用 {num_workers} 个worker处理...")
    
    rows = list(df_filtered.iterrows())
    results = []
    
    try:
        with mp.Pool(num_workers, initializer=init_worker) as pool:
            for result in tqdm(
                pool.imap(process_row, rows, chunksize=50),
                total=len(rows),
                desc="计算器官体积"
            ):
                results.append(result)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，正在保存已处理的数据...")
    
    # 生成DataFrame并保存
    df_output = pd.DataFrame(results)
    
    # 确保列顺序正确
    df_output = df_output[output_columns]
    
    # 保存CSV
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df_output.to_csv(output_csv, index=False, float_format='%.6f')
    
    print(f"\n[INFO] 结果已保存到: {output_csv}")
    print(f"[INFO] 总行数: {len(df_output)}")
    print(f"[INFO] 总列数: {len(df_output.columns)} (6个基础列 + 117个器官体积列)")
    
    # 生成统计信息
    print("\n[INFO] 器官体积统计（非零值，前10个器官）：")
    for organ in ORGAN_NAMES[:10]:
        non_zero = df_output[df_output[organ] > 0][organ]
        if len(non_zero) > 0:
            print(f"  {organ}: count={len(non_zero)}, median={non_zero.median():.2f}ml, max={non_zero.max():.2f}ml")
        else:
            print(f"  {organ}: count=0 (未检测到)")
    
    return df_output


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成器官体积详表')
    parser.add_argument('--file-index', default='outputs/stage0_数据探索/file_index.csv',
                        help='文件索引CSV路径')
    parser.add_argument('--outliers', default='outputs/stage0_数据探索/stage0_outliers.csv',
                        help='Outliers CSV路径')
    parser.add_argument('--output', default='outputs/organ_volume_table.csv',
                        help='输出CSV路径')
    parser.add_argument('--workers', type=int, default=4,
                        help='并行worker数量')
    parser.add_argument('--sample', type=int, default=None,
                        help='测试模式：只处理前N个文件')
    
    args = parser.parse_args()
    
    generate_volume_table(
        args.file_index,
        args.outliers,
        args.output,
        args.workers,
        args.sample
    )
