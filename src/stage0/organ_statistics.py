"""
阶段0 - 器官统计相关任务 (0.2-0.4)
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import logging
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from match_data import get_mask_labels, get_mask_label_counts
from src.utils.organ_mapping import get_organ_name, LABEL_TO_ORGAN
from src.utils.file_io import load_npz_keys
from Seg_lables import totalsegmentator_dict
logger = logging.getLogger(__name__)


def _occurrence_worker(mask_path: str) -> list:
    try:
        labels = get_mask_labels(mask_path)
        return labels.tolist()
    except Exception as e:
        logger.warning(f"处理文件失败 {mask_path}: {e}")
        return []


def compute_organ_occurrence(file_index_path: str, output_path: str = None,
                             sample_size: int = None, num_workers: int = None,
                             chunksize: int = 8) -> pd.DataFrame:
    """
    任务0.2: 器官出现频率统计
    
    Args:
        file_index_path: 文件索引表路径
        output_path: 输出CSV路径
        sample_size: 测试模式样本数量
    
    Returns:
        pd.DataFrame: 器官出现频率统计表
    """
    logger.info("=" * 70)
    logger.info("任务0.2: 器官出现频率统计")
    logger.info("=" * 70)
    
    # 加载索引表
    df_index = pd.read_csv(file_index_path)
    
    # 只处理ct_exists=True的记录
    df_valid = df_index[df_index['ct_exists'] == True]
    total_files = len(df_valid)
    
    logger.info(f"有效样本数: {total_files:,}")
    
    if sample_size:
        df_valid = df_valid.head(sample_size)
        logger.info(f"测试模式: 处理 {len(df_valid)} 个样本")
    
    # 统计器官出现次数
    organ_counts = defaultdict(int)
    if num_workers is None:
        num_workers = max(1, min(4, mp.cpu_count() - 1))

    mask_paths = df_valid['mask_path'].tolist()
    with mp.get_context("fork").Pool(processes=num_workers) as pool:
        for labels in tqdm(pool.imap_unordered(_occurrence_worker, mask_paths, chunksize=chunksize),
                           total=len(mask_paths), desc="统计器官出现频率"):
            for label in labels:
                organ_name = LABEL_TO_ORGAN.get(int(label))
                if organ_name:
                    organ_counts[organ_name] += 1
    
    # 创建统计表
    records = []
    for organ_name, count in organ_counts.items():
        label = totalsegmentator_dict.get(organ_name)
        if label:
            occurrence_rate = count / total_files * 100
            records.append({
                'organ_name': organ_name,
                'organ_label': label,
                'occurrence_count': count,
                'occurrence_rate': occurrence_rate
            })
    
    df_result = pd.DataFrame(records)
    df_result = df_result.sort_values('occurrence_count', ascending=False).reset_index(drop=True)
    
    # 打印统计结果
    logger.info("\n最常见器官TOP10:")
    for idx, row in df_result.head(10).iterrows():
        logger.info(f"  {idx+1}. {row['organ_name']}: {row['occurrence_count']:,} ({row['occurrence_rate']:.2f}%)")
    
    logger.info("\n最罕见器官TOP10:")
    for idx, row in df_result.tail(10).iterrows():
        logger.info(f"  {idx+1}. {row['organ_name']}: {row['occurrence_count']:,} ({row['occurrence_rate']:.2f}%)")
    
    logger.info(f"\n检测到器官种类数: {len(df_result)}")
    
    # 保存结果
    if output_path:
        df_result.to_csv(output_path, index=False)
        logger.info(f"\n结果已保存: {output_path}")
    
    return df_result


def _volume_worker(args: tuple) -> dict:
    mask_path, ct_path = args
    try:
        label_counts = get_mask_label_counts(mask_path)
        # ct_data = load_npz_keys(ct_path, ['spacing']) if ct_path else None
        # if ct_data is None:
        #     return {}

        spacing = 3
        voxel_volume_ml = np.prod(spacing) / 1000

        volume_by_label = {}
        for label, count in label_counts.items():
            organ_name = LABEL_TO_ORGAN.get(int(label))
            if not organ_name:
                continue
            volume_by_label[organ_name] = count * voxel_volume_ml

        return volume_by_label
    except Exception as e:
        logger.warning(f"处理文件失败 {mask_path}: {e}")
        return {}


def compute_organ_volumes(file_index_path: str, output_path: str = None,
                          sample_size: int = None, num_workers: int = None,
                          chunksize: int = 4) -> pd.DataFrame:
    """
    任务0.3: 器官体积分布统计
    
    Args:
        file_index_path: 文件索引表路径
        output_path: 输出CSV路径
        sample_size: 测试模式样本数量
    
    Returns:
        pd.DataFrame: 器官体积统计表
    """
    logger.info("=" * 70)
    logger.info("任务0.3: 器官体积分布统计")
    logger.info("=" * 70)
    
    # 加载索引表
    df_index = pd.read_csv(file_index_path)
    
    # 只处理ct_exists=True的记录
    df_valid = df_index[df_index['ct_exists'] == True]
    total_files = len(df_valid)
    
    logger.info(f"有效样本数: {total_files:,}")
    
    if sample_size:
        df_valid = df_valid.head(sample_size)
        logger.info(f"测试模式: 处理 {len(df_valid)} 个样本")
    
    # 收集体积数据
    organ_volumes = defaultdict(list)
    if num_workers is None:
        num_workers = max(1, min(4, mp.cpu_count() - 1))

    work_items = list(zip(df_valid['mask_path'].tolist(), df_valid['ct_path'].tolist()))
    with mp.get_context("fork").Pool(processes=num_workers) as pool:
        for volume_by_label in tqdm(pool.imap_unordered(_volume_worker, work_items, chunksize=chunksize),
                                    total=len(work_items), desc="计算器官体积"):
            if not volume_by_label:
                continue
            for organ_name, volume_ml in volume_by_label.items():
                organ_volumes[organ_name].append(volume_ml)
    
    # 计算统计量
    records = []
    for organ_name, volumes in organ_volumes.items():
        if len(volumes) == 0:
            continue
        
        volumes_array = np.array(volumes)
        label = totalsegmentator_dict[organ_name]
        
        if label:
            records.append({
                'organ_name': organ_name,
                'sample_count': len(volumes),
                'volume_mean_ml': np.mean(volumes_array),
                'volume_std_ml': np.std(volumes_array),
                'volume_min_ml': np.min(volumes_array),
                'volume_p01_ml': np.percentile(volumes_array, 1),
                'volume_p05_ml': np.percentile(volumes_array, 5),
                'volume_p10_ml': np.percentile(volumes_array, 10),
                'volume_p25_ml': np.percentile(volumes_array, 25),
                'volume_median_ml': np.median(volumes_array),
                'volume_p75_ml': np.percentile(volumes_array, 75),
                'volume_p90_ml': np.percentile(volumes_array, 90),
                'volume_p95_ml': np.percentile(volumes_array, 95),
                'volume_p99_ml': np.percentile(volumes_array, 99),
                'volume_max_ml': np.max(volumes_array)
            })
    
    df_result = pd.DataFrame(records)
    df_result = df_result.sort_values('sample_count', ascending=False).reset_index(drop=True)
    
    # 打印主要器官的体积统计
    logger.info("\n主要器官体积统计:")
    main_organs = ['liver', 'spleen', 'kidney_left', 'kidney_right', 'heart']
    for organ in main_organs:
        organ_data = df_result[df_result['organ_name'] == organ]
        if not organ_data.empty:
            row = organ_data.iloc[0]
            logger.info(f"  {organ}:")
            logger.info(f"    样本数: {row['sample_count']:,}")
            logger.info(f"    平均体积: {row['volume_mean_ml']:.1f} ml (std: {row['volume_std_ml']:.1f})")
            logger.info(f"    中位数: {row['volume_median_ml']:.1f} ml")
            logger.info(f"    范围: [{row['volume_min_ml']:.1f}, {row['volume_max_ml']:.1f}] ml")
    
    # 保存结果
    if output_path:
        df_result.to_csv(output_path, index=False)
        logger.info(f"\n结果已保存: {output_path}")
    
    return df_result


def _cooccurrence_worker(mask_path: str) -> list:
    try:
        labels = get_mask_labels(mask_path)
        return labels.tolist()
    except Exception as e:
        logger.warning(f"处理文件失败 {mask_path}: {e}")
        return []


def compute_organ_cooccurrence(file_index_path: str, output_path: str = None,
                               sample_size: int = None, num_workers: int = None,
                               chunksize: int = 8) -> pd.DataFrame:
    """
    任务0.4: 器官共现矩阵
    
    Args:
        file_index_path: 文件索引表路径
        output_path: 输出CSV路径
        sample_size: 测试模式样本数量
    
    Returns:
        pd.DataFrame: 器官共现矩阵
    """
    logger.info("=" * 70)
    logger.info("任务0.4: 器官共现矩阵")
    logger.info("=" * 70)
    
    # 加载索引表
    df_index = pd.read_csv(file_index_path)
    
    # 只处理ct_exists=True的记录
    df_valid = df_index[df_index['ct_exists'] == True]
    total_files = len(df_valid)
    
    logger.info(f"有效样本数: {total_files:,}")
    
    if sample_size:
        df_valid = df_valid.head(sample_size)
        logger.info(f"测试模式: 处理 {len(df_valid)} 个样本")
    
    # 创建117×117的共现矩阵
    max_label = 117
    cooccurrence_matrix = np.zeros((max_label + 1, max_label + 1), dtype=np.int32)

    if num_workers is None:
        num_workers = max(1, min(4, mp.cpu_count() - 1))

    mask_paths = df_valid['mask_path'].tolist()
    with mp.get_context("fork").Pool(processes=num_workers) as pool:
        for labels in tqdm(pool.imap_unordered(_cooccurrence_worker, mask_paths, chunksize=chunksize),
                           total=len(mask_paths), desc="计算共现矩阵"):
            if not labels:
                continue
            # 更新共现矩阵
            for i in labels:
                for j in labels:
                    cooccurrence_matrix[int(i), int(j)] += 1
    
    # 只保存上三角矩阵的非零元素
    records = []
    for i in range(1, max_label + 1):
        for j in range(i, max_label + 1):
            count = cooccurrence_matrix[i, j]
            if count > 0:
                organ1_name = LABEL_TO_ORGAN.get(i, f"organ_{i}")
                organ2_name = LABEL_TO_ORGAN.get(j, f"organ_{j}")
                cooccurrence_rate = count / total_files * 100
                records.append({
                    'organ1_label': i,
                    'organ1_name': organ1_name,
                    'organ2_label': j,
                    'organ2_name': organ2_name,
                    'cooccurrence_count': count,
                    'cooccurrence_rate': cooccurrence_rate
                })
    
    df_result = pd.DataFrame(records)
    df_result = df_result.sort_values('cooccurrence_count', ascending=False).reset_index(drop=True)
    
    # 打印共现率最高的组合
    logger.info("\n共现率最高的器官组合TOP10:")
    for idx, row in df_result.head(10).iterrows():
        if row['organ1_name'] != row['organ2_name']:  # 排除自身
            logger.info(f"  {idx+1}. {row['organ1_name']} - {row['organ2_name']}: "
                       f"{row['cooccurrence_count']:,} ({row['cooccurrence_rate']:.2f}%)")
    
    # 保存结果
    if output_path:
        df_result.to_csv(output_path, index=False)
        logger.info(f"\n结果已保存: {output_path}")
    
    return df_result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='器官统计任务')
    parser.add_argument('--task', type=str, choices=['occurrence', 'volumes', 'cooccurrence'],
                        required=True, help='要执行的任务')
    parser.add_argument('--index', type=str, default='outputs/stage0_数据探索/file_index.csv',
                        help='文件索引表路径')
    parser.add_argument('--output', type=str, required=True, help='输出CSV路径')
    parser.add_argument('--sample', type=int, default=None, help='测试模式样本数量')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.task == 'occurrence':
        compute_organ_occurrence(args.index, args.output, args.sample)
    elif args.task == 'volumes':
        compute_organ_volumes(args.index, args.output, args.sample)
    elif args.task == 'cooccurrence':
        compute_organ_cooccurrence(args.index, args.output, args.sample)
