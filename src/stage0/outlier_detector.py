"""
阶段0 - 任务0.6: 初步异常检测
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from match_data import get_mask_label_counts
from src.utils.organ_mapping import LABEL_TO_ORGAN
from src.utils.file_io import load_npz_keys

logger = logging.getLogger(__name__)


def _outlier_worker(args: tuple) -> list:
    row_dict, volume_thresholds = args
    try:
        label_counts = get_mask_label_counts(row_dict['mask_path'])

        spacing = 3
        voxel_volume_ml = np.prod(spacing) / 1000

        outlier_records = []
        for label, count in label_counts.items():
            organ_name = LABEL_TO_ORGAN.get(int(label))
            if not organ_name:
                continue
            if organ_name not in volume_thresholds:
                continue

            volume_ml = count * voxel_volume_ml
            thresholds = volume_thresholds[organ_name]
            p01 = thresholds['p01']
            p99 = thresholds['p99']

            if volume_ml < p01:
                severity = 'high' if volume_ml < p01 * 0.5 else 'medium'
                outlier_records.append({
                    'accession_number': row_dict['accession_number'],
                    'series_number': row_dict['series_number'],
                    'mask_path': row_dict['mask_path'],
                    'organ_name': organ_name,
                    'organ_label': int(label),
                    'volume_ml': volume_ml,
                    'threshold_min_ml': p01,
                    'threshold_max_ml': p99,
                    'outlier_type': 'volume_too_small',
                    'severity': severity
                })
            elif volume_ml > p99:
                severity = 'high' if volume_ml > p99 * 2 else 'medium'
                outlier_records.append({
                    'accession_number': row_dict['accession_number'],
                    'series_number': row_dict['series_number'],
                    'mask_path': row_dict['mask_path'],
                    'organ_name': organ_name,
                    'organ_label': int(label),
                    'volume_ml': volume_ml,
                    'threshold_min_ml': p01,
                    'threshold_max_ml': p99,
                    'outlier_type': 'volume_too_large',
                    'severity': severity
                })

        return outlier_records
    except Exception as e:
        logger.warning(f"处理文件失败 {row_dict['mask_path']}: {e}")
        return []


def detect_outliers(file_index_path: str, organ_volumes_path: str,
                    output_path: str = None, sample_size: int = None,
                    num_workers: int = None, chunksize: int = 4) -> pd.DataFrame:
    """
    任务0.6: 初步异常检测
    
    Args:
        file_index_path: 文件索引表路径
        organ_volumes_path: 器官体积统计表路径
        output_path: 输出CSV路径
        sample_size: 测试模式样本数量
    
    Returns:
        pd.DataFrame: 异常检测结果表
    """
    logger.info("=" * 70)
    logger.info("任务0.6: 初步异常检测")
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
    
    # 加载体积统计表,提取阈值
    df_volumes = pd.read_csv(organ_volumes_path)
    volume_thresholds = {}
    
    for idx, row in df_volumes.iterrows():
        organ_name = row['organ_name']
        volume_thresholds[organ_name] = {
            'p01': row['volume_p01_ml'],
            'p99': row['volume_p99_ml']
        }
    
    logger.info(f"加载了 {len(volume_thresholds)} 个器官的体积阈值")
    
    # 检测异常
    outlier_records = []

    if num_workers is None:
        num_workers = max(1, min(4, mp.cpu_count() - 1))

    work_items = []
    for _, row in df_valid.iterrows():
        work_items.append((row.to_dict(), volume_thresholds))

    with mp.get_context("fork").Pool(processes=num_workers) as pool:
        for records in tqdm(pool.imap_unordered(_outlier_worker, work_items, chunksize=chunksize),
                            total=len(work_items), desc="检测异常"):
            if records:
                outlier_records.extend(records)
    
    df_result = pd.DataFrame(outlier_records)
    
    # 统计
    total_outlier_records = len(df_result)
    # 异常样本数 = 去重后的case数量
    total_outlier_cases = df_result['accession_number'].nunique() if not df_result.empty else 0
    outlier_rate = total_outlier_cases / len(df_valid) * 100 if len(df_valid) > 0 else 0
    
    logger.info("\n异常检测结果:")
    logger.info(f"  异常记录数: {total_outlier_records:,}")
    logger.info(f"  异常样本数: {total_outlier_cases:,}")
    logger.info(f"  异常率: {outlier_rate:.2f}%")
    
    if not df_result.empty:
        # 按异常类型统计
        type_counts = df_result['outlier_type'].value_counts()
        logger.info("\n  异常类型分布:")
        for outlier_type, count in type_counts.items():
            logger.info(f"    {outlier_type}: {count:,}")
        
        # 按严重度统计
        severity_counts = df_result['severity'].value_counts()
        logger.info("\n  严重度分布:")
        for severity, count in severity_counts.items():
            logger.info(f"    {severity}: {count:,}")
        
        # 最常见的异常器官
        organ_counts = df_result['organ_name'].value_counts().head(10)
        logger.info("\n  最常见的异常器官TOP10:")
        for organ, count in organ_counts.items():
            logger.info(f"    {organ}: {count:,}")
    
    # 保存结果
    if output_path:
        df_result.to_csv(output_path, index=False)
        logger.info(f"\n结果已保存: {output_path}")
    
    return df_result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='初步异常检测')
    parser.add_argument('--index', type=str, default='outputs/stage0_数据探索/file_index.csv',
                        help='文件索引表路径')
    parser.add_argument('--volumes', type=str, 
                        default='outputs/stage0_数据探索/stage0_organ_volumes.csv',
                        help='器官体积统计表路径')
    parser.add_argument('--output', type=str, 
                        default='outputs/stage0_数据探索/stage0_outliers.csv',
                        help='输出CSV路径')
    parser.add_argument('--sample', type=int, default=None, help='测试模式样本数量')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    detect_outliers(args.index, args.volumes, args.output, args.sample)
