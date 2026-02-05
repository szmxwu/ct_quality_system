"""
阶段0 - 任务0.5: 部位分布分析
"""
import os
import sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import logging
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from match_data import get_mask_labels
from src.utils.organ_mapping import infer_body_part_from_organs, LABEL_TO_ORGAN

logger = logging.getLogger(__name__)


def _body_part_worker(mask_path: str) -> dict:
    try:
        labels = get_mask_labels(mask_path)
        detected_organs = set()
        for label in labels:
            organ_name = LABEL_TO_ORGAN.get(int(label))
            if organ_name:
                detected_organs.add(organ_name)

        body_part = infer_body_part_from_organs(detected_organs)
        return {
            'body_part': body_part,
            'detected_organs': detected_organs
        }
    except Exception as e:
        logger.warning(f"处理文件失败 {mask_path}: {e}")
        return {'body_part': 'UNKNOWN', 'detected_organs': set()}


def analyze_body_part_distribution(file_index_path: str, 
                                   organ_occurrence_path: str = None,
                                   output_path: str = None,
                                   sample_size: int = None,
                                   num_workers: int = None,
                                   chunksize: int = 8) -> pd.DataFrame:
    """
    任务0.5: 部位分布分析
    
    Args:
        file_index_path: 文件索引表路径
        organ_occurrence_path: 器官出现频率表路径 (可选)
        output_path: 输出CSV路径
        sample_size: 测试模式样本数量
    
    Returns:
        pd.DataFrame: 部位分布统计表
    """
    logger.info("=" * 70)
    logger.info("任务0.5: 部位分布分析")
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
    
    # 统计各部位的样本数量和常见器官
    body_part_counts = defaultdict(int)
    body_part_organs = defaultdict(lambda: defaultdict(int))

    if num_workers is None:
        num_workers = max(1, min(4, mp.cpu_count() - 1))

    mask_paths = df_valid['mask_path'].tolist()
    with mp.get_context("fork").Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(_body_part_worker, mask_paths, chunksize=chunksize),
                           total=len(mask_paths), desc="分析部位分布"):
            body_part = result['body_part']
            detected_organs = result['detected_organs']
            body_part_counts[body_part] += 1
            for organ in detected_organs:
                body_part_organs[body_part][organ] += 1
    
    # 创建结果表
    records = []
    for body_part, count in sorted(body_part_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(df_valid) * 100 if len(df_valid) > 0 else 0
        
        # 获取该部位最常见的10个器官
        common_organs = sorted(body_part_organs[body_part].items(), 
                              key=lambda x: x[1], reverse=True)[:10]
        common_organs_str = ', '.join([f"{organ}({cnt})" for organ, cnt in common_organs])
        
        records.append({
            'body_part': body_part,
            'case_count': count,
            'percentage': percentage,
            'common_organs': common_organs_str
        })
    
    df_result = pd.DataFrame(records)
    
    # 打印结果
    logger.info("\n部位分布统计:")
    logger.info("-" * 70)
    for idx, row in df_result.iterrows():
        logger.info(f"  {row['body_part']:15s}: {row['case_count']:6,} ({row['percentage']:5.1f}%)")
    
    # 验证: 胸部样本应最多,四肢最少
    if 'CHEST' in body_part_counts and 'EXTREMITY' in body_part_counts:
        if body_part_counts['CHEST'] > body_part_counts['EXTREMITY']:
            logger.info("\n✓ 分布符合预期: 胸部样本多于四肢样本")
        else:
            logger.warning("\n⚠ 分布异常: 胸部样本少于四肢样本")
    
    # 保存结果
    if output_path:
        df_result.to_csv(output_path, index=False)
        logger.info(f"\n结果已保存: {output_path}")
    
    return df_result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='部位分布分析')
    parser.add_argument('--index', type=str, default='outputs/stage0_数据探索/file_index.csv',
                        help='文件索引表路径')
    parser.add_argument('--organ-occurrence', type=str, default=None,
                        help='器官出现频率表路径(可选)')
    parser.add_argument('--output', type=str, 
                        default='outputs/stage0_数据探索/stage0_body_part_distribution.csv',
                        help='输出CSV路径')
    parser.add_argument('--sample', type=int, default=None, help='测试模式样本数量')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    analyze_body_part_distribution(
        args.index, 
        args.organ_occurrence, 
        args.output, 
        args.sample
    )
