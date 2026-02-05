"""
阶段0 - 任务0.1: 构建文件索引表
"""
import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.file_io import load_leaf_index, find_matching_ct

logger = logging.getLogger(__name__)


def build_file_index(leaf_index_path: str, mask_base_dir: str = '/media/wmx/KINGIDISK/shenzhen_mask', 
                     ct_base_dir: str = '/media/wmx/KINGIDISK/', output_path: str = None,
                     sample_size: int = None) -> pd.DataFrame:
    """
    构建文件索引表
    
    Args:
        leaf_index_path: leaf_index.json路径
        mask_base_dir: mask基础目录
        ct_base_dir: CT基础目录
        output_path: 输出CSV路径
        sample_size: 测试模式下的样本数量,None表示处理全部
    
    Returns:
        pd.DataFrame: 文件索引表
    """
    logger.info("=" * 70)
    logger.info("任务0.1: 构建文件索引表")
    logger.info("=" * 70)
    
    # 加载leaf_index
    logger.info(f"加载索引文件: {leaf_index_path}")
    leaf_index = load_leaf_index(leaf_index_path)
    logger.info(f"索引中包含 {len(leaf_index):,} 个AccessionNumber")
    
    # 收集所有mask文件
    logger.info("扫描mask文件...")
    records = []
    
    for accession_number, ct_dir in tqdm(leaf_index.items(), desc="扫描mask文件"):
        # 从CT路径推断mask路径
        # CT路径: /media/wmx/KINGIDISK\2410\0000109279\M24101403890
        # Mask路径: /media/wmx/KINGIDISK/shenzhen_mask/2410/0000109279/M24101403890
        
        parts = ct_dir.replace('/', '\\').split('\\')
        if len(parts) >= 3:
            batch = parts[-3]  # 2410
            patient_id = parts[-2]  # 0000109279
            
            mask_dir = os.path.join(mask_base_dir, batch, patient_id, accession_number)
            
            if os.path.exists(mask_dir):
                # 查找该目录下的所有.npz文件
                npz_files = list(Path(mask_dir).glob('*.npz'))
                for npz_file in npz_files:
                    series_number = npz_file.stem
                    mask_path = str(npz_file)
                    
                    # 查找对应的CT文件
                    ct_path = find_matching_ct({
                        'batch': batch,
                        'patient_id': patient_id,
                        'accession_number': accession_number,
                        'series_number': series_number
                    }, ct_base_dir)
                    
                    records.append({
                        'batch': batch,
                        'patient_id': patient_id,
                        'accession_number': accession_number,
                        'series_number': series_number,
                        'mask_path': mask_path,
                        'ct_path': ct_path,
                        'mask_exists': True,
                        'ct_exists': ct_path is not None
                    })
        
        # 测试模式:限制样本数量
        if sample_size and len(records) >= sample_size:
            logger.info(f"测试模式: 已收集 {sample_size} 个样本")
            break
    
    # 创建DataFrame
    df = pd.DataFrame(records)
    
    # 统计信息
    total_files = len(df)
    matched_files = df['ct_exists'].sum()
    match_rate = matched_files / total_files * 100 if total_files > 0 else 0
    
    logger.info("\n" + "=" * 70)
    logger.info("文件索引统计:")
    logger.info(f"  总文件数: {total_files:,}")
    logger.info(f"  Mask-CT配对成功: {matched_files:,} ({match_rate:.2f}%)")
    
    # 各批次分布
    batch_dist = df['batch'].value_counts().sort_index()
    logger.info("\n  各批次文件分布:")
    for batch, count in batch_dist.items():
        logger.info(f"    {batch}: {count:,}")
    
    # 检查匹配率
    if match_rate < 80:
        logger.error(f"警告: 匹配率低于80% ({match_rate:.2f}%),请检查数据路径!")
    else:
        logger.info(f"✓ 匹配率正常: {match_rate:.2f}%")
    
    # 保存结果
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"\n索引表已保存: {output_path}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='构建文件索引表')
    parser.add_argument('--leaf-index', type=str, default='leaf_index.json',
                        help='leaf_index.json路径')
    parser.add_argument('--mask-base', type=str, default='/media/wmx/KINGIDISK/shenzhen_mask',
                        help='mask基础目录')
    parser.add_argument('--ct-base', type=str, default='/media/wmx/KINGIDISK/',
                        help='CT基础目录')
    parser.add_argument('--output', type=str, default='outputs/stage0_数据探索/file_index.csv',
                        help='输出CSV路径')
    parser.add_argument('--sample', type=int, default=None,
                        help='测试模式样本数量')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    build_file_index(
        leaf_index_path=args.leaf_index,
        mask_base_dir=args.mask_base,
        ct_base_dir=args.ct_base,
        output_path=args.output,
        sample_size=args.sample
    )
