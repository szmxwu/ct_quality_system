"""
阶段2: 质量评分系统
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure
from tqdm import tqdm
import logging
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from match_data import get_npz_masks
from src.utils.organ_mapping import LABEL_TO_ORGAN, classify_organ_type
from src.utils.geometry import extract_boundary, compute_gradient_magnitude
from src.utils.file_io import load_npz_file

logger = logging.getLogger(__name__)


# HU值期望范围
HU_RANGES = {
    # 实质器官
    'liver': (-10, 70),
    'spleen': (40, 60),
    'kidney_left': (30, 50),
    'kidney_right': (30, 50),
    'pancreas': (30, 50),
    'heart': (30, 70),
    
    # 肺组织
    'lung_upper_lobe_left': (-900, -500),
    'lung_lower_lobe_left': (-900, -500),
    'lung_upper_lobe_right': (-900, -500),
    'lung_middle_lobe_right': (-900, -500),
    'lung_lower_lobe_right': (-900, -500),
    
    # 血管(平扫)
    'aorta': (30, 80),
    'portal_vein_and_splenic_vein': (30, 80),
    'inferior_vena_cava': (30, 80),
    
    # 脑组织
    'brain': (20, 45),
    
    # 含气器官
    'stomach': (-100, 50),
    'small_bowel': (-100, 50),
    'duodenum': (-100, 50),
    'colon': (-100, 50),
}

# 器官类型期望梯度比
EXPECTED_GRADIENT_RATIOS = {
    'soft_tissue': 1.5,
    'vessel': 2.0,
    'bone': 3.0,
    'lung': 2.5,
    'air': 4.0
}

# 实质器官列表(用于紧凑度计算)
SOLID_ORGANS = [
    'liver', 'spleen', 'kidney_left', 'kidney_right', 'heart',
    'pancreas', 'stomach', 'urinary_bladder', 'prostate', 'gallbladder',
    'brain'
]


def compute_boundary_score(ct_image: np.ndarray, mask: np.ndarray, 
                           organ_name: str) -> float:
    """
    计算边界质量分数
    
    Args:
        ct_image: CT图像数组
        mask: 器官mask
        organ_name: 器官名称
    
    Returns:
        float: 边界分数 (0-1)
    """
    try:
        # 提取边界voxels
        eroded = ndimage.binary_erosion(mask, iterations=1)
        boundary = (mask & ~eroded).astype(bool)
        
        if not boundary.any() or not eroded.any():
            return 1.0
        
        # 计算CT图像梯度
        gradient_magnitude = compute_gradient_magnitude(ct_image)
        
        # 计算边界和内部的梯度
        boundary_gradient = gradient_magnitude[boundary]
        interior_gradient = gradient_magnitude[eroded]
        
        if len(boundary_gradient) == 0 or len(interior_gradient) == 0:
            return 1.0
        
        median_boundary = np.median(boundary_gradient)
        median_interior = np.median(interior_gradient)
        
        gradient_ratio = median_boundary / (median_interior + 1e-6)
        
        # 根据器官类型选择期望值
        organ_type = classify_organ_type(organ_name)
        expected = EXPECTED_GRADIENT_RATIOS.get(organ_type, 1.5)
        
        # 归一化到0-1
        boundary_score = min(gradient_ratio / expected, 1.0)
        
        return boundary_score
    
    except Exception as e:
        logger.warning(f"计算边界分数失败 {organ_name}: {e}")
        return 1.0


def compute_connectivity_score(mask: np.ndarray) -> float:
    """
    计算连通性分数 - 使用更快的ndimage.label
    
    Args:
        mask: 器官mask
    
    Returns:
        float: 连通性分数 (0-1)
    """
    try:
        # 使用scipy.ndimage.label比skimage.measure.label快2-3倍
        labels, num_components = ndimage.label(mask)
        
        if num_components == 0:
            return 0.0
        
        if num_components == 1:
            return 1.0
        
        # 使用bincount快速计算各连通分量大小
        component_sizes = np.bincount(labels.ravel())[1:]  # 跳过背景0
        largest_ratio = component_sizes.max() / component_sizes.sum()
        return largest_ratio
    
    except Exception as e:
        logger.warning(f"计算连通性分数失败: {e}")
        return 1.0


def compute_compactness_score(mask: np.ndarray, organ_name: str) -> float:
    """
    计算紧凑度分数
    
    Args:
        mask: 器官mask
        organ_name: 器官名称
    
    Returns:
        float: 紧凑度分数 (0-1)
    """
    # 管状/条状结构不考核紧凑度
    if organ_name not in SOLID_ORGANS:
        return 1.0
    
    try:
        volume = np.sum(mask)
        if volume == 0:
            return 0.0
        
        # 计算表面voxel数量
        eroded = ndimage.binary_erosion(mask)
        surface_voxels = np.sum(mask) - np.sum(eroded)
        
        # 理想球体的表面积/体积比
        ideal_ratio = 4.836 * (volume ** (2/3)) / volume
        actual_ratio = surface_voxels / volume
        
        compactness_score = min(ideal_ratio / (actual_ratio + 1e-6), 1.0)
        return compactness_score
    
    except Exception as e:
        logger.warning(f"计算紧凑度分数失败 {organ_name}: {e}")
        return 1.0


def compute_morphology_score(mask: np.ndarray, organ_name: str) -> dict:
    """
    计算形态学质量分数
    
    Args:
        mask: 器官mask
        organ_name: 器官名称
    
    Returns:
        dict: 包含各项子分数的字典
    """
    connectivity_score = compute_connectivity_score(mask)
    compactness_score = compute_compactness_score(mask, organ_name)
    morphology_score = (connectivity_score + compactness_score) / 2
    
    return {
        'connectivity_score': connectivity_score,
        'compactness_score': compactness_score,
        'morphology_score': morphology_score
    }


def compute_hu_score(ct_image: np.ndarray, mask: np.ndarray, 
                     organ_name: str) -> float:
    """
    计算HU值分布分数
    
    Args:
        ct_image: CT图像数组
        mask: 器官mask
        organ_name: 器官名称
    
    Returns:
        float: HU分数 (0-1)
    """
    # 确定HU范围
    if 'vertebrae_' in organ_name or 'rib_' in organ_name:
        hu_range = (200, 600)
    elif 'muscle' in organ_name or 'gluteus' in organ_name or 'iliopsoas' in organ_name:
        hu_range = (10, 70)
    elif organ_name in HU_RANGES:
        hu_range = HU_RANGES[organ_name]
    else:
        # 未知器官不评分
        return 1.0
    
    try:
        hu_values = ct_image[mask > 0]
        if len(hu_values) == 0:
            return 1.0
        
        median_hu = np.median(hu_values)
        min_hu, max_hu = hu_range
        
        if min_hu <= median_hu <= max_hu:
            # 中位数在合理范围内,进一步检查异常值比例
            outlier_low = hu_values < (min_hu - 100)
            outlier_high = hu_values > (max_hu + 100)
            outlier_ratio = (np.sum(outlier_low) + np.sum(outlier_high)) / len(hu_values)
            
            hu_score = max(1.0 - outlier_ratio, 0.0)
        else:
            # 中位数不在范围内,给低分
            if median_hu < min_hu:
                deviation = (min_hu - median_hu) / abs(min_hu + 1)
            else:
                deviation = (median_hu - max_hu) / abs(max_hu + 1)
            
            hu_score = max(1.0 - deviation, 0.0)
        
        return hu_score
    
    except Exception as e:
        logger.warning(f"计算HU分数失败 {organ_name}: {e}")
        return 1.0


def compute_overall_score(boundary_score: float, morphology_score: float, 
                          hu_score: float) -> float:
    """
    计算综合分数
    
    Args:
        boundary_score: 边界分数
        morphology_score: 形态学分数
        hu_score: HU分数
    
    Returns:
        float: 综合分数 (0-1)
    """
    return 0.5 * boundary_score + 0.3 * morphology_score + 0.2 * hu_score


def score_case(mask_path: str, ct_path: str) -> list:
    """
    为单个case的所有器官打分 - 优化版：缓存CT梯度
    
    Args:
        mask_path: mask文件路径
        ct_path: CT文件路径
    
    Returns:
        list: 器官评分记录列表
    """
    records = []
    
    try:
        # 加载mask
        masks = get_npz_masks(mask_path)
        
        # 加载CT
        ct_data = load_npz_file(ct_path)
        if ct_data is None or 'image' not in ct_data:
            return records
        
        ct_image = ct_data['image']
        
        # 预计算CT梯度（所有器官共用）
        gradient_magnitude = compute_gradient_magnitude(ct_image)
        
        # 为每个器官打分
        for organ_name, mask in masks.items():
            try:
                organ_label = LABEL_TO_ORGAN.get(organ_name, 0)
                
                # 计算各项分数 - 使用缓存的梯度
                boundary_score = compute_boundary_score_with_gradient(
                    ct_image, mask, organ_name, gradient_magnitude
                )
                morph_scores = compute_morphology_score(mask, organ_name)
                hu_score = compute_hu_score(ct_image, mask, organ_name)
                overall_score = compute_overall_score(
                    boundary_score, 
                    morph_scores['morphology_score'], 
                    hu_score
                )
                
                records.append({
                    'organ_name': organ_name,
                    'organ_label': organ_label,
                    'boundary_score': boundary_score,
                    'connectivity_score': morph_scores['connectivity_score'],
                    'compactness_score': morph_scores['compactness_score'],
                    'morphology_score': morph_scores['morphology_score'],
                    'hu_score': hu_score,
                    'overall_score': overall_score
                })
            
            except Exception as e:
                logger.warning(f"评分失败 {organ_name}: {e}")
                continue
        
        # 释放内存
        del ct_data, gradient_magnitude
    
    except Exception as e:
        logger.warning(f"处理case失败 {mask_path}: {e}")
    
    return records


def compute_boundary_score_with_gradient(ct_image: np.ndarray, mask: np.ndarray, 
                                         organ_name: str, gradient_magnitude: np.ndarray) -> float:
    """
    计算边界质量分数 - 使用预计算的梯度
    
    Args:
        ct_image: CT图像数组
        mask: 器官mask
        organ_name: 器官名称
        gradient_magnitude: 预计算的梯度幅值
    
    Returns:
        float: 边界分数 (0-1)
    """
    try:
        # 提取边界voxels
        eroded = ndimage.binary_erosion(mask, iterations=1)
        boundary = (mask & ~eroded).astype(bool)
        
        if not boundary.any() or not eroded.any():
            return 1.0
        
        # 使用预计算的梯度
        boundary_gradient = gradient_magnitude[boundary]
        interior_gradient = gradient_magnitude[eroded]
        
        if len(boundary_gradient) == 0 or len(interior_gradient) == 0:
            return 1.0
        
        median_boundary = np.median(boundary_gradient)
        median_interior = np.median(interior_gradient)
        
        gradient_ratio = median_boundary / (median_interior + 1e-6)
        
        # 根据器官类型选择期望值
        organ_type = classify_organ_type(organ_name)
        expected = EXPECTED_GRADIENT_RATIOS.get(organ_type, 1.5)
        
        # 归一化到0-1
        boundary_score = min(gradient_ratio / expected, 1.0)
        
        return boundary_score
    
    except Exception as e:
        logger.warning(f"计算边界分数失败 {organ_name}: {e}")
        return 1.0


def _score_case_worker(args: tuple) -> tuple:
    row_dict = args
    organ_scores = score_case(row_dict['mask_path'], row_dict['ct_path'])
    return row_dict, organ_scores


def run_quality_scoring(filter_results_path: str, output_dir: str = None,
                        sample_size: int = None, checkpoint_interval: int = 10000,
                        num_workers: int = None, chunksize: int = 1) -> tuple:
    """
    执行质量评分
    
    Args:
        filter_results_path: 过滤结果表路径
        output_dir: 输出目录
        sample_size: 测试模式样本数量
        checkpoint_interval: checkpoint保存间隔
    
    Returns:
        tuple: (器官评分表, case汇总表)
    """
    logger.info("=" * 70)
    logger.info("阶段2: 质量评分系统")
    logger.info("=" * 70)
    
    # 加载过滤结果
    df_filter = pd.read_csv(filter_results_path)
    
    # 只处理通过硬过滤的样本
    df_passed = df_filter[df_filter['passed_hard_filter'] == True]
    
    logger.info(f"通过硬过滤的样本数: {len(df_passed):,}")
    
    if sample_size:
        df_passed = df_passed.head(sample_size)
        logger.info(f"测试模式: 处理 {len(df_passed)} 个样本")
    
    # 处理每个case
    organ_records = []
    case_records = []

    if num_workers is None:
        num_workers = max(1, min(4, mp.cpu_count() - 1))

    work_items = [row.to_dict() for _, row in df_passed.iterrows()]

    with mp.get_context("fork").Pool(processes=num_workers) as pool:
        for idx, (row_dict, organ_scores) in enumerate(
            tqdm(pool.imap_unordered(_score_case_worker, work_items, chunksize=chunksize),
                 total=len(work_items), desc="质量评分")
        ):
            try:
                if not organ_scores:
                    continue

                # 添加case信息
                for record in organ_scores:
                    record['accession_number'] = row_dict['accession_number']
                    record['series_number'] = row_dict['series_number']
                    organ_records.append(record)

                # 计算case级别汇总
                avg_boundary = np.mean([r['boundary_score'] for r in organ_scores])
                avg_morphology = np.mean([r['morphology_score'] for r in organ_scores])
                avg_hu = np.mean([r['hu_score'] for r in organ_scores])
                case_overall = np.mean([r['overall_score'] for r in organ_scores])

                # 质量分级
                if case_overall >= 0.8:
                    quality_tier = 'HIGH'
                elif case_overall >= 0.6:
                    quality_tier = 'MEDIUM'
                else:
                    quality_tier = 'LOW'

                case_records.append({
                    'accession_number': row_dict['accession_number'],
                    'series_number': row_dict['series_number'],
                    'num_organs': len(organ_scores),
                    'avg_boundary_score': avg_boundary,
                    'avg_morphology_score': avg_morphology,
                    'avg_hu_score': avg_hu,
                    'case_overall_score': case_overall,
                    'quality_tier': quality_tier
                })

                # 保存checkpoint
                if checkpoint_interval and (idx + 1) % checkpoint_interval == 0:
                    checkpoint_path = f"{output_dir}/stage2_checkpoint_{idx+1}.csv"
                    pd.DataFrame(organ_records).to_csv(checkpoint_path, index=False)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

            except Exception as e:
                logger.warning(f"处理case失败 {row_dict.get('accession_number')}: {e}")
                continue
    
    df_organ_scores = pd.DataFrame(organ_records)
    df_case_summary = pd.DataFrame(case_records)
    
    # 统计
    logger.info("\n" + "=" * 70)
    logger.info("评分结果总览:")
    logger.info(f"  评分总样本数: {len(df_case_summary):,}")
    logger.info(f"  评分总器官数: {len(df_organ_scores):,}")
    
    if not df_case_summary.empty:
        avg_organs = df_case_summary['num_organs'].mean()
        logger.info(f"  平均每case器官数: {avg_organs:.1f}")
        
        # 质量分级分布
        tier_counts = df_case_summary['quality_tier'].value_counts()
        logger.info("\n  质量分级分布 (按case):")
        for tier in ['HIGH', 'MEDIUM', 'LOW']:
            if tier in tier_counts:
                count = tier_counts[tier]
                percentage = count / len(df_case_summary) * 100
                logger.info(f"    {tier}: {count:,} ({percentage:.1f}%)")
        
        # 各评分维度平均分
        logger.info("\n  各评分维度平均分:")
        logger.info(f"    边界质量: {df_case_summary['avg_boundary_score'].mean():.3f}")
        logger.info(f"    形态质量: {df_case_summary['avg_morphology_score'].mean():.3f}")
        logger.info(f"    HU值分布: {df_case_summary['avg_hu_score'].mean():.3f}")
        logger.info(f"    综合分数: {df_case_summary['case_overall_score'].mean():.3f}")
    
    # 保存结果
    if output_dir:
        organ_output = os.path.join(output_dir, 'stage2_organ_quality_scores.csv')
        case_output = os.path.join(output_dir, 'stage2_case_summary.csv')
        
        df_organ_scores.to_csv(organ_output, index=False)
        df_case_summary.to_csv(case_output, index=False)
        
        logger.info(f"\n器官评分已保存: {organ_output}")
        logger.info(f"case汇总已保存: {case_output}")
    
    return df_organ_scores, df_case_summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='质量评分系统')
    parser.add_argument('--filter-results', type=str, 
                        default='outputs/stage1_硬过滤/stage1_filter_results.csv',
                        help='过滤结果表路径')
    parser.add_argument('--output-dir', type=str, 
                        default='outputs/stage2_质量评分',
                        help='输出目录')
    parser.add_argument('--sample', type=int, default=None, help='测试模式样本数量')
    parser.add_argument('--checkpoint-interval', type=int, default=10000,
                        help='checkpoint保存间隔')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    run_quality_scoring(
        args.filter_results, 
        args.output_dir, 
        args.sample, 
        args.checkpoint_interval
    )
