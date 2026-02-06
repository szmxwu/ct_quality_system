"""
阶段1: 严重错误过滤
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from match_data import get_mask_labels, get_mask_label_counts, get_npz_masks
from src.utils.organ_mapping import infer_body_part_from_organs, LABEL_TO_ORGAN
from src.utils.geometry import compute_centroid
from src.utils.file_io import load_npz_keys

logger = logging.getLogger(__name__)


class HardFilter:
    """硬过滤器"""
    
    def __init__(self, rules_path: str, volume_stats_path: str):
        """
        初始化硬过滤器
        
        Args:
            rules_path: 规则配置文件路径
            volume_stats_path: 器官体积统计表路径
        """
        # 加载规则
        with open(rules_path, 'r') as f:
            self.rules = json.load(f)
        
        # 加载体积阈值
        df_volumes = pd.read_csv(volume_stats_path)
        self.volume_thresholds = {}
        for idx, row in df_volumes.iterrows():
            self.volume_thresholds[row['organ_name']] = {
                'min': row['volume_p01_ml'],
                'max': row['volume_p99_ml']
            }
        
        # logger.info(f"硬过滤器初始化完成")
        # logger.info(f"  加载了 {len(self.volume_thresholds)} 个器官的体积阈值")
    
    def check_critical_organs(self, detected_organs: set, body_part: str) -> list:
        """
        检查关键器官缺失
        
        Args:
            detected_organs: 检测到的器官集合
            body_part: 推断的扫描部位
        
        Returns:
            list: 错误列表
        """
        errors = []
        
        if body_part not in self.rules['critical_organ_rules']:
            return errors
        
        rule = self.rules['critical_organ_rules'][body_part]
        
        # 检查required
        if 'required' in rule:
            required = set(rule['required'])
            missing = required - detected_organs
            if missing:
                errors.append(f"missing_critical:{','.join(missing)}")
        
        # 检查alternative_required
        if 'alternative_required' in rule:
            alternatives = rule['alternative_required']
            satisfied = False
            for alt_set in alternatives:
                if set(alt_set).issubset(detected_organs):
                    satisfied = True
                    break
            if not satisfied:
                errors.append(f"missing_critical_alternative:{body_part}")
        
        return errors
    
    def check_cooccurrence(self, detected_organs: set) -> list:
        """
        检查共现规则违反
        
        Args:
            detected_organs: 检测到的器官集合
        
        Returns:
            list: 错误列表
        """
        errors = []
        
        for rule in self.rules['cooccurrence_rules']:
            if_exists = rule['if_exists']
            must_exist = rule['must_exist']
            reason = rule['reason']
            
            if if_exists in detected_organs:
                for must_organ in must_exist:
                    if must_organ not in detected_organs:
                        errors.append(f"cooccurrence_violated:{reason}")
        
        return errors
    
    def check_volume_counts(self, label_counts: dict, spacing: np.ndarray) -> list:
        """
        检查体积异常
        
        Args:
            masks: 器官mask字典
            spacing: spacing数组
        
        Returns:
            list: 错误列表
        """
        errors = []
        
        voxel_volume_ml = np.prod(spacing) / 1000
        
        for label, count in label_counts.items():
            organ_name = LABEL_TO_ORGAN.get(int(label))
            if not organ_name or organ_name not in self.volume_thresholds:
                continue
            
            volume_ml = count * voxel_volume_ml
            thresholds = self.volume_thresholds[organ_name]
            min_vol = thresholds['min']
            max_vol = thresholds['max']
            
            if volume_ml < min_vol:
                errors.append(f"volume_too_small:{organ_name}={volume_ml:.1f}<{min_vol:.1f}")
            elif volume_ml > max_vol:
                errors.append(f"volume_too_large:{organ_name}={volume_ml:.1f}>{max_vol:.1f}")
        
        return errors
    
    def check_organ_count(self, detected_organs: set, body_part: str) -> list:
        """
        检查器官数量
        
        Args:
            detected_organs: 检测到的器官集合
            body_part: 推断的扫描部位
        
        Returns:
            list: 错误列表
        """
        errors = []
        
        min_count = self.rules['minimum_organ_count'].get(body_part, 2)
        if len(detected_organs) < min_count:
            errors.append(f"too_few_organs:{len(detected_organs)}<{min_count}")
        
        return errors
    
    def check_spatial_constraints(self, masks: dict) -> list:
        """
        检查空间约束
        
        Args:
            masks: 器官mask字典
        
        Returns:
            list: 错误列表
        """
        errors = []
        
        for constraint in self.rules['spatial_constraints']:
            organ1, organ2 = constraint['organ_pair']
            check_method = constraint['check_method']
            
            if organ1 not in masks or organ2 not in masks:
                continue
            
            centroid1 = compute_centroid(masks[organ1])
            centroid2 = compute_centroid(masks[organ2])
            
            if centroid1 is None or centroid2 is None:
                continue
            
            if check_method == 'centroid_x_comparison':
                # 脾脏应在肝脏左侧 (x坐标更小)
                if organ1 == 'liver' and organ2 == 'spleen':
                    if centroid2[2] > centroid1[2]:  # [z,y,x]
                        errors.append(f"spatial_violation:{constraint['rule']}")
            
            elif check_method == 'centroid_z_distance':
                # 双肾应大致在同一水平
                z_distance = abs(centroid1[0] - centroid2[0])
                max_distance = constraint.get('max_distance_slices', 20)
                if z_distance > max_distance:
                    errors.append(f"spatial_violation:{constraint['rule']}")
        
        return errors
    
    def filter_case(self, mask_path: str, ct_path: str) -> dict:
        """
        过滤单个case
        
        Args:
            mask_path: mask文件路径
            ct_path: CT文件路径
        
        Returns:
            dict: 过滤结果
        """
        errors = []
        
        try:
            # 读取标签集合 (避免构建全部mask)
            labels = get_mask_labels(mask_path)
            detected_organs = set()
            for label in labels:
                organ_name = LABEL_TO_ORGAN.get(int(label))
                if organ_name:
                    detected_organs.add(organ_name)
            
            # 推断扫描部位
            body_part = infer_body_part_from_organs(detected_organs)
            
            # 执行各项检查
            errors.extend(self.check_critical_organs(detected_organs, body_part))
            errors.extend(self.check_cooccurrence(detected_organs))
            errors.extend(self.check_organ_count(detected_organs, body_part))
            
            # 体积检查需要CT数据
            spacing = 3
            label_counts = get_mask_label_counts(mask_path)
            errors.extend(self.check_volume_counts(label_counts, spacing))

            # 空间约束需要mask,按需构建
            if any(self.rules.get('spatial_constraints', [])):
                masks = get_npz_masks(mask_path)
                errors.extend(self.check_spatial_constraints(masks))
            
            return {
                'detected_organ_count': len(detected_organs),
                'inferred_body_part': body_part,
                'passed': len(errors) == 0,
                'error_count': len(errors),
                'errors': ';'.join(errors) if errors else ''
            }
        
        except Exception as e:
            logger.warning(f"处理文件失败 {mask_path}: {e}")
            return {
                'detected_organ_count': 0,
                'inferred_body_part': 'UNKNOWN',
                'passed': False,
                'error_count': 1,
                'errors': f'processing_error:{str(e)}'
            }


def _hard_filter_worker(args: tuple) -> dict:
    row_dict, rules_path, volume_stats_path = args
    filter_obj = HardFilter(rules_path, volume_stats_path)
    result = filter_obj.filter_case(row_dict['mask_path'], row_dict['ct_path'])
    return {
        'accession_number': row_dict['accession_number'],
        'series_number': row_dict['series_number'],
        'batch': row_dict['batch'],
        'mask_path': row_dict['mask_path'],
        'ct_path': row_dict['ct_path'],
        'inferred_body_part': result['inferred_body_part'],
        'detected_organ_count': result['detected_organ_count'],
        'passed_hard_filter': result['passed'],
        'error_count': result['error_count'],
        'errors': result['errors']
    }


def run_hard_filter(file_index_path: str, rules_path: str, volume_stats_path: str,
                    output_path: str = None, reject_log_path: str = None,
                    sample_size: int = None, num_workers: int = None,
                    chunksize: int = 2) -> pd.DataFrame:
    """
    执行硬过滤
    
    Args:
        file_index_path: 文件索引表路径
        rules_path: 规则配置文件路径
        volume_stats_path: 器官体积统计表路径
        output_path: 过滤结果输出路径
        reject_log_path: 拒绝日志输出路径
        sample_size: 测试模式样本数量
    
    Returns:
        pd.DataFrame: 过滤结果表
    """
    logger.info("=" * 70)
    logger.info("阶段1: 严重错误过滤")
    logger.info("=" * 70)
    
    # 加载索引表
    df_index = pd.read_csv(file_index_path)
    df_valid = df_index[df_index['ct_exists'] == True]
    
    logger.info(f"有效样本数: {len(df_valid):,}")
    
    if sample_size:
        df_valid = df_valid.head(sample_size)
        logger.info(f"测试模式: 处理 {len(df_valid)} 个样本")
    
    # 处理每个case
    results = []
    reject_records = []

    if num_workers is None:
        num_workers = max(1, min(4, mp.cpu_count() - 1))

    work_items = []
    for _, row in df_valid.iterrows():
        work_items.append((row.to_dict(), rules_path, volume_stats_path))

    with mp.get_context("fork").Pool(processes=num_workers) as pool:
        for record in tqdm(pool.imap_unordered(_hard_filter_worker, work_items, chunksize=chunksize),
                           total=len(work_items), desc="执行硬过滤"):
            results.append(record)

            # 记录拒绝详情
            if not record['passed_hard_filter']:
                for error in record['errors'].split(';'):
                    if ':' in error:
                        error_type, error_detail = error.split(':', 1)
                    else:
                        error_type = error
                        error_detail = ''

                    # 判断严重度
                    if 'missing_critical' in error_type or 'cooccurrence_violated' in error_type:
                        severity = 'critical'
                    elif 'volume' in error_type:
                        severity = 'major'
                    else:
                        severity = 'minor'

                    reject_records.append({
                        'accession_number': record['accession_number'],
                        'series_number': record['series_number'],
                        'error_type': error_type,
                        'error_detail': error_detail,
                        'severity': severity
                    })
    
    df_result = pd.DataFrame(results)
    
    # 统计
    total = len(df_result)
    passed = df_result['passed_hard_filter'].sum()
    rejected = total - passed
    reject_rate = rejected / total * 100 if total > 0 else 0
    
    logger.info("\n" + "=" * 70)
    logger.info("过滤结果总览:")
    logger.info(f"  处理总数: {total:,}")
    logger.info(f"  通过过滤: {passed:,} ({100-reject_rate:.1f}%)")
    logger.info(f"  被拒绝: {rejected:,} ({reject_rate:.1f}%)")
    
    # 验证拒绝率
    if 3 <= reject_rate <= 8:
        logger.info(f"✓ 拒绝率在预期范围内 (3-8%)")
    else:
        logger.warning(f"⚠ 拒绝率超出预期范围 (3-8%): {reject_rate:.1f}%")
    
    # 拒绝原因分布
    if reject_records:
        df_reject = pd.DataFrame(reject_records)
        error_type_counts = df_reject['error_type'].value_counts()
        logger.info("\n拒绝原因分布:")
        for error_type, count in error_type_counts.head(10).items():
            percentage = count / rejected * 100 if rejected > 0 else 0
            logger.info(f"  {error_type}: {count:,} ({percentage:.1f}%)")
        
        # 各部位拒绝率
        body_part_reject = df_result.groupby('inferred_body_part').agg({
            'passed_hard_filter': ['count', 'sum']
        }).reset_index()
        body_part_reject.columns = ['body_part', 'total', 'passed']
        body_part_reject['rejected'] = body_part_reject['total'] - body_part_reject['passed']
        body_part_reject['reject_rate'] = body_part_reject['rejected'] / body_part_reject['total'] * 100
        body_part_reject = body_part_reject.sort_values('reject_rate', ascending=False)
        
        logger.info("\n各部位拒绝率:")
        for idx, row in body_part_reject.iterrows():
            logger.info(f"  {row['body_part']:15s}: {row['reject_rate']:5.1f}% "
                       f"({row['rejected']}/{row['total']})")
    
    # 保存结果
    if output_path:
        df_result.to_csv(output_path, index=False)
        logger.info(f"\n过滤结果已保存: {output_path}")
    
    if reject_log_path and reject_records:
        df_reject = pd.DataFrame(reject_records)
        df_reject.to_csv(reject_log_path, index=False)
        logger.info(f"拒绝日志已保存: {reject_log_path}")
    
    return df_result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='严重错误过滤')
    parser.add_argument('--index', type=str, default='outputs/stage0_数据探索/file_index.csv',
                        help='文件索引表路径')
    parser.add_argument('--rules', type=str, default='config/hard_filter_rules.json',
                        help='规则配置文件路径')
    parser.add_argument('--volumes', type=str, 
                        default='outputs/stage0_数据探索/stage0_organ_volumes.csv',
                        help='器官体积统计表路径')
    parser.add_argument('--output', type=str, 
                        default='outputs/stage1_硬过滤/stage1_filter_results.csv',
                        help='过滤结果输出路径')
    parser.add_argument('--reject-log', type=str,
                        default='outputs/stage1_硬过滤/stage1_reject_log.csv',
                        help='拒绝日志输出路径')
    parser.add_argument('--sample', type=int, default=None, help='测试模式样本数量')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    run_hard_filter(
        args.index, 
        args.rules, 
        args.volumes, 
        args.output, 
        args.reject_log, 
        args.sample
    )
