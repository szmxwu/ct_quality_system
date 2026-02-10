"""
CT分割质量管理系统 - 主流程脚本

阶段0-2完整pipeline
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from src.stage0 import (
    build_file_index,
    compute_organ_occurrence,
    compute_organ_volumes,
    compute_organ_cooccurrence,
    analyze_body_part_distribution,
    detect_outliers
)
from src.stage1 import run_hard_filter
from src.stage2 import run_quality_scoring
from src.utils.file_io import ensure_dir

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_stage0(leaf_index_path: str, mask_base_dir: str = '/media/wmx/KINGIDISK/shenzhen_mask',
               ct_base_dir: str = '/media/wmx/KINGIDISK', output_dir: str = 'outputs/stage0_数据探索',
               sample_size: int = None, num_workers: int = None):
    """
    运行阶段0: 数据探索与基线建立
    
    Args:
        leaf_index_path: leaf_index.json路径
        mask_base_dir: mask基础目录
        ct_base_dir: CT基础目录
        output_dir: 输出目录
        sample_size: 测试模式样本数量
    """
    logger.info("\n" + "=" * 70)
    logger.info("阶段0: 数据探索与基线建立")
    logger.info("=" * 70)
    
    ensure_dir(output_dir)
    
    # 任务0.1: 构建文件索引表
    file_index_path = os.path.join(output_dir, 'file_index.csv')
    if not os.path.exists(file_index_path):
        build_file_index(
            leaf_index_path=leaf_index_path,
            mask_base_dir=mask_base_dir,
            ct_base_dir=ct_base_dir,
            output_path=file_index_path,
            sample_size=sample_size
        )
    else:
        logger.info(f"文件索引表已存在,跳过: {file_index_path}")
    
    # 任务0.2: 器官出现频率统计
    occurrence_path = os.path.join(output_dir, 'stage0_organ_occurrence.csv')
    if not os.path.exists(occurrence_path):
        compute_organ_occurrence(
            file_index_path=file_index_path,
            output_path=occurrence_path,
            sample_size=sample_size,
            num_workers=num_workers
        )
    else:
        logger.info(f"器官出现频率表已存在,跳过: {occurrence_path}")
    
    # 任务0.3: 器官体积分布统计
    volumes_path = os.path.join(output_dir, 'stage0_organ_volumes.csv')
    if not os.path.exists(volumes_path):
        compute_organ_volumes(
            file_index_path=file_index_path,
            output_path=volumes_path,
            sample_size=sample_size,
            num_workers=num_workers
        )
    else:
        logger.info(f"器官体积统计表已存在,跳过: {volumes_path}")
    
    # 任务0.4: 器官共现矩阵
    cooccurrence_path = os.path.join(output_dir, 'stage0_organ_cooccurrence.csv')
    if not os.path.exists(cooccurrence_path):
        compute_organ_cooccurrence(
            file_index_path=file_index_path,
            output_path=cooccurrence_path,
            sample_size=sample_size,
            num_workers=num_workers
        )
    else:
        logger.info(f"器官共现矩阵已存在,跳过: {cooccurrence_path}")
    
    # 任务0.5: 部位分布分析
    body_part_path = os.path.join(output_dir, 'stage0_body_part_distribution.csv')
    if not os.path.exists(body_part_path):
        analyze_body_part_distribution(
            file_index_path=file_index_path,
            output_path=body_part_path,
            sample_size=sample_size,
            num_workers=num_workers
        )
    else:
        logger.info(f"部位分布表已存在,跳过: {body_part_path}")
    
    # 任务0.6: 初步异常检测
    outliers_path = os.path.join(output_dir, 'stage0_outliers.csv')
    if not os.path.exists(outliers_path):
        detect_outliers(
            file_index_path=file_index_path,
            organ_volumes_path=volumes_path,
            output_path=outliers_path,
            sample_size=sample_size,
            num_workers=num_workers
        )
    else:
        logger.info(f"异常检测结果已存在,跳过: {outliers_path}")
    
    # 生成阶段0汇总报告
    generate_stage0_report(output_dir)
    
    logger.info("\n阶段0完成!")
    
    return file_index_path, volumes_path


def run_stage1(file_index_path: str, volumes_path: str, 
               rules_path: str = 'config/hard_filter_rules.json',
               output_dir: str = 'outputs/stage1_硬过滤',
               sample_size: int = None, num_workers: int = None):
    """
    运行阶段1: 严重错误过滤
    
    Args:
        file_index_path: 文件索引表路径
        volumes_path: 器官体积统计表路径
        rules_path: 规则配置文件路径
        output_dir: 输出目录
        sample_size: 测试模式样本数量
    """
    logger.info("\n" + "=" * 70)
    logger.info("阶段1: 严重错误过滤")
    logger.info("=" * 70)
    
    ensure_dir(output_dir)
    
    filter_results_path = os.path.join(output_dir, 'stage1_filter_results.csv')
    reject_log_path = os.path.join(output_dir, 'stage1_reject_log.csv')
    
    if not os.path.exists(filter_results_path):
        run_hard_filter(
            file_index_path=file_index_path,
            rules_path=rules_path,
            volume_stats_path=volumes_path,
            output_path=filter_results_path,
            reject_log_path=reject_log_path,
            sample_size=sample_size,
            num_workers=num_workers
        )
    else:
        logger.info(f"过滤结果已存在,跳过: {filter_results_path}")
    
    # 生成阶段1汇总报告
    generate_stage1_report(output_dir)
    
    logger.info("\n阶段1完成!")
    
    return filter_results_path


def run_stage2(filter_results_path: str, 
               output_dir: str = 'outputs/stage2_质量评分',
               sample_size: int = None, checkpoint_interval: int = 10000,
               num_workers: int = None):
    """
    运行阶段2: 质量评分系统
    
    Args:
        filter_results_path: 过滤结果表路径
        output_dir: 输出目录
        sample_size: 测试模式样本数量
        checkpoint_interval: checkpoint保存间隔
    """
    logger.info("\n" + "=" * 70)
    logger.info("阶段2: 质量评分系统")
    logger.info("=" * 70)
    
    ensure_dir(output_dir)
    
    organ_scores_path = os.path.join(output_dir, 'stage2_organ_quality_scores.csv')
    case_summary_path = os.path.join(output_dir, 'stage2_case_summary.csv')
    
    if not os.path.exists(case_summary_path):
        run_quality_scoring(
            filter_results_path=filter_results_path,
            output_dir=output_dir,
            sample_size=sample_size,
            checkpoint_interval=checkpoint_interval,
            num_workers=num_workers
        )
    else:
        logger.info(f"质量评分结果已存在,跳过: {case_summary_path}")
    
    # 生成阶段2汇总报告
    generate_stage2_report(output_dir)
    
    logger.info("\n阶段2完成!")


def generate_stage0_report(output_dir: str):
    """生成阶段0汇总报告"""
    report_path = os.path.join(output_dir, 'stage0_summary_report.txt')
    
    try:
        # 加载各统计表
        file_index = pd.read_csv(os.path.join(output_dir, 'file_index.csv'))
        occurrence = pd.read_csv(os.path.join(output_dir, 'stage0_organ_occurrence.csv'))
        volumes = pd.read_csv(os.path.join(output_dir, 'stage0_organ_volumes.csv'))
        body_part = pd.read_csv(os.path.join(output_dir, 'stage0_body_part_distribution.csv'))
        outliers = pd.read_csv(os.path.join(output_dir, 'stage0_outliers.csv'))
        
        total_count = len(file_index)
        matched_count = file_index['ct_exists'].sum()
        matched_rate = matched_count / total_count * 100
        
        report = f"""
{'='*70}
阶段0: 数据探索与基线建立 - 汇总报告
{'='*70}

1. 数据集基本信息
   总文件数: {total_count:,}
   Mask-CT配对成功: {matched_count:,} ({matched_rate:.2f}%)
   
2. 器官统计
   检测到器官种类: {len(occurrence)}
   最常见器官TOP10:
"""
        for idx, row in occurrence.head(10).iterrows():
            report += f"      {idx+1}. {row['organ_name']}: {row['occurrence_count']:,} ({row['occurrence_rate']:.2f}%)\n"
        
        report += "\n   最罕见器官TOP10:\n"
        for idx, row in occurrence.tail(10).iterrows():
            report += f"      {idx+1}. {row['organ_name']}: {row['occurrence_count']:,} ({row['occurrence_rate']:.2f}%)\n"
        
        report += "\n3. 扫描部位分布\n"
        for idx, row in body_part.iterrows():
            report += f"   {row['body_part']:15s}: {row['case_count']:6,} ({row['percentage']:5.1f}%)\n"
        
        outlier_record_count = len(outliers)
        # 异常样本数 = 去重后的case数量
        outlier_case_count = outliers['accession_number'].nunique() if not outliers.empty else 0
        outlier_rate = outlier_case_count / total_count * 100 if total_count > 0 else 0
        
        report += f"""
4. 初步异常检测
   异常记录数: {outlier_record_count:,}
   异常样本数: {outlier_case_count:,} ({outlier_rate:.2f}%)
   
   按异常类型分布:
"""
        if not outliers.empty:
            type_counts = outliers['outlier_type'].value_counts()
            for outlier_type, count in type_counts.items():
                report += f"      {outlier_type}: {count:,}\n"
        
        report += f"""
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"阶段0汇总报告已生成: {report_path}")
    
    except Exception as e:
        logger.error(f"生成阶段0报告失败: {e}")


def generate_stage1_report(output_dir: str):
    """生成阶段1汇总报告"""
    report_path = os.path.join(output_dir, 'stage1_summary_report.txt')
    
    try:
        filter_results = pd.read_csv(os.path.join(output_dir, 'stage1_filter_results.csv'))
        reject_log = pd.read_csv(os.path.join(output_dir, 'stage1_reject_log.csv'))
        
        total = len(filter_results)
        passed = filter_results['passed_hard_filter'].sum()
        rejected = total - passed
        reject_rate = rejected / total * 100 if total > 0 else 0
        
        report = f"""
{'='*70}
阶段1: 严重错误过滤 - 汇总报告
{'='*70}

1. 过滤结果总览
   处理总数: {total:,}
   通过过滤: {passed:,} ({100-reject_rate:.1f}%)
   被拒绝: {rejected:,} ({reject_rate:.1f}%)

2. 拒绝原因分布 (按去重case统计)
"""
        # 按case去重统计拒绝原因
        case_error_types = reject_log.drop_duplicates(subset=['accession_number', 'series_number', 'error_type'])
        error_type_counts = case_error_types['error_type'].value_counts()
        for error_type, count in error_type_counts.head(10).items():
            percentage = count / rejected * 100 if rejected > 0 else 0
            report += f"   {error_type}: {count:,} ({percentage:.1f}%)\n"
        
        report += "\n3. 各部位拒绝率\n"
        body_part_reject = filter_results.groupby('inferred_body_part').agg({
            'passed_hard_filter': ['count', 'sum']
        }).reset_index()
        body_part_reject.columns = ['body_part', 'total', 'passed']
        body_part_reject['rejected'] = body_part_reject['total'] - body_part_reject['passed']
        body_part_reject['reject_rate'] = body_part_reject['rejected'] / body_part_reject['total'] * 100
        body_part_reject = body_part_reject.sort_values('reject_rate', ascending=False)
        
        for idx, row in body_part_reject.iterrows():
            report += f"   {row['body_part']:15s}: {row['reject_rate']:5.1f}%\n"
        
        report += "\n4. 最常见的具体错误TOP10\n"
        error_detail_counts = reject_log['error_detail'].value_counts()
        for idx, (error_detail, count) in enumerate(error_detail_counts.head(10).items()):
            report += f"   {idx+1}. {error_detail}: {count:,}\n"
        
        report += f"""
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"阶段1汇总报告已生成: {report_path}")
    
    except Exception as e:
        logger.error(f"生成阶段1报告失败: {e}")


def generate_stage2_report(output_dir: str):
    """生成阶段2汇总报告"""
    report_path = os.path.join(output_dir, 'stage2_summary_report.txt')
    
    try:
        organ_scores = pd.read_csv(os.path.join(output_dir, 'stage2_organ_quality_scores.csv'))
        case_summary = pd.read_csv(os.path.join(output_dir, 'stage2_case_summary.csv'))
        
        total_cases = len(case_summary)
        total_organs = len(organ_scores)
        avg_organs = case_summary['num_organs'].mean()
        
        report = f"""
{'='*70}
阶段2: 质量评分 - 汇总报告
{'='*70}

1. 评分结果总览
   评分总样本数: {total_cases:,}
   评分总器官数: {total_organs:,}
   平均每case器官数: {avg_organs:.1f}

2. 质量分级分布 (按case)
"""
        tier_counts = case_summary['quality_tier'].value_counts()
        for tier in ['HIGH', 'MEDIUM', 'LOW']:
            if tier in tier_counts:
                count = tier_counts[tier]
                percentage = count / total_cases * 100
                threshold = '>=0.8' if tier == 'HIGH' else ('0.6-0.8' if tier == 'MEDIUM' else '<0.6')
                report += f"   {tier} ({threshold}): {count:,} ({percentage:.1f}%)\n"
        
        report += f"""
3. 各评分维度平均分
   边界质量: {case_summary['avg_boundary_score'].mean():.3f}
   形态质量: {case_summary['avg_morphology_score'].mean():.3f}
   HU值分布: {case_summary['avg_hu_score'].mean():.3f}
   综合分数: {case_summary['case_overall_score'].mean():.3f}

4. 各器官平均分数TOP10 (最高质量)
"""
        organ_avg = organ_scores.groupby('organ_name').agg({
            'overall_score': 'mean',
            'organ_name': 'count'
        }).rename(columns={'organ_name': 'count'}).reset_index()
        organ_avg = organ_avg.sort_values('overall_score', ascending=False)
        
        for idx, row in organ_avg.head(10).iterrows():
            report += f"   {idx+1}. {row['organ_name']}: {row['overall_score']:.3f} (样本数: {row['count']:,})\n"
        
        report += "\n5. 各器官平均分数BOTTOM10 (最低质量)\n"
        for idx, row in organ_avg.tail(10).iterrows():
            report += f"   {idx+1}. {row['organ_name']}: {row['overall_score']:.3f} (样本数: {row['count']:,})\n"
        
        report += "\n6. 分数分布直方图 (case级别)\n"
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(case_summary['case_overall_score'], bins=bins)
        for i in range(len(hist)):
            report += f"   [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:,}\n"
        
        report += f"""
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"阶段2汇总报告已生成: {report_path}")
    
    except Exception as e:
        logger.error(f"生成阶段2报告失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='CT分割质量管理系统')
    parser.add_argument('--leaf-index', type=str, default='leaf_index.json',
                        help='leaf_index.json路径')
    parser.add_argument('--mask-base', type=str, default='/media/wmx/KINGIDISK/shenzhen_mask',
                        help='mask基础目录')
    parser.add_argument('--ct-base', type=str, default='/media/wmx/KINGIDISK',
                        help='CT基础目录')
    parser.add_argument('--stages', type=int, nargs='+', choices=[0, 1, 2],
                        default=[0, 1, 2], help='要运行的阶段')
    parser.add_argument('--sample', type=int, default=None,
                        help='测试模式样本数量')
    parser.add_argument('--workers', type=int, default=6,
                        help='并行worker数量(默认自动)')
    parser.add_argument('--checkpoint-interval', type=int, default=10000,
                        help='checkpoint保存间隔')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("CT分割质量管理系统 - 开始运行")
    logger.info("=" * 70)
    logger.info(f"运行阶段: {args.stages}")
    if args.sample:
        logger.info(f"测试模式: 处理 {args.sample} 个样本")
    
    # 确保输出目录存在
    ensure_dir('outputs')
    
    file_index_path = None
    volumes_path = None
    filter_results_path = None
    
    # 运行阶段0
    if 0 in args.stages:
        file_index_path, volumes_path = run_stage0(
            leaf_index_path=args.leaf_index,
            mask_base_dir=args.mask_base,
            ct_base_dir=args.ct_base,
            sample_size=args.sample,
            num_workers=args.workers
        )
    
    # 运行阶段1
    if 1 in args.stages:
        if file_index_path is None:
            file_index_path = 'outputs/stage0_数据探索/file_index.csv'
        if volumes_path is None:
            volumes_path = 'outputs/stage0_数据探索/stage0_organ_volumes.csv'
        
        filter_results_path = run_stage1(
            file_index_path=file_index_path,
            volumes_path=volumes_path,
            sample_size=args.sample,
            num_workers=args.workers
        )
    
    # 运行阶段2
    if 2 in args.stages:
        if filter_results_path is None:
            filter_results_path = 'outputs/stage1_硬过滤/stage1_filter_results.csv'
        
        run_stage2(
            filter_results_path=filter_results_path,
            sample_size=args.sample,
            checkpoint_interval=args.checkpoint_interval,
            num_workers=args.workers
        )
    
    logger.info("\n" + "=" * 70)
    logger.info("CT分割质量管理系统 - 运行完成!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
