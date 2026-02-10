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
import signal
import time
from functools import wraps

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
        
        # 加载体积阈值 - 使用p5/p95而非p1/p99，更宽松
        df_volumes = pd.read_csv(volume_stats_path)
        self.volume_thresholds = {}
        for idx, row in df_volumes.iterrows():
            # 使用p5/p95，并添加10%容差
            p05 = row['volume_p05_ml'] if 'volume_p05_ml' in row else row['volume_p01_ml']
            p95 = row['volume_p95_ml'] if 'volume_p95_ml' in row else row['volume_p99_ml']
            self.volume_thresholds[row['organ_name']] = {
                'min': p05 * 0.9,   # p5减10%
                'max': p95 * 1.1    # p95加10%
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
        检查体积异常 - 放宽阈值使用p5/p95，减少误判
        
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
            
            # 使用更宽松的阈值：p1/p99 -> p5/p95
            # 异常值定义为超出5%-95%范围，而非1%-99%
            min_vol = thresholds['min']  # p5
            max_vol = thresholds['max']  # p95
            
            # 额外放宽：允许10%的容差
            min_vol = min_vol * 0.9
            max_vol = max_vol * 1.1
            
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


def _init_worker():
    """子进程初始化函数 - 忽略SIGINT信号，让父进程处理"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("处理超时")


def _run_in_batches(work_items, rules_path, volume_stats_path, 
                   output_path, reject_log_path, num_workers):
    """
    分批处理大量数据，每批结束后保存进度
    避免单个进程池处理过多任务导致的问题
    """
    batch_size = 5000
    total = len(work_items)
    all_results = []
    
    logger.info(f"批处理模式: 总共 {total} 个任务, 每批 {batch_size} 个")
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = work_items[batch_start:batch_end]
        
        logger.info(f"处理批次 {batch_start//batch_size + 1}/{(total-1)//batch_size + 1}: {batch_start}-{batch_end}")
        
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=num_workers, initializer=_init_worker)
        
        batch_results = []
        try:
            # 使用imap_unordered流式处理
            iterator = pool.imap_unordered(_hard_filter_worker, batch, chunksize=2)
            
            with tqdm(total=len(batch), desc=f"批次 {batch_start//batch_size + 1}") as pbar:
                last_update = time.time()
                
                while len(batch_results) < len(batch):
                    try:
                        result = next(iterator)
                        batch_results.append(result)
                        pbar.update(1)
                        last_update = time.time()
                    except StopIteration:
                        break
                        
        finally:
            pool.close()
            pool.join()
        
        all_results.extend(batch_results)
        logger.info(f"批次完成: 获得 {len(batch_results)} 条结果，累计 {len(all_results)}/{total}")
        
        # 每批结束后保存中间结果
        if output_path and batch_results:
            temp_df = pd.DataFrame(all_results)
            temp_path = output_path.replace('.csv', f'_temp_batch{batch_start//batch_size + 1}.csv')
            temp_df.to_csv(temp_path, index=False)
            logger.info(f"中间结果已保存: {temp_path}")
    
    logger.info(f"所有批次完成，总共 {len(all_results)} 条结果")
    
    # 删除临时文件
    if output_path:
        import glob
        temp_pattern = output_path.replace('.csv', '_temp_batch*.csv')
        temp_files = glob.glob(temp_pattern)
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                logger.info(f"删除临时文件: {temp_file}")
            except Exception as e:
                logger.warning(f"删除临时文件失败 {temp_file}: {e}")
    
    # 处理拒绝记录
    reject_records = []
    for record in all_results:
        if not record.get('passed_hard_filter', False):
            for error in record.get('errors', '').split(';'):
                if not error:
                    continue
                if ':' in error:
                    error_type, error_detail = error.split(':', 1)
                else:
                    error_type = error
                    error_detail = ''

                severity = 'minor'
                if 'missing_critical' in error_type or 'cooccurrence_violated' in error_type:
                    severity = 'critical'
                elif 'volume' in error_type:
                    severity = 'major'
                elif 'timeout' in error_type:
                    severity = 'major'

                reject_records.append({
                    'accession_number': record['accession_number'],
                    'series_number': record['series_number'],
                    'error_type': error_type,
                    'error_detail': error_detail,
                    'severity': severity
                })
    
    # 保存最终结果
    df_result = pd.DataFrame(all_results)
    
    if output_path:
        df_result.to_csv(output_path, index=False)
        logger.info(f"过滤结果已保存: {output_path}")
    
    if reject_log_path and reject_records:
        df_reject = pd.DataFrame(reject_records)
        df_reject.to_csv(reject_log_path, index=False)
        logger.info(f"拒绝日志已保存: {reject_log_path}")
    
    # 统计
    total_cases = len(df_result)
    passed = df_result['passed_hard_filter'].sum()
    rejected = total_cases - passed
    reject_rate = rejected / total_cases * 100 if total_cases > 0 else 0
    
    logger.info("\n" + "=" * 70)
    logger.info("过滤结果总览:")
    logger.info(f"  处理总数: {total_cases:,}")
    logger.info(f"  通过过滤: {passed:,} ({100-reject_rate:.1f}%)")
    logger.info(f"  被拒绝: {rejected:,} ({reject_rate:.1f}%)")
    
    return df_result


def with_timeout(seconds):
    """装饰器：为函数添加超时机制"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 设置信号处理（仅Unix）
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # 取消定时器
                return result
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


def _hard_filter_worker(args: tuple) -> dict:
    """工作进程函数 - 使用spawn模式更安全，带超时和详细日志"""
    row_dict, rules_path, volume_stats_path = args
    mask_path = row_dict['mask_path']
    accession = row_dict['accession_number']
    
    start_time = time.time()
    
    try:
        # 每个文件30秒超时
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        filter_obj = HardFilter(rules_path, volume_stats_path)
        
        # 记录开始处理
        sys.stderr.write(f"[START] {accession} - {os.path.basename(mask_path)}\n")
        sys.stderr.flush()
        
        result = filter_obj.filter_case(mask_path, row_dict['ct_path'])
        
        # 取消定时器
        signal.alarm(0)
        
        elapsed = time.time() - start_time
        sys.stderr.write(f"[DONE] {accession} - {elapsed:.2f}s\n")
        sys.stderr.flush()
        
        return {
            'accession_number': accession,
            'series_number': row_dict['series_number'],
            'batch': row_dict['batch'],
            'mask_path': mask_path,
            'ct_path': row_dict['ct_path'],
            'inferred_body_part': result['inferred_body_part'],
            'detected_organ_count': result['detected_organ_count'],
            'passed_hard_filter': result['passed'],
            'error_count': result['error_count'],
            'errors': result['errors'],
            'process_time': elapsed
        }
    except TimeoutError as e:
        elapsed = time.time() - start_time
        sys.stderr.write(f"[TIMEOUT] {accession} - {elapsed:.2f}s\n")
        sys.stderr.flush()
        return {
            'accession_number': accession,
            'series_number': row_dict['series_number'],
            'batch': row_dict['batch'],
            'mask_path': mask_path,
            'ct_path': row_dict['ct_path'],
            'inferred_body_part': 'UNKNOWN',
            'detected_organ_count': 0,
            'passed_hard_filter': False,
            'error_count': 1,
            'errors': f'timeout:处理超过30秒',
            'process_time': elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        sys.stderr.write(f"[ERROR] {accession} - {e} - {elapsed:.2f}s\n")
        sys.stderr.flush()
        return {
            'accession_number': accession,
            'series_number': row_dict['series_number'],
            'batch': row_dict['batch'],
            'mask_path': mask_path,
            'ct_path': row_dict['ct_path'],
            'inferred_body_part': 'UNKNOWN',
            'detected_organ_count': 0,
            'passed_hard_filter': False,
            'error_count': 1,
            'errors': f'worker_error:{str(e)}',
            'process_time': elapsed
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

    if num_workers is None:
        num_workers = max(1, min(4, mp.cpu_count() - 1))

    work_items = []
    for _, row in df_valid.iterrows():
        work_items.append((row.to_dict(), rules_path, volume_stats_path))

    logger.info(f"启动多进程处理: {num_workers} workers, chunksize={chunksize}")
    
    # 根据样本数自动选择模式
    # 如果样本数很大(>10000)，使用批处理模式避免多进程问题
    if len(work_items) > 10000:
        logger.info(f"大样本量({len(work_items)}), 使用批处理模式")
        return _run_in_batches(work_items, rules_path, volume_stats_path, 
                               output_path, reject_log_path, num_workers)
    
    logger.info(f"使用 'spawn' 模式避免 fork 死锁问题")

    # 使用 spawn 模式创建进程池
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=num_workers, initializer=_init_worker)
    
    results = []
    
    try:
        # 使用imap_unordered流式处理，可以更好地处理异常
        iterator = pool.imap_unordered(_hard_filter_worker, work_items, chunksize=chunksize)
        
        with tqdm(total=len(work_items), desc="执行硬过滤") as pbar:
            last_update_time = time.time()
            last_count = 0
            
            while len(results) < len(work_items):
                try:
                    result = next(iterator)
                    results.append(result)
                    pbar.update(1)
                    last_update_time = time.time()
                    last_count = len(results)
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"处理任务出错: {e}")
                    pbar.update(1)
                    
    except KeyboardInterrupt:
        logger.warning("收到中断信号...")
        pool.terminate()
        raise
    finally:
        pool.close()
        pool.join()
    
    # 小样本模式：处理拒绝记录
    return _process_results(results, output_path, reject_log_path)


def _process_results(results, output_path, reject_log_path):
    """处理结果并生成报告"""
    # 处理拒绝记录
    reject_records = []
    for record in results:
        if not record.get('passed_hard_filter', False):
            for error in record.get('errors', '').split(';'):
                if not error:
                    continue
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
                elif 'timeout' in error_type:
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
