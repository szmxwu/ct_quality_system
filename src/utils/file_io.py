"""
文件读写工具函数
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_leaf_index(index_path: str) -> Dict[str, str]:
    """
    加载leaf_index.json索引文件
    
    Args:
        index_path: 索引文件路径
    
    Returns:
        Dict: {accession_number: directory_path}
    """
    with open(index_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('index', {})


def find_mask_files(mask_base_dir: str, leaf_index: Dict[str, str]) -> List[Dict]:
    """
    根据leaf_index查找所有mask文件
    
    Args:
        mask_base_dir: mask基础目录 (如 '/media/wmx/KINGIDISK/shenzhen_mask')
        leaf_index: leaf_index字典
    
    Returns:
        List[Dict]: 文件信息列表
    """
    records = []
    
    for accession_number, ct_dir in leaf_index.items():
        # 从CT路径推断mask路径
        # CT路径: /media/wmx/KINGIDISK/2410/0000109279/M24101403890
        # Mask路径: /media/wmx/KINGIDISK/shenzhen_mask/2410/0000109279/M24101403890
        
        parts = ct_dir.split('/')
        if len(parts) >= 3:
            batch = parts[1]  # 2410
            patient_id = parts[2]  # 0000109279
            
            mask_dir = os.path.join(mask_base_dir, batch, patient_id, accession_number)
            
            if os.path.exists(mask_dir):
                # 查找该目录下的所有.npz文件
                for npz_file in Path(mask_dir).glob('*.npz'):
                    series_number = npz_file.stem
                    records.append({
                        'batch': batch,
                        'patient_id': patient_id,
                        'accession_number': accession_number,
                        'series_number': series_number,
                        'mask_path': str(npz_file),
                        'ct_dir': ct_dir
                    })
    
    return records


def find_matching_ct(mask_record: Dict, ct_base_dir: str = '/media/wmx/KINGIDISK/') -> Optional[str]:
    """
    查找mask对应的CT文件
    
    Args:
        mask_record: mask记录字典
        ct_base_dir: CT基础目录
    
    Returns:
        str or None: CT文件路径
    """
    batch = mask_record['batch']
    patient_id = mask_record['patient_id']
    accession_number = mask_record['accession_number']
    series_number = mask_record['series_number']
    
    # 构建CT文件路径
    ct_dir = os.path.join(ct_base_dir, batch, patient_id, accession_number)
    ct_path = os.path.join(ct_dir, f"{series_number}.npz")
    
    if os.path.exists(ct_path):
        return ct_path
    
    return None


def load_npz_file(npz_path: str) -> Optional[Dict]:
    """
    安全加载NPZ文件
    
    Args:
        npz_path: NPZ文件路径
    
    Returns:
        Dict or None: 加载的数据
    """
    try:
        data = np.load(npz_path)
        return {key: data[key] for key in data.files}
    except FileNotFoundError:
        logger.error(f"File not found: {npz_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading {npz_path}: {str(e)}")
        return None


def load_npz_keys(npz_path: str, keys: List[str]) -> Optional[Dict]:
    """
    只加载npz中指定的key，避免不必要的内存占用

    Args:
        npz_path: NPZ文件路径
        keys: 需要加载的key列表

    Returns:
        Dict or None: 指定key的数据
    """
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            return {key: data[key] for key in keys if key in data}
    except FileNotFoundError:
        logger.error(f"File not found: {npz_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading {npz_path}: {str(e)}")
        return None


def save_checkpoint(data: List[Dict], checkpoint_path: str):
    """
    保存checkpoint
    
    Args:
        data: 数据列表
        checkpoint_path: 保存路径
    """
    df = pd.DataFrame(data)
    df.to_csv(checkpoint_path, index=False)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str) -> pd.DataFrame:
    """
    加载checkpoint
    
    Args:
        checkpoint_path: checkpoint路径
    
    Returns:
        pd.DataFrame: 数据
    """
    return pd.read_csv(checkpoint_path)


def ensure_dir(dir_path: str):
    """
    确保目录存在
    
    Args:
        dir_path: 目录路径
    """
    os.makedirs(dir_path, exist_ok=True)
