# totalsegmentator_dict字典里定义了器官名称和对应的标签值
from Seg_lables import totalsegmentator_dict
import os
import numpy as np
import json


totalsegmentator_dict = {v: k for k, v in totalsegmentator_dict.items()}


def load_mask_array(npz_path: str) -> np.ndarray:
    """加载npz中的mask数组"""
    with np.load(npz_path, allow_pickle=False) as data:
        return data['mask']


def get_mask_labels(npz_path: str) -> np.ndarray:
    """获取mask中出现的标签值(不包含背景0)"""
    mask = load_mask_array(npz_path)
    labels = np.unique(mask)
    labels = labels[labels != 0]
    return labels.astype(np.int32, copy=False)


def get_mask_label_counts(npz_path: str) -> dict:
    """获取mask中各标签的体素计数(不包含背景0)"""
    mask = load_mask_array(npz_path)
    mask_int = mask.astype(np.int64, copy=False)
    counts = np.bincount(mask_int.ravel())
    result = {}
    for label, count in enumerate(counts):
        if label == 0 or count == 0:
            continue
        result[int(label)] = int(count)
    return result


def get_npz_masks(npz_path: str) -> dict:
    """从 .npz 文件中提取器官mask，输出一个字典，键为器官名称，值为对应的二值mask数组。"""
    masks = {}
    with np.load(npz_path, allow_pickle=False) as data:
        mask_array = data['mask']
        mask_values = np.unique(mask_array)
        for i in mask_values:
            i = int(i)
            if i == 0:
                continue  # 跳过背景
            label = totalsegmentator_dict[i]
            masks[label] = (mask_array == i).astype(np.uint8)
    return masks
if __name__ == '__main__':
    # 示例用法
    npz_file = '/media/wmx/KINGIDISK/shenzhen_mask/2410/000546/M24101000105/202.npz'

    masks = get_npz_masks(npz_file)
    labels = list(masks.keys())
    print(f'Masks in {npz_file} with {len(labels)}: {labels}')
