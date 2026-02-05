"""
器官标签映射工具
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from Seg_lables import totalsegmentator_dict

# 正向映射: 器官名称 -> 标签值
ORGAN_TO_LABEL = totalsegmentator_dict

# 反向映射: 标签值 -> 器官名称
LABEL_TO_ORGAN = {v: k for k, v in totalsegmentator_dict.items()}

# 器官分类
def classify_organ_type(organ_name: str) -> str:
    """
    根据器官名称分类器官类型
    
    Returns:
        str: 器官类型 ('soft_tissue', 'vessel', 'bone', 'lung', 'air')
    """
    soft_tissue_organs = [
        'liver', 'spleen', 'kidney_left', 'kidney_right', 'pancreas',
        'heart', 'prostate', 'gallbladder', 'adrenal_gland_left', 'adrenal_gland_right'
    ]
    vessel_organs = [
        'aorta', 'portal_vein_and_splenic_vein', 'pulmonary_vein',
        'superior_vena_cava', 'inferior_vena_cava', 
        'iliac_artery_left', 'iliac_artery_right'
    ]
    lung_organs = [
        'lung_upper_lobe_left', 'lung_lower_lobe_left',
        'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right'
    ]
    air_organs = ['stomach', 'small_bowel', 'duodenum', 'colon', 'trachea', 'esophagus']
    
    if organ_name in soft_tissue_organs:
        return 'soft_tissue'
    elif organ_name in vessel_organs or 'vein' in organ_name or 'artery' in organ_name:
        return 'vessel'
    elif any(bone_keyword in organ_name for bone_keyword in ['vertebrae', 'rib', 'femur', 'humerus', 'scapula', 'clavicula', 'hip', 'sacrum', 'sternum']):
        return 'bone'
    elif organ_name in lung_organs:
        return 'lung'
    elif organ_name in air_organs:
        return 'air'
    else:
        return 'soft_tissue'  # 默认


def get_organ_label(organ_name: str) -> int:
    """获取器官标签值"""
    return ORGAN_TO_LABEL.get(organ_name, None)


def get_organ_name(label: int) -> str:
    """根据标签值获取器官名称"""
    return LABEL_TO_ORGAN.get(label, None)


def get_all_organ_names() -> list:
    """获取所有器官名称列表"""
    return list(ORGAN_TO_LABEL.keys())


def get_all_labels() -> list:
    """获取所有标签值列表"""
    return list(LABEL_TO_ORGAN.keys())


# 部位推断相关
def infer_body_part_from_organs(detected_organs: set) -> str:
    """
    根据检测到的器官推断扫描部位
    
    Args:
        detected_organs: set of organ names
    
    Returns:
        str: 部位名称 ('CHEST', 'ABDOMEN', 'PELVIS', 'CHEST_ABDOMEN', 
             'HEAD', 'EXTREMITY', 'UNKNOWN')
    """
    # 定义各部位的特征器官
    chest_indicators = {
        'lung_upper_lobe_left', 'lung_lower_lobe_left',
        'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right',
        'heart', 'trachea'
    }
    abdomen_indicators = {'liver', 'spleen', 'kidney_left', 'kidney_right', 'pancreas'}
    pelvis_indicators = {'urinary_bladder', 'prostate', 'femur_left', 'femur_right'}
    head_indicators = {'brain', 'skull'}
    extremity_indicators = {'humerus_left', 'humerus_right', 'femur_left', 'femur_right'}
    
    # 计算交集
    has_chest = len(chest_indicators & detected_organs) >= 2
    has_abdomen = len(abdomen_indicators & detected_organs) >= 1
    has_pelvis = len(pelvis_indicators & detected_organs) >= 1
    has_head = len(head_indicators & detected_organs) >= 1
    has_extremity = len(extremity_indicators & detected_organs) >= 1 and not (has_chest or has_abdomen)
    
    # 判断
    if has_chest and has_abdomen:
        return 'CHEST_ABDOMEN'
    elif has_chest:
        return 'CHEST'
    elif has_abdomen:
        return 'ABDOMEN'
    elif has_pelvis:
        return 'PELVIS'
    elif has_head:
        return 'HEAD'
    elif has_extremity:
        return 'EXTREMITY'
    else:
        return 'UNKNOWN'
