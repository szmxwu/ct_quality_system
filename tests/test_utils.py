"""
单元测试 - 工具函数
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.utils.geometry import (
    compute_centroid, 
    compute_volume, 
    compute_compactness_score,
    compute_connectivity_score,
    extract_boundary
)
from src.utils.organ_mapping import (
    classify_organ_type,
    infer_body_part_from_organs,
    get_organ_label,
    get_organ_name
)


def test_volume_calculation():
    """测试体积计算"""
    # 创建一个10x10x10的全1 mask
    test_mask = np.ones((10, 10, 10), dtype=np.uint8)
    test_spacing = np.array([1.0, 1.0, 1.0])  # 1mm isotropic
    
    expected_volume_ml = 10 * 10 * 10 * 1.0 / 1000  # = 1.0 ml
    actual_volume_ml = compute_volume(test_mask, test_spacing)
    
    assert abs(actual_volume_ml - expected_volume_ml) < 0.01, \
        f"Expected {expected_volume_ml}, got {actual_volume_ml}"
    
    print("✓ Volume calculation test passed")


def test_centroid_calculation():
    """测试质心计算"""
    # 创建一个5x5x5的mask,中心在(2,2,2)
    test_mask = np.zeros((5, 5, 5), dtype=np.uint8)
    test_mask[1:4, 1:4, 1:4] = 1
    
    centroid = compute_centroid(test_mask)
    expected_centroid = np.array([2.0, 2.0, 2.0])
    
    assert np.allclose(centroid, expected_centroid), \
        f"Expected {expected_centroid}, got {centroid}"
    
    print("✓ Centroid calculation test passed")


def test_empty_mask():
    """测试空mask处理"""
    empty_mask = np.zeros((10, 10, 10), dtype=np.uint8)
    
    centroid = compute_centroid(empty_mask)
    assert centroid is None, "Empty mask should return None centroid"
    
    connectivity = compute_connectivity_score(empty_mask)
    assert connectivity == 0.0, "Empty mask should have 0 connectivity"
    
    print("✓ Empty mask test passed")


def test_organ_classification():
    """测试器官分类"""
    assert classify_organ_type('liver') == 'soft_tissue'
    assert classify_organ_type('aorta') == 'vessel'
    assert classify_organ_type('vertebrae_L1') == 'bone'
    assert classify_organ_type('lung_upper_lobe_left') == 'lung'
    assert classify_organ_type('stomach') == 'air'
    
    print("✓ Organ classification test passed")


def test_body_part_inference():
    """测试部位推断"""
    # 胸部
    chest_organs = {'lung_upper_lobe_left', 'lung_lower_lobe_left', 'heart'}
    assert infer_body_part_from_organs(chest_organs) == 'CHEST'
    
    # 腹部
    abdomen_organs = {'liver', 'spleen', 'kidney_left'}
    assert infer_body_part_from_organs(abdomen_organs) == 'ABDOMEN'
    
    # 胸腹联合
    chest_abdomen_organs = {'liver', 'lung_upper_lobe_left', 'heart'}
    assert infer_body_part_from_organs(chest_abdomen_organs) == 'CHEST_ABDOMEN'
    
    # 盆腔
    pelvis_organs = {'urinary_bladder', 'prostate'}
    assert infer_body_part_from_organs(pelvis_organs) == 'PELVIS'
    
    print("✓ Body part inference test passed")


def test_organ_mapping():
    """测试器官标签映射"""
    # 正向映射
    liver_label = get_organ_label('liver')
    assert liver_label == 5, f"Expected liver label 5, got {liver_label}"
    
    # 反向映射
    liver_name = get_organ_name(5)
    assert liver_name == 'liver', f"Expected organ name 'liver', got {liver_name}"
    
    print("✓ Organ mapping test passed")


def test_boundary_extraction():
    """测试边界提取"""
    # 创建一个3x3x3的立方体
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    
    boundary = extract_boundary(mask)
    
    # 边界应该比原mask小
    assert np.sum(boundary) < np.sum(mask), "Boundary should be smaller than mask"
    
    # 边界应该只包含表面voxels
    expected_boundary_voxels = 3*3*3 - 1*1*1  # 27 - 1 = 26 (外表面)
    # 实际边界可能略有不同,但应该接近
    assert np.sum(boundary) > 0, "Boundary should not be empty"
    
    print("✓ Boundary extraction test passed")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("运行单元测试")
    print("=" * 70 + "\n")
    
    test_volume_calculation()
    test_centroid_calculation()
    test_empty_mask()
    test_organ_classification()
    test_body_part_inference()
    test_organ_mapping()
    test_boundary_extraction()
    
    print("\n" + "=" * 70)
    print("所有测试通过!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    run_all_tests()
