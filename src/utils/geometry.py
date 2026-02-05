"""
几何计算工具函数
"""
import numpy as np
from scipy import ndimage


def compute_centroid(mask: np.ndarray) -> np.ndarray:
    """
    计算3D mask的质心坐标
    
    Args:
        mask: np.ndarray, shape (D, H, W), dtype uint8/bool
    
    Returns:
        np.ndarray, shape (3,), [z_centroid, y_centroid, x_centroid]
        如果mask为空,返回None
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0)  # [z_mean, y_mean, x_mean]


def compute_volume(mask: np.ndarray, spacing: np.ndarray) -> float:
    """
    计算器官体积(ml)
    
    Args:
        mask: np.ndarray, 二值mask
        spacing: np.ndarray, shape (3,), [z_spacing, y_spacing, x_spacing] in mm
    
    Returns:
        float: 体积(ml)
    """
    voxel_count = np.sum(mask)
    voxel_volume_mm3 = np.prod(spacing)
    volume_ml = voxel_count * voxel_volume_mm3 / 1000
    return volume_ml


def compute_surface_voxels(mask: np.ndarray) -> int:
    """
    计算表面voxel数量
    
    Args:
        mask: np.ndarray, 二值mask
    
    Returns:
        int: 表面voxel数量
    """
    eroded = ndimage.binary_erosion(mask)
    surface_voxels = np.sum(mask) - np.sum(eroded)
    return surface_voxels


def compute_compactness_score(mask: np.ndarray) -> float:
    """
    计算紧凑度分数
    
    Args:
        mask: np.ndarray, 二值mask
    
    Returns:
        float: 紧凑度分数 (0-1)
    """
    volume = np.sum(mask)
    if volume == 0:
        return 0.0
    
    surface_voxels = compute_surface_voxels(mask)
    
    # 理想球体的表面积/体积比
    ideal_ratio = 4.836 * (volume ** (2/3)) / volume
    actual_ratio = surface_voxels / volume
    
    compactness_score = min(ideal_ratio / (actual_ratio + 1e-6), 1.0)
    return compactness_score


def compute_connectivity_score(mask: np.ndarray) -> float:
    """
    计算连通性分数
    
    Args:
        mask: np.ndarray, 二值mask
    
    Returns:
        float: 连通性分数 (0-1)
    """
    from skimage import measure
    
    labels, num_components = measure.label(mask, return_num=True)
    
    if num_components == 0:
        return 0.0
    
    # 计算每个连通分量的大小
    component_sizes = []
    for i in range(1, num_components + 1):
        size = np.sum(labels == i)
        component_sizes.append(size)
    
    # 最大连通分量占总体积的比例
    largest_ratio = max(component_sizes) / sum(component_sizes)
    return largest_ratio


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """
    提取mask的边界voxels
    
    Args:
        mask: np.ndarray, 二值mask
    
    Returns:
        np.ndarray: 边界mask
    """
    eroded = ndimage.binary_erosion(mask, iterations=1)
    boundary = mask & ~eroded
    return boundary


def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    计算图像梯度幅值
    
    Args:
        image: np.ndarray, CT图像
    
    Returns:
        np.ndarray: 梯度幅值
    """
    gradient_z = np.gradient(image, axis=0)
    gradient_y = np.gradient(image, axis=1)
    gradient_x = np.gradient(image, axis=2)
    gradient_magnitude = np.sqrt(gradient_z**2 + gradient_y**2 + gradient_x**2)
    return gradient_magnitude
