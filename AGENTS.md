# CT分割质量管理系统 - AI Agent指南

## 项目概述

本项目是一个CT分割质量管理系统,用于对33万例TotalSegmentator分割结果进行质量筛选,为训练YOLO器官检测模型准备高质量数据集。

**核心目标:**
- 识别图像中包含哪些器官并输出BBox
- 筛选高质量的分割结果用于模型训练
- 预计剔除3-8%的低质量数据

**处理流程:**
1. **阶段0** - 数据探索与基线建立: 构建文件索引、统计器官出现频率、体积分布、共现矩阵、部位分布、异常检测
2. **阶段1** - 严重错误过滤: 基于规则的硬过滤(关键器官缺失、共现规则、体积异常、空间关系)
3. **阶段2** - 质量评分系统: 对每个器官的边界质量、形态学质量、HU值分布进行评分

## 技术栈

- **语言:** Python 3.8+
- **核心依赖:** NumPy, Pandas, scikit-image, scipy, tqdm
- **数据格式:** NPZ (NumPy压缩格式)
- **分割标准:** TotalSegmentator (117个器官类别)

## 项目结构

```
ct_quality_system/
├── config/
│   └── hard_filter_rules.json      # 硬过滤规则配置
├── src/
│   ├── utils/                      # 工具函数模块
│   │   ├── organ_mapping.py        # 器官标签映射、部位推断
│   │   ├── geometry.py             # 几何计算工具(体积、质心、边界等)
│   │   └── file_io.py              # 文件读写工具
│   ├── stage0/                     # 阶段0: 数据探索与基线建立
│   │   ├── file_indexer.py         # 任务0.1: 构建文件索引
│   │   ├── organ_statistics.py     # 任务0.2-0.4: 器官统计
│   │   ├── body_part_analyzer.py   # 任务0.5: 部位分布分析
│   │   └── outlier_detector.py     # 任务0.6: 异常检测
│   ├── stage1/                     # 阶段1: 严重错误过滤
│   │   └── hard_filter.py          # 硬过滤实现
│   └── stage2/                     # 阶段2: 质量评分系统
│       └── quality_scorer.py       # 质量评分实现
├── outputs/                        # 输出目录
│   ├── stage0_数据探索/             # 阶段0输出
│   ├── stage1_硬过滤/               # 阶段1输出
│   └── stage2_质量评分/             # 阶段2输出
├── main_pipeline.py                # 主流程脚本
├── Seg_lables.py                   # 器官标签定义(117个器官)
├── match_data.py                   # Mask提取工具
├── total_segmentator_dicts.json    # 器官中英文对照
├── index_leaf_dirs.py              # 目录索引工具
├── requirements.txt                # 依赖包清单
└── tests/
    └── test_utils.py               # 单元测试
```

## 数据格式

### 目录结构
```
项目根目录/
├── shenzhen_mask/              # 分割结果目录
│   ├── 2410/                   # 批次目录
│   │   ├── 000175/             # 患者ID
│   │   │   └── M24101000105/   # AccessionNumber目录
│   │   │       ├── 201.npz
│   │   │       └── ...
├── 2410/                       # 原始CT目录
│   └── ...
└── leaf_index.json             # 目录索引文件
```

### NPZ文件结构

**Mask NPZ文件** (shenzhen_mask目录下):
```python
{
    'mask': np.ndarray,     # shape: (D, H, W), dtype: uint8
                            # 每个voxel值 = 器官类别ID (1-117)
}
```

**原始CT NPZ文件**:
```python
{
    'image': np.ndarray,    # shape: (D, H, W), dtype: int16 (HU值)
    'spacing': np.ndarray,  # shape: (3,), [z_spacing, y_spacing, x_spacing] in mm
}
```

### 器官标签定义

器官标签定义在 `Seg_lables.py` 中的 `totalsegmentator_dict`,包含117个器官:
- 1-21: 主要器官(肝脏、脾脏、肾脏、肺叶等)
- 22-50: 骨骼结构(椎骨、骶骨等)
- 51-68: 血管系统(主动脉、静脉等)
- 69-78: 四肢骨骼(肱骨、股骨等)
- 79-89: 肌肉组织
- 90-91: 头部(大脑、颅骨)
- 92-117: 肋骨和胸骨

## 运行命令

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行完整Pipeline
```bash
# 运行所有阶段
python main_pipeline.py --stages 0 1 2

# 测试模式 (100个样本)
python main_pipeline.py --stages 0 1 2 --sample 100

# 只运行阶段0
python main_pipeline.py --stages 0

# 指定数据路径
python main_pipeline.py --leaf-index /path/to/leaf_index.json \
    --mask-base /media/wmx/KINGIDISK/shenzhen_mask \
    --ct-base /media/wmx/KINGIDISK/
```

### 单独运行各阶段

**阶段0: 数据探索**
```bash
python src/stage0/file_indexer.py --leaf-index leaf_index.json
python src/stage0/organ_statistics.py --task occurrence --index outputs/stage0_数据探索/file_index.csv
python src/stage0/organ_statistics.py --task volumes --index outputs/stage0_数据探索/file_index.csv
python src/stage0/organ_statistics.py --task cooccurrence --index outputs/stage0_数据探索/file_index.csv
python src/stage0/body_part_analyzer.py --index outputs/stage0_数据探索/file_index.csv
python src/stage0/outlier_detector.py --index outputs/stage0_数据探索/file_index.csv \
    --volumes outputs/stage0_数据探索/stage0_organ_volumes.csv
```

**阶段1: 硬过滤**
```bash
python src/stage1/hard_filter.py --index outputs/stage0_数据探索/file_index.csv \
    --volumes outputs/stage0_数据探索/stage0_organ_volumes.csv
```

**阶段2: 质量评分**
```bash
python src/stage2/quality_scorer.py --filter-results outputs/stage1_硬过滤/stage1_filter_results.csv
```

### 运行测试
```bash
python tests/test_utils.py
```

## 代码风格指南

### 命名规范
- **函数名:** 小写+下划线 (`compute_organ_volumes`, `run_hard_filter`)
- **类名:** 大驼峰 (`HardFilter`)
- **常量:** 全大写 (`HU_RANGES`, `SOLID_ORGANS`)
- **文件名:** 小写+下划线 (`hard_filter.py`, `organ_mapping.py`)

### 代码组织
- 每个模块文件开头包含中文文档字符串说明功能
- 使用类型注解 (Python typing)
- 使用多进程并行处理大量数据 (`multiprocessing.Pool`)
- 日志使用 `logging` 模块,级别为 INFO

### 导入规范
```python
# 标准库
import os
import sys
import json
import logging
import multiprocessing as mp

# 第三方库
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure
from tqdm import tqdm

# 项目内部模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.file_io import load_npz_keys
from match_data import get_mask_labels
```

## 核心模块说明

### src/utils/organ_mapping.py
- `ORGAN_TO_LABEL`: 器官名称到标签值的正向映射
- `LABEL_TO_ORGAN`: 标签值到器官名称的反向映射
- `classify_organ_type()`: 根据器官名称分类器官类型
- `infer_body_part_from_organs()`: 根据检测到的器官推断扫描部位

### src/utils/geometry.py
- `compute_centroid()`: 计算3D mask的质心坐标
- `compute_volume()`: 计算器官体积(ml)
- `compute_surface_voxels()`: 计算表面voxel数量
- `compute_compactness_score()`: 计算紧凑度分数
- `compute_connectivity_score()`: 计算连通性分数
- `extract_boundary()`: 提取mask的边界voxels
- `compute_gradient_magnitude()`: 计算图像梯度幅值

### src/utils/file_io.py
- `load_leaf_index()`: 加载leaf_index.json索引文件
- `find_mask_files()`: 根据leaf_index查找所有mask文件
- `find_matching_ct()`: 查找mask对应的CT文件
- `load_npz_file()`: 安全加载NPZ文件
- `load_npz_keys()`: 只加载npz中指定的key
- `save_checkpoint()`: 保存checkpoint
- `ensure_dir()`: 确保目录存在

## 配置文件

### config/hard_filter_rules.json
定义硬过滤规则:
- `critical_organ_rules`: 关键器官规则(按部位定义)
- `cooccurrence_rules`: 共现规则(有A必有B)
- `spatial_constraints`: 空间约束(器官位置关系)
- `minimum_organ_count`: 各部位最少器官数量

## 质量评分维度

每个器官独立评分,权重如下:

| 评分维度 | 权重 | 说明 |
|---------|------|------|
| 边界质量 | 50% | 分割边界是否对应CT密度梯度 |
| 形态学质量 | 30% | 连通性、紧凑度等形态指标 |
| HU值分布 | 20% | 器官内CT值是否符合预期 |

**综合分数** = 0.5×边界分数 + 0.3×形态分数 + 0.2×HU分数

### 质量分级标准
- **HIGH**: 综合分数 ≥ 0.8
- **MEDIUM**: 0.6 ≤ 综合分数 < 0.8
- **LOW**: 综合分数 < 0.6

## 输出文件说明

### 阶段0输出
- `file_index.csv`: 文件索引表
- `stage0_organ_occurrence.csv`: 器官出现频率统计
- `stage0_organ_volumes.csv`: 器官体积分布统计
- `stage0_organ_cooccurrence.csv`: 器官共现矩阵
- `stage0_body_part_distribution.csv`: 部位分布分析
- `stage0_outliers.csv`: 初步异常检测
- `stage0_summary_report.txt`: 汇总报告

### 阶段1输出
- `stage1_filter_results.csv`: 过滤结果
- `stage1_reject_log.csv`: 拒绝日志
- `stage1_summary_report.txt`: 汇总报告

### 阶段2输出
- `stage2_organ_quality_scores.csv`: 器官级别评分
- `stage2_case_summary.csv`: case级别汇总
- `stage2_summary_report.txt`: 汇总报告
- `stage2_checkpoint_*.csv`: 进度检查点

## 注意事项

1. **内存管理:** 
   - 处理大文件后会立即释放内存 (`del ct_data`)
   - 使用 `load_npz_keys()` 只加载需要的key

2. **多进程:**
   - 默认worker数量: `max(1, min(4, mp.cpu_count() - 1))`
   - 使用 `mp.get_context("fork").Pool()` 创建进程池

3. **路径处理:**
   - Windows路径使用 `replace('/', '\\').split('\\')` 处理
   - 默认数据路径基于 `/media/wmx/KINGIDISK/`

4. **测试模式:**
   - 使用 `--sample N` 参数限制处理样本数
   - 适合开发和调试使用

5. **进度保存:**
   - 阶段2每处理10000个样本保存checkpoint
   - 防止意外中断丢失进度

## 扩展开发指南

### 添加新的过滤规则
编辑 `config/hard_filter_rules.json`,在相应部分添加规则:
```json
{
  "cooccurrence_rules": [
    {
      "if_exists": "organ_a",
      "must_exist": ["organ_b"],
      "reason": "有organ_a必有organ_b"
    }
  ]
}
```

### 添加新的HU范围
编辑 `src/stage2/quality_scorer.py` 中的 `HU_RANGES` 字典。

### 添加新的评分维度
1. 在 `src/stage2/quality_scorer.py` 中实现评分函数
2. 在 `compute_overall_score()` 中调整权重
3. 更新输出记录的字段

## 验证要点

**阶段0:**
- 文件索引的配对成功率 > 90%
- 器官出现频率: 肝脏应最高(>80%)
- 主要器官体积中位数: 肝脏 1200-1800ml

**阶段1:**
- 拒绝率在3-8%之间
- 最常见拒绝原因是"关键器官缺失"

**阶段2:**
- 所有分数都在0-1之间
- 平均综合分数在0.6-0.8之间
- HIGH质量样本 > 50%
