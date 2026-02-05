# CT分割质量管理系统

对33万例TotalSegmentator分割结果进行质量筛选,为训练YOLO器官检测模型准备高质量数据集。

## 项目概述

### 目标
- 识别图像中包含哪些器官并输出BBox
- 筛选高质量的分割结果用于模型训练
- 预计剔除3-8%的低质量数据

### 技术栈
- Python 3.8+
- NumPy, Pandas, scikit-image, scipy
- tqdm (进度条)

### 硬件限制
- 内存: 12GB (需优化内存使用)
- CPU: 6核处理器
- 存储: 300G可用空间存储中间结果

## 项目结构

```
ct_quality_system/
├── config/
│   └── hard_filter_rules.json      # 硬过滤规则配置
├── src/
│   ├── utils/
│   │   ├── organ_mapping.py        # 器官标签映射
│   │   ├── geometry.py             # 几何计算工具
│   │   └── file_io.py              # 文件读写工具
│   ├── stage0/                     # 数据探索与基线建立
│   │   ├── file_indexer.py         # 任务0.1: 构建文件索引
│   │   ├── organ_statistics.py     # 任务0.2-0.4: 器官统计
│   │   ├── body_part_analyzer.py   # 任务0.5: 部位分布分析
│   │   └── outlier_detector.py     # 任务0.6: 异常检测
│   ├── stage1/                     # 严重错误过滤
│   │   └── hard_filter.py          # 硬过滤实现
│   └── stage2/                     # 质量评分系统
│       └── quality_scorer.py       # 质量评分实现
├── outputs/                        # 输出目录
│   ├── stage0_数据探索/             # 阶段0输出
│   ├── stage1_硬过滤/               # 阶段1输出
│   └── stage2_质量评分/             # 阶段2输出
├── main_pipeline.py                # 主流程脚本
├── Seg_lables.py                   # 器官标签定义
├── match_data.py                   # Mask提取工具
└── requirements.txt                # 依赖包清单
```

## 数据结构

### 目录结构
```
项目根目录/
├── shenzhen_mask/              # 分割结果目录
│   ├── 2410/                   # 批次目录
│   │   ├── 000175/             # 患者ID
│   │   │   └── M24101000105/   # AccessionNumber目录
│   │   │       ├── 201.npz
│   │   │       └── ...
│   └── ...
├── 2410/                       # 原始CT目录
│   ├── 000175/
│   │   └── M24101000105/
│   │       ├── 201.npz
│   │       └── ...
└── ...
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

## 使用说明

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
python main_pipeline.py --leaf-index /path/to/leaf_index.json --mask-base /media/wmx/KINGIDISK/shenzhen_mask --ct-base /media/wmx/KINGIDISK/
```

### 单独运行各阶段

**阶段0: 数据探索**
```bash
python src/stage0/file_indexer.py --leaf-index /path/to/leaf_index.json
python src/stage0/organ_statistics.py --task occurrence --index outputs/stage0_数据探索/file_index.csv
python src/stage0/organ_statistics.py --task volumes --index outputs/stage0_数据探索/file_index.csv
python src/stage0/organ_statistics.py --task cooccurrence --index outputs/stage0_数据探索/file_index.csv
python src/stage0/body_part_analyzer.py --index outputs/stage0_数据探索/file_index.csv
python src/stage0/outlier_detector.py --index outputs/stage0_数据探索/file_index.csv --volumes outputs/stage0_数据探索/stage0_organ_volumes.csv
```

**阶段1: 硬过滤**
```bash
python src/stage1/hard_filter.py --index outputs/stage0_数据探索/file_index.csv --volumes outputs/stage0_数据探索/stage0_organ_volumes.csv
```

**阶段2: 质量评分**
```bash
python src/stage2/quality_scorer.py --filter-results outputs/stage1_硬过滤/stage1_filter_results.csv
```

## 输出文件

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

## 质量评分维度

每个器官独立评分,包含以下维度:

| 评分维度 | 权重 | 说明 |
|---------|------|------|
| 边界质量 | 50% | 分割边界是否对应CT密度梯度 |
| 形态学质量 | 30% | 连通性、紧凑度等形态指标 |
| HU值分布 | 20% | 器官内CT值是否符合预期 |

**综合分数** = 0.5×边界分数 + 0.3×形态分数 + 0.2×HU分数

## 质量分级标准

- **HIGH**: 综合分数 ≥ 0.8
- **MEDIUM**: 0.6 ≤ 综合分数 < 0.8
- **LOW**: 综合分数 < 0.6

## 硬过滤规则

### 关键器官缺失检查
- 胸部扫描必须检测到双肺
- 腹部扫描必须检测到肝脏
- 盆腔扫描必须检测到膀胱

### 共现规则检查
- 有脾脏必有肝脏
- 有胆囊必有肝脏
- 有门静脉必有肝脏
- 有前列腺必有膀胱

### 体积异常检查
- 体积 < P1: 异常小
- 体积 > P99: 异常大

### 空间关系检查
- 脾脏应在肝脏左侧
- 双肾应大致在同一水平

## 注意事项

1. **数据路径**: 确保leaf_index.json中的路径与实际数据路径一致
2. **内存管理**: 处理大文件时会自动释放内存
3. **进度保存**: 阶段2会定期保存checkpoint,防止意外中断
4. **日志记录**: 所有操作都会记录到outputs/pipeline.log

## 开发指南

### 添加新的过滤规则
编辑 `config/hard_filter_rules.json`:
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

## 性能优化

- 批量处理: 每次处理1000个文件
- 多进程并行: 可使用 `--workers` 参数(待实现)
- 内存管理: 处理完一个文件后立即释放内存
- 进度保存: 每处理10000个样本保存checkpoint

## 验证要点

**阶段0**:
- [ ] 文件索引的配对成功率 > 90%
- [ ] 器官出现频率: 肝脏应最高(>80%)
- [ ] 主要器官体积中位数: 肝脏 1200-1800ml

**阶段1**:
- [ ] 拒绝率在3-8%之间
- [ ] 最常见拒绝原因是"关键器官缺失"

**阶段2**:
- [ ] 所有分数都在0-1之间
- [ ] 平均综合分数在0.6-0.8之间
- [ ] HIGH质量样本 > 50%
