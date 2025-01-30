"""评估指标工具."""

import logging
import numpy as np
from collections import Counter
from typing import Dict

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def print_class_distribution(dataset: Dataset) -> None:
    """打印数据集的类别分布.
    
    Args:
        dataset: 数据集对象
    """
    labels = [dataset.labels[idx] for idx in dataset.indices]
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    logger.info("\n类别分布:")
    for label in sorted(class_counts.keys()):
        count = class_counts[label]
        percentage = count / total_samples * 100
        logger.info(f"类别 {label}: {count} 样本 ({percentage:.2f}%)") 