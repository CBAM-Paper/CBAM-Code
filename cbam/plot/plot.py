#!/usr/bin/env python3
#
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
"""
This file contains general plot functionality.
"""

from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib import patches


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Default Settings
# -----------------------------------------------------------------------------
def cm(value: float or int) -> float:
    """Calculate the number that has to be given as size to obtain cm."""
    return value / 2.54


# 计算 IEEE 格式中的宽度，241.14749 是某个特定宽度，0.03514 是转换系数（可能是英寸转厘米的系数）
IEEE_WIDTH = 241.14749 * 0.03514  # pt in cm

# 设置图表的基本字体大小为 9
font_size = 9

# 设置刻度标签的字体大小，比基本字体小 1 个单位（即 8）
ticks_fontsize = font_size - 1

# 设置图例的字体大小，与刻度标签一致（即 8）
legend_font_size = font_size - 1

# 定义 matplotlib 的默认图表样式设置
default_settings = {
    'font.size': font_size,  # 设置字体的基本大小
    'legend.fontsize': legend_font_size,  # 设置图例的字体大小
    'axes.titlesize': font_size,  # 设置坐标轴标题的字体大小
    'axes.labelsize': font_size,  # 设置坐标轴标签的字体大小
    'ytick.labelsize': ticks_fontsize,  # 设置 y 轴刻度标签的字体大小
    'xtick.labelsize': ticks_fontsize,  # 设置 x 轴刻度标签的字体大小
    'hatch.linewidth': 0.8,  # 设置图形填充（hatch）的边线宽度
    'xtick.minor.pad': 1,  # 设置 x 轴次刻度与主刻度之间的间距为 1
    'axes.labelpad': 3,  # 设置坐标轴标签与坐标轴之间的间距为 3
    'legend.framealpha': 1,  # 设置图例框的透明度为 1（完全不透明）
    'legend.edgecolor': 'black',  # 设置图例框的边缘颜色为黑色
    'legend.fancybox': False,  # 设置图例框不使用圆角（使用方形边框）
    'legend.handletextpad': 0.2,  # 设置图例项文本与图例标识符之间的间距为 0.2
    'legend.columnspacing': 0.8,  # 设置图例中各列之间的间距为 0.8
    'figure.dpi': 1000,  # 设置图像的分辨率为 1000 DPI（高分辨率）
    # 'figure.autolayout': True,  # 自动布局，这里被注释掉，可能是根据需要打开
    'legend.facecolor': 'white',  # 设置图例的背景颜色为白色
    'lines.linewidth': 1.5,  # 设置线条的宽度为 1.5
    'errorbar.capsize': 3,  # 设置误差条帽的长度为 3（帽子的“长条”部分）
    'lines.markeredgewidth': 0.7,  # 设置线条标记（如圆点）的边框宽度为 0.7
    'lines.markersize': 3,  # 设置线条标记的大小为 3
    # 'text.usetex' : True  # 是否启用 LaTeX 渲染，这里被注释掉
}

# 更新 matplotlib 的全局设置，使得所有图表都采用上述定义的样式
plt.rcParams.update(default_settings)



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


class Legend(object):
    """Represents a legend object

    empty_positions: Empty spaces in legend, positions where no legend item is placed
    order: Reorder legend according to a list of indices, controlling the order of the legend items
    axis: The axis to add the legend to, typically a `matplotlib` axis object
    markers: List of plot objects, representing the elements to include in the legend
    legend_labels: The labels to use for each item in the legend
    location: 'top' or default matplotlib values, specifies the location of the legend
    """

    # 常量，表示图例的类型
    STACKS: str = 'STACKS'  # 堆叠图
    BARS: str = 'BARS'  # 条形图
    TOP: str = 'top'  # 图例位置，表示顶部位置

    # 设置图例目标类型，默认为堆叠图
    target: str = STACKS

    # 图例位置，默认为 None，可以在实例化时设置为 'top' 或其他位置
    location: str = None

    # 存储与图例相关的标记对象（如绘图元素）
    markers: list = []

    # 图例的列数，默认为 None，可以在实例化时指定
    ncols: int = None

    # 存储图例的标签
    labels: List[str] = []

    # 图例所属的坐标轴，默认为 matplotlib 的全局坐标轴 plt
    axis = plt

    # 控制图例项的显示顺序，可以是一个整数元组或列表，表示图例项的排列顺序
    order: Union[Tuple[int], List[int], None] = None

    # 存储空白位置，表示图例中的空白项，默认为 None
    empty_positions: Union[List[int], None] = None

    # 自定义标签，默认为 None，允许用户传入自定义的标签
    custom_labels: List[tuple] = None

    def __init__(self, handles, labels: List[str], axis=plt, location: str = None,
                 custom_labels: List[tuple] = None,
                 empty_positions: Union[List[int], None] = None,
                 order: Union[Tuple[int], List[int], None] = None,
                 ncols: int = None):
        """初始化 Legend 对象

        :param handles: 绘图对象的列表（如线条、点等）
        :param labels: 对应每个绘图对象的标签列表
        :param axis: 图例要添加到的坐标轴，默认为全局坐标轴 plt
        :param location: 图例的位置（例如 'top', 'upper left' 等）
        :param custom_labels: 自定义标签列表，每个标签是一个元组 (handle, label)
        :param empty_positions: 图例中空白项的位置
        :param order: 图例项的显示顺序，按列表或元组的顺序排列
        :param ncols: 图例的列数，默认为 None
        """
        self.markers = handles  # 绘图对象
        self.labels = labels  # 图例标签
        self.axis = axis  # 坐标轴
        self.location = location  # 图例位置
        self.custom_labels = custom_labels  # 自定义标签
        self.empty_positions = empty_positions  # 空白位置
        self.order = order  # 显示顺序
        self.ncols = ncols  # 图例列数

    def make(self):
        """Add legend to axis and return the legend object."""

        # 如果有空白位置，将图例中的空白项插入到指定位置
        if self.empty_positions is not None:
            r = patches.Rectangle((0, 0), 1, 1, fill=False,
                                  edgecolor='none',
                                  visible=False)  # 创建一个不可见的矩形对象作为占位符
            for pos in self.empty_positions:
                self.labels.insert(pos, "")  # 在指定位置插入空标签
                self.markers.insert(pos, r)  # 在指定位置插入空图形元素（占位符）

        # 如果有自定义标签，将其添加到图例中
        if self.custom_labels is not None:
            if len(self.labels) == 0:
                # 如果只需要自定义标签而不需要默认的标签
                self.markers = []  # 清空 markers
            for m, t in self.custom_labels:
                self.markers.append(m)  # 将自定义的绘图对象添加到 markers 中
                self.labels.append(t)  # 将自定义的标签添加到 labels 中

        # 根据显示顺序重新排列图例项
        if self.order is not None:
            self.markers = [self.markers[i] for i in self.order]
            self.labels = [self.labels[i] for i in self.order]

        # 设置图例的列数，如果未指定，则根据标签的数量来决定
        if self.ncols is not None:
            columns = self.ncols
        elif len(self.labels) <= 5:
            columns = len(self.labels)
        else:
            columns = np.ceil(len(self.labels) / 2)  # 如果标签数大于 5，分成 2 列

        # 根据位置设置图例的位置
        if self.location == "top":
            # 设置图例位置为顶部居中
            legend = self.axis.legend(self.markers, self.labels,
                                      loc='center', ncol=columns,
                                      bbox_to_anchor=(0.5, 1))  # 图例框位置调整
        elif self.location == 'above':
            # 设置图例位置为图表上方
            legend = self.axis.legend(self.markers, self.labels,
                                      loc='lower center', ncol=columns,
                                      bbox_to_anchor=(0.5, 1))
        else:
            # 使用默认或最佳位置
            legend = self.axis.legend(self.markers, self.labels,
                                      loc=self.location, ncol=columns)

        # 设置图例框的边框线宽
        legend.get_frame().set_linewidth(0.4)

        return legend  # 返回生成的图例对象


def mean_confidence_interval(data: list, confidence: float = 0.99) -> \
        Tuple[float, float]:
    """Compute the mean and the corresponding confidence interval of the given data.

    :param confidence: Confidence interval to use, default: 99%
    :param data: List of numbers to compute mean and interval for
    :return: A tuple containing the mean and the confidence interval (lower, upper)
    """
    # 将数据转换为 NumPy 数组
    a = 1.0 * np.array(data)
    n = len(a)  # 数据的个数
    if n == 1:
        return a[0], 0  # 如果数据只有一个点，返回该点的值和 0 作为置信区间

    # 计算数据的均值和标准误差
    m, se = np.mean(a), scipy.stats.sem(a)

    # 计算置信区间的半宽度（margin of error）
    h: float = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)  # t 分布的临界值

    # 返回均值和置信区间的上下界

    return float(m), h
