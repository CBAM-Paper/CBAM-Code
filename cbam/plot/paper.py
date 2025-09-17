import logging
from pathlib import Path
from typing import List, Iterable, Dict

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
import os
import shutil
import matplotlib  # 导入整个 matplotlib 模块
import matplotlib.font_manager as fm  # 新增：用于中文字体支持
# 现在才导入 pyplot
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter

# --- 1. 彻底清除matplotlib缓存 (再次强调，务必执行) ---
print("--- Attempting to clear Matplotlib font cache ---")
cache_dir = None
try:
    cache_dir = fm.get_cachedir()  # 尝试使用新版本方法
except AttributeError:
    try:
        cache_dir = matplotlib.get_cachedir()  # 尝试使用旧版本方法
    except AttributeError:
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'matplotlib')  # 手动猜测常见路径

if cache_dir and os.path.exists(cache_dir):
    try:
        shutil.rmtree(cache_dir)  # 彻底删除整个缓存目录
        print(f"Deleted entire matplotlib cache directory: {cache_dir}")
    except OSError as e:
        print(f"Error deleting cache directory {cache_dir}: {e}")
else:
    print(f"Matplotlib cache directory not found or already cleared at: {cache_dir}")
print("--- Matplotlib font cache cleared ---")

# 强制matplotlib重新加载字体管理器
try:
    fm._rebuild()
    print("Matplotlib font manager rebuilt.")
except AttributeError:
    print("Warning: fm._rebuild() not found or failed, relying on cache clear and restart.")

# --- 2. 设置 Matplotlib 后端 (对于云服务器很重要) ---
# 在任何绘图操作之前设置非交互式后端
matplotlib.use('Agg')
print("Matplotlib backend set to 'Agg'.")

# --- 3. 配置中文字体 (在任何 plt.rcParams.update() 之前) ---
# 确保 'font.family' 在 'font.sans-serif' 之前设置，并使用你 fc-list 中确认存在的字体
plt.rcParams['font.family'] = 'sans-serif'  # 确保使用 sans-serif 字体家族
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei', 'Songti SC',
                                   'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
from matplotlib.ticker import MultipleLocator  # 新增：用于统一横坐标刻度差值

from cbam.eval.main import comp_results
from cbam.plot import plot
from cbam.plot.plot import cm
from cbam.utils.config import Config
from matplotlib.ticker import MultipleLocator, MaxNLocator  # 保留 MultipleLocator，尽管X轴不用，但Y轴可能用

log = logging.getLogger()
COLUMN_WIDTH = plot.IEEE_WIDTH
DOUBLE_COLUMN_WIDTH = 506.295 * 0.03514 * 1.1
# DOUBLE_COLUMN_WIDTH = 506.295 * 0.03514 * 1.0
result_file = Config.get_output_dir() + "case{}/results.csv"  # Needs format with case ID
plot_dir = "plots/"
Path(plot_dir).mkdir(parents=True, exist_ok=True)
ALL_COLUMNS = ['ID', 'Dataset Train', 'Dataset Test', 'Protection Train', 'Protection Test',
               'Epsilon Train', 'Epsilon Test',
               ]

# --- 新增/修改：中文字体配置 ---
# 建议选择您系统上已有的中文字体，例如 'SimHei' (黑体), 'Microsoft YaHei' (微软雅黑), 'Songti SC' (宋体-简) 等。
# 如果运行时中文字符显示为方块，请检查您的系统是否安装了这些字体。
# 您可以通过以下代码查找系统中的字体（在Python环境中运行）：
# import matplotlib.font_manager as fm; print(fm.findSystemFonts(fontpaths=None, fontext='ttf'))
# 如果需要使用特定字体文件，可以先将其添加到字体管理器：
# font_path = '/path/to/your/SimHei.ttf' # 替换为实际字体文件路径
# fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'sans-serif'  # 添加这一行，确保使用 sans-serif 字体家族
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei', 'Songti SC',
                                   'PingFang SC', 'sans-serif']  # 设置默认字体，并提供备用字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

font_size = 8
settings = {
    'font.size': font_size,
    'legend.fontsize': font_size - 2,
    'axes.titlesize': font_size - 2,
    'axes.labelsize': font_size - 2,
    'ytick.labelsize': font_size - 2,
    'xtick.labelsize': font_size - 2,
    # 'lines.markersize': 3, # 此行可以删除，或保留但会被 example_trajectories 中的具体设置覆盖
    'figure.autolayout': True
}
plt.rcParams.update(settings)


def modify_strings(cases: List[dict]) -> List[dict]:
    replacements = {
        'TDRIVE': 'T-Drive',
        'CNOISE': 'CNoise',
        'GEOLIFE': 'GeoLife'
    }
    for case in cases:
        for col in case:
            for old in replacements:
                if type(case[col]) is str and old in case[col]:
                    case[col] = case[col].replace(old, replacements[old])
    return cases


from matplotlib.ticker import MaxNLocator


def example_trajectories(originals: Dict[str, pd.DataFrame],
                         protected: Dict[str, pd.DataFrame],
                         reconstructed: Dict[str, pd.DataFrame],
                         tids: List[str],
                         n_rows: int = 1,
                         protection_mechanism='SDD 0.1',
                         filename: bool = None):
    markers = ['s', 'o', '^']  # 's' 代表正方形，'o' 代表圆形，'^' 代表三角形
    colors = ['#3498db', '#f39c12', '#e74c3c']  # 颜色列表
    n_cols = int(len(tids) / n_rows)
    # 增大
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(cm(DOUBLE_COLUMN_WIDTH), cm(3)))
    # fig, axs = plt.subplots(n_rows, n_cols, figsize=(cm(DOUBLE_COLUMN_WIDTH*1.8), cm(4.8)))
    if n_rows == 1:
        axs = np.array([axs])  # 确保axs总是二维的，便于统一处理
    chinese_protection_mechanism = protection_mechanism.replace('CNoise 10', 'CNoise机制 ε=10.0')
    labels = ['原始轨迹', chinese_protection_mechanism, '重构轨迹']
    lines = list()
    for row in range(n_rows):
        for col in range(n_cols):
            current_ax = axs[row][col]  # 明确定义current_ax
            tid = tids[row * n_cols + col]
            ts = [originals[tid], protected[tid], reconstructed[tid]]

            for i, t in enumerate(ts):
                line, = current_ax.plot(t.longitude, t.latitude,
                                        marker=markers[i], linestyle='-', color=colors[i],
                                        linewidth=0.5, label=labels[i] if i == 0 else '')  # 添加标记
                lines.append(line)

            # 计算当前子图的经度范围
            all_longitudes = []
            for t in ts:
                all_longitudes.extend(t.longitude.values)
            min_lon = min(all_longitudes)
            max_lon = max(all_longitudes)

            # 生成5个等间隔的刻度
            ticks = np.linspace(min_lon, max_lon, 5)

            # 设置横坐标刻度
            current_ax.xaxis.set_major_locator(FixedLocator(ticks))

            # 设置刻度格式为保留两位小数
            current_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            current_ax.set_xlabel("经度")
            if col == 0:  # 只在最左侧的列设置纬度标签
                current_ax.set_ylabel("纬度")
            current_ax.set_title(f"轨迹ID: {tid}", x=0.5, y=0.91)

    plot.Legend(lines[:len(labels)], labels, axis=fig, location='top', ncols=3).make()
    plt.subplots_adjust(wspace=-1.3, hspace=None)
    # --- 修改保存图片的格式 ---
    if filename is not None:
        # 修改文件扩展名为 .pdf 或 .svg
        if filename.endswith('.png'):
            # 默认是 .png，可以根据需要改成 .pdf 或 .svg
            # filename_pdf = filename.replace('.png', '.pdf') # 生成 PDF 版本
            filename_vector = filename.replace('.png', '.svg')  # 生成 SVG 版本

            # plt.savefig(filename_pdf, bbox_inches='tight', dpi=600) # 保存为 PDF
            # print(f"Stored to {filename_pdf}.")

            plt.savefig(filename_vector, bbox_inches='tight', dpi=600)  # 保存为 SVG
            print(f"Stored to {filename_vector}.")

            # 如果你只想保存一种格式，选择其中一个即可。
            # 如果想同时保存多种，就保留多个 savefig 调用。
        else:
            # 如果 filename 本身就是其他格式，直接保存
            plt.savefig(filename, bbox_inches='tight', dpi=600)
            print(f"Stored to {filename}.")



def plot_line(axis: plt.Axes, x: Iterable[float], data: Iterable[float], label: str = None, fmt: str = None):
    """
    Plot a line with error-bars at each point.
    :param axis: The Axes object to plot onto
    :param x: The x coordinates len(x) == len(data)
    :param data: List of y values to compute the mean and confidence interval from
    :param label: Line label for legend
    :param fmt: Line format/style
    :return:
    """
    y, error = [], []
    for d in data:
        tmp_y, tmp_e = plot.mean_confidence_interval(d)
        y.append(tmp_y)
        error.append(tmp_e)
    return axis.errorbar(
        x,
        y,
        fmt=fmt,
        yerr=error,
        label=label,
    )


def plot_lines(axis: plt.Axes, x: Iterable[float], data: Iterable[Iterable[float]], labels: List[str],
               title: str = None):
    """
    Plot multiple lines with y-errorbars.
    :param axis: The axis object to print ontoz
    :param x: The x coordinates len(x) == len(data[i]) for all i in range(len(data))
    :param data: List of lists, one list for each line
                 I.e., each element is a list of values. data = [[], [], []]
                 len(x) == len(data[i]) for all i in range(len(data))
    :param labels: Labels for each line. len(labels) == len(data)
    :param title: Title of the plot
    :return:
    """
    fmts = ['-x', '-o', '-D', '-s']
    pad = 2
    lines = []
    for i, y in enumerate(data):
        lines.append(plot_line(axis, x, y, label=labels[i], fmt=fmts[i]))
    axis.set_title(title, pad=pad)
    axis.set_xticks(x)
    return lines


def return_results(idx: List[str]) -> (List[Iterable[float]], List[Iterable[float]]):
    """
    Return the results for the given cases
    :param idx: List of case IDs
    :return: (List[Euclidean Improvements], List[Hausdorff Imp.])
    """
    cases = []
    for cid in idx:
        try:
            cases.append(pd.read_csv(result_file.format(cid)))
        except FileNotFoundError as e:
            log.warning(f"[SKIPPED] Case(s) missing: {e.filename}")

    data_euclid = []
    data_hausdorff = []
    for case in cases:
        ep = case['Euclidean Original - Protected']
        er = case['Euclidean Original - Reconstructed']
        hp = case['Hausdorff Original - Protected']
        hr = case['Hausdorff Original - Reconstructed']
        improvement_euclid = 100. * (ep - er) / abs(ep)
        improvement_hausdorff = 100. * (hp - hr) / abs(hp)
        data_euclid.append(improvement_euclid)
        data_hausdorff.append(improvement_hausdorff)
    return data_euclid, data_hausdorff


def print_all_results() -> None:
    """
    Print a table with all results for cases 1 - 36 for the appendix.
    :return: None
    """
    print_partial_table([str(i) for i in range(1, 37)])


def print_partial_table(case_ids: List[int], columns: List[str] = ALL_COLUMNS, no_results=False) -> None:
    """
    Print a table with the results for the given cases.
    :param case_ids: IDs of the cases to print
    :param columns: Columns of the table
    :param no_results: Only print the case properties
    :return: None
    """
    # Create complete table
    from cbam.eval.main import get_cases
    cases = [case for case in get_cases() if case['ID'] in case_ids]
    cases = modify_strings(cases)

    for case in cases:
        cid = case['ID']
        filename = result_file.format(cid)
        try:
            df = pd.read_csv(filename)
            line = [f"{case[column]}" for column in columns]
            if not no_results:
                e_imp, h_imp, jp, jr = comp_results(df)
                line.extend([
                    f"\SI{{{e_imp:.1f}}}{{\%}}",
                    f"\SI{{{h_imp:.1f}}}{{\%}}",
                    f"\\num{{{jp:1.2e}}}",
                    f"\\num{{{jr:1.2e}}}"
                ])
            line[-1] += '\\\\'
            print(
                ' & '.join(line)
            )
        except FileNotFoundError:
            print(f"\033[31m Case {cid} not found.\033[0m")


# # 可以传递颜色
def plot_bar(axis: plt.Axes, x: Iterable, data: List[List[float]], bar_width=0.5, labels=None, color=None):
    y, error = [], []
    for d in data:
        tmp_y, tmp_e = plot.mean_confidence_interval(d)
        y.append(tmp_y)
        error.append(tmp_e)

    # 如果 color 参数为空，使用默认颜色
    if color is None:
        color = ['blue'] * len(data)  # 默认所有的条形图使用相同的颜色

    # 如果提供了多个颜色，确保颜色的数量与数据的数量匹配
    if isinstance(color, list) and len(color) != len(data):
        raise ValueError(f"Length of color list ({len(color)}) must match the number of data sets ({len(data)})")

    bars = axis.bar(
        x=x,
        height=y,
        width=bar_width,
        yerr=error,
        label=labels,
        align='edge',
        color=color  # 使用 color 参数
    )
    bar_labels = axis.bar_label(bars, fmt='%.1f%%', label_type='center', fontsize=font_size - 2)
    for i in range(len(y)):
        if y[i] < 0:
            bar_labels[i].xy = (bar_labels[i].xy[0], 10)

    return bars



def transfer_figure(title: str,
                    euclid1: Iterable[Iterable[float]], hausdorff1: Iterable[Iterable[float]],
                    euclid2: Iterable[Iterable[float]], hausdorff2: Iterable[Iterable[float]],
                    x_ticks: Iterable[float], x_labels: Iterable, filename: str = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 2, figsize=(cm(COLUMN_WIDTH), cm(3)), sharey=True)
    pad = 2
    labels = ("Euclidean Distance", "Hausdorff Distance")
    x = np.array([0.6, 1.6])
    bar_width = 0.4

    ax[0].set_title(f'T-Drive to GeoLife [{title}]', pad=pad)
    ax[0].set_xticks(x_ticks)  # 设置刻度的位置
    ax[0].set_xticklabels(x_labels)  # 设置刻度的标签
    lines = list()
    lines.append(plot_bar(ax[0], x, euclid1, bar_width, color='blue'))  # 设定euclid1的颜色为蓝色
    lines.append(plot_bar(ax[0], x + bar_width, hausdorff1, bar_width, color='green'))  # 设定hausdorff1的颜色为绿色

    ax[1].set_title(f'GeoLife to T-Drive [{title}]', pad=pad)
    ax[1].set_xticks(x_ticks)  # 设置刻度的位置
    ax[1].set_xticklabels(x_labels)  # 设置刻度的标签
    lines.append(plot_bar(ax[1], x, euclid2, bar_width, color='orange'))  # 设定euclid2的颜色为橙色
    lines.append(plot_bar(ax[1], x + bar_width, hausdorff2, bar_width, color='red'))  # 设定hausdorff2的颜色为红色

    for axis in ax:
        axis.set_xticks(x_ticks)  # 设置刻度的位置
        axis.set_xticklabels(x_labels)  # 设置刻度的标签
        axis.set_xlabel('\u03B5', labelpad=-10)
        axis.set_ylim(0, 100)

    ax[0].set_ylabel('Distance Reduction [%]')
    plot.Legend(lines, labels, axis=fig, location='top', ncols=2).make()
    fig.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Stored to {filename}.")

    return fig


# save_to_file 如果为True 图像被保存到指定路径 如果为False 仅绘制图像而不保存
# train 如果为True 表示重新训练模型 如果为False 仅绘制图像而不保存
def figure_3(save_to_file: bool = False, train: bool = False) -> None:
    """Create Figure 3 from Paper

    :param save_to_file: Store into output directory?
    :param train: Train the model from scratch instead of using existing parameters
    :return: None
    """
    import random

    from cbam.utils import helpers
    from cbam.ml import model

    # 图片保存路径
    filename = plot_dir + 'example-trajs.png' if save_to_file else None
    # 数据加载部分
    originals = helpers.read_trajectories_from_csv(
        "processed_csv/geolife/originals.csv")
    # 这个可以修改 这里是经过SDD保护机制扰动的轨迹数据
    protected = helpers.read_trajectories_from_csv(
        "processed_csv/geolife/cnoise_M16500_e10.0_1.csv")
    # protected的所有轨迹ID 用于随机选择一部分轨迹进行训练和测试
    tid_range = list(protected.keys())
    tid_range_file = Config.get_parameter_path() + 'tid_range_fig3.pickle'

    parameter_file = Config.get_parameter_path() + 'figure3.hdf5'

    # 如果trani为Fasle且模型文件不存在 默认使用parameters_fold_1.hd5
    if not train and not Path(parameter_file).exists():
        print("figure3.hdf5 not found, using parameters_fold_1.hdf5")
        parameter_file = "output/case13/parameters_fold_1.hdf5"
    # 存在则直接加载训练好的模型权重
    elif not train:
        print("figure3.hdf5 exists, loading tid_range from tid_range_file")
        tid_range = helpers.load(tid_range_file)
    # 对轨迹进行归一化处理时的参考点
    # 缩放因子：用于对轨迹坐标进行标准化 便于模型处理
    print(f"Loading parameters from: {parameter_file}")
    all_trajs = list(protected.values())
    lat0, lon0 = helpers.compute_reference_point(all_trajs)
    scale_factor = helpers.compute_scaling_factor(all_trajs, lat0, lon0)
    log.info(f"Reference Point: ({lat0}, {lon0})")
    log.info(f"Scaling factor: {scale_factor}")

    # 实例化模型
    raopt = model.AttackModel(
        reference_point=(lat0, lon0),
        scale_factor=scale_factor,
        max_length=200,
    )
    # 如果要重新训练模型
    if train:
        from cbam.ml import encoder
        keys = list(protected.keys())
        random.shuffle(keys)
        n_test = int(0.2 * len(keys))
        tid_range = keys[:n_test]
        helpers.store(tid_range, tid_range_file)
        train_idx = keys[n_test:]
        # 将protected和originals编码为模型可以处理的格式
        protected_encoded = encoder.encode_trajectory_dict(protected)
        originals_encoded = encoder.encode_trajectory_dict(originals)
        # 将轨迹数据随机划分为训练集train_idx和测试集 tid_range
        trainX = [protected_encoded[key] for key in train_idx]
        trainY = [originals_encoded[key] for key in train_idx]
        log.info("Start Training")
        # 训练模型并保存历史
        history = raopt.train(trainX, trainY, use_val_loss=True)
        log.info(f"Training complete after {len(history.history['loss'])} epochs.")
    else:
        log.info(f"Loading parameters from: {parameter_file}")
        raopt.model.load_weights(parameter_file)

    # tids = random.sample(tid_range, 4)
    # tids = ["6264", "61453", "112508", "31855"]  # 你的指定轨迹
    tids = ["085_20090815062008_5", "062_20080917062507_2", "068_20090608091159_2", "041_20090407223938_3"]

    #    #使用训练好的模型对选定的轨迹ID（tids)进行预测 生成重构轨迹
    #     # 将模型的预测结果转为轨迹的字典格式
    reconstructed = raopt.predict([protected[i] for i in tids])
    reconstructed = helpers.dictify_trajectories(reconstructed)
    example_trajectories(
        originals=originals,
        protected=protected,
        reconstructed=reconstructed,
        tids=tids,
        protection_mechanism='CNoise 10',
        filename=filename
    )


if __name__ == '__main__':
    # # Figure 3
    # # Warning! Requires Model execution
    print("#" * 80)
    print("Figure 3")
    print("#" * 80)
    figure_3(True, True)
    print("#" * 80)
