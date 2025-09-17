import logging
import multiprocessing as mp
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from cbam.utils.config import Config
from cbam.utils.helpers import store, load, load_trajectory_dict

log = logging.getLogger()

# 时间戳编码
def encode_timestamp(df: pd.DataFrame, label: str) -> (np.ndarray, np.ndarray):
    """
    Encode a timestamp into hour-of-day and day-of-week.
    :param df: Trajectory 轨迹数据
    :param label: Timestamp column's label 时间戳列的标签
    :return: (hour-of-day, day-of-week) both as one-hot encodings
    （小时编码，星期几编码）
    """
    hour = df[label].dt.hour.to_numpy()
    dow = df[label].dt.dayofweek
    try:
        # 转为one-hot编码
        hour_encoded = tf.keras.utils.to_categorical(hour, num_classes=24)
        dow_encoded = tf.keras.utils.to_categorical(dow, num_classes=7)
    except IndexError as e:
        log.error(
            f"There is an error in the time information of trajectory {df['trajectory_id']}."
        )
        raise e
    return hour_encoded, dow_encoded

# 编码轨迹
def encode_trajectory(t: pd.DataFrame,
                      ignore_time: bool = False,
                      categorical_features: list = list(),
                      vocab_size: dict = dict(),
                      numerical_features: list = list()
                      ) -> np.ndarray:
    # """
    #    将轨迹数据编码为矩阵形式。
    #    每一行代表一个停靠点，每一列代表一个特征。
    #    :param t: 轨迹数据
    #    :param ignore_time: 是否忽略时间信息
    #    :param categorical_features: 要编码的分类特征列表
    #    :param vocab_size: 每个分类特征的编码字典
    #    :param numerical_features: 要编码的数值特征列表
    #    :return: 返回编码后的轨迹数据矩阵
    # """
    # 删除所有的NaN值
    t.dropna(inplace=True)
    t.reset_index(inplace=True, drop=True)
    lat = t['latitude'].to_numpy().reshape((-1, 1))
    lon = t['longitude'].to_numpy().reshape((-1, 1))
    parts = [lat, lon]
    #  # 如果不忽略时间且时间戳存在，则编码时间
    if not ignore_time and 'timestamp' in t:
        parts.extend(encode_timestamp(t, 'timestamp'))
    # Categorical attributes
    #编码分类特征
    for f in categorical_features:
        parts.append(
            tf.keras.utils.to_categorical(t[f], num_classes=vocab_size[f])
        )
    #  # 添加数值特征
    for f in numerical_features:
        parts.append(t[f].to_numpy().reshape((-1, 1)))
    res = np.concatenate(parts, axis=1)
    return res

# 轨迹解码 该函数将编码后的轨迹数据解码回原始的 DataFrame 格式。
def decode_trajectory(t: np.ndarray, ignore_time: bool = False) -> pd.DataFrame:
    # """
    #     解码由 encode_trajectory 编码的轨迹。
    #     :param t: 编码后的轨迹
    #     :param ignore_time: 是否忽略时间信息
    #     :return: 解码后的轨迹数据
    # """

    d = {}
    # 是否忽略
    if ignore_time:
        # 如果 t 包含超过 2 列，则提取纬度列（第 0 列）和经度列（第 1 列），并忽略剩余的列。
        if len(t) > 2:
            lat, lon, _ = np.split(t, [1, 2], axis=-1)
        # 如果 t 只有 2 列或更少，则直接将第 0 列作为纬度，第 1 列作为经度。
        else:
            lat, lon = np.split(t, [1], axis=-1)
    else:
        lat, lon, hour_encoded, dow_encoded = np.split(t, [1, 2, 26], axis=-1)
        hour = np.argmax(hour_encoded, axis=-1)
        dow = np.argmax(dow_encoded, axis=-1)
        d['hour'] = hour
        d['dow'] = dow
    lat = lat.flatten()
    d['latitude'] = lat
    lon = lon.flatten()
    d['longitude'] = lon

    res = pd.DataFrame(d)
    return res

# 减去参考点
def subtract_reference_point(t: np.ndarray, lat0: float, lon0: float) -> pd.DataFrame:
    # """
    #    从纬度和经度中减去参考点
    #    :param t: 轨迹数据
    #    :param lat0: 参考点的纬度
    #    :param lon0: 参考点的经度
    #    :return: 修改后的轨迹
    # """

    t['latitude'] -= lat0
    t['longitude'] -= lon0
    return t

# 添加参考点
def add_reference_point(t: np.ndarray, lat0: float, lon0: float) -> pd.DataFrame:
    t['latitude'] += lat0
    t['longitude'] += lon0
    return t


def _encode_wrapper(args):
    return encode_trajectory(*args)

# 轨迹字典编码
def encode_trajectory_dict(trajectory_dict: Dict[str or int, pd.DataFrame],
                           ignore_time: bool = False) -> Dict[str or int, np.ndarray]:
    # """
    #    从包含 pandas.DataFrame 的轨迹字典创建编码后的轨迹字典
    #    :param trajectory_dict: {轨迹ID: pd.DataFrame}
    #    :param ignore_time: 是否忽略时间信息
    #    :return: 编码后的轨迹字典 {轨迹ID: np.ndarray}
    # """

    log.info("Encoding trajectories...")
    start = timer()
    if Config.parallelization_enabled():
        keys = list(trajectory_dict.keys())
        generator = ((trajectory_dict[key], ignore_time) for key in keys)
        with mp.Pool(mp.cpu_count()) as pool:
            encoded_dict = dict(zip(keys, pool.map(_encode_wrapper, generator)))
    else:
        encoded_dict = {
            key: encode_trajectory(trajectory_dict[key], ignore_time=ignore_time) for key in tqdm(
                trajectory_dict, desc='Encoding', total=len(trajectory_dict))
        }
    log.info(f"Encoded trajectories in {round(timer() - start)}s.")
    return encoded_dict


def get_encoded_trajectory_dict(dataset: str, basename: str, encoded_file: str = None, ignore_time: bool = False,
                                trajectory_dict:  Dict[str or int, pd.DataFrame] = None) -> dict:
    # """
    #     加载已编码的轨迹数据，如果缓存文件存在则从缓存中加载；否则对轨迹数据进行编码。
    #     :param dataset: 数据集名称
    #     :param basename: 文件名（如 "original" 或使用的保护机制名）
    #     :param encoded_file: 用于存储编码后的轨迹的文件路径
    #     :param ignore_time: 是否忽略时间信息，仅编码纬度和经度
    #     :param trajectory_dict: 一个字典，包含轨迹数据（如果提供了此参数，就不需要从文件加载轨迹数据）
    #     :return: 编码后的轨迹字典
    #  """


    encoded_file = Config.get_cache_dir(dataset=dataset) + basename + "_encoded_dict.pickle" if \
        encoded_file is None else encoded_file
    if Config.is_caching() and Path(encoded_file).exists():
        encoded_dict = load(encoded_file)
    else:
        if trajectory_dict is None:
            trajectory_dict = load_trajectory_dict(dataset=dataset, basename=basename)
        encoded_dict = encode_trajectory_dict(trajectory_dict, ignore_time=ignore_time)
        if Config.is_caching():
            store(encoded_dict, encoded_file)
    return encoded_dict

# SemanticEncoder 是一个类，用于对具有语义信息（如分类特征和数值特征）的轨迹进行编码。它在普通轨迹编码的基础上添加了对分类特征的处理。
class SemanticEncoder:

    def __init__(self, categorical_features: List[str], numerical_features: List[str]):
        self.vocabulary = {}
        self.encoders = {}
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

    def transform_categorical(
            self,
            trajectories: dict,
    ) -> dict:
        df = pd.concat(list(trajectories.values()))
        for f in self.categorical_features:
            if f not in self.encoders:
                # Fit encoder
                self.encoders[f] = LabelEncoder().fit(df[f])
            df[f] = self.encoders[f].transform(df[f])
        trajectories = {key: t.reset_index(drop=True) for key, t in df.groupby('trajectory_id')}
        return trajectories

    def get_vocab_sizes(self):
        return {f: len(self.encoders[f].classes_) for f in self.categorical_features}

    def encode_semantic(self, trajectories: dict) -> np.ndarray:
        trajectories = self.transform_categorical(trajectories)
        encoded = {k: encode_trajectory(trajectories[k],
                                        categorical_features=self.categorical_features,
                                        vocab_size=self.get_vocab_sizes(),
                                        numerical_features=self.numerical_features
                                        ) for k in trajectories}
        return encoded



