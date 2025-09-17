import logging

import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit
from shapely.geometry import Polygon

from cbam.utils.helpers import get_latlon_matrix
from cbam.preprocessing.coordinates import is_polar_coord_pd

log = logging.getLogger()


def euclidean_distance_pd(traj1: pd.DataFrame, traj2: pd.DataFrame,
                          use_haversine=True, disable_checks: bool = False) -> float:

    if not disable_checks:
        check_haversine_usable_pd(traj1, traj2, use_haversine)

        # 获取经纬度矩阵
    t2 = get_latlon_matrix(traj2)
    t1 = get_latlon_matrix(traj1)

    # 检查轨迹长度是否相等
    if len(t2) != len(t1):
        if 'trajectory_id' in traj1.columns and 'trajectory_id' in traj2.columns:
            # 确保列存在再访问
            logging.warning(
                f"Using asynchronous Euclidean distance! "
                f"Trajectory ID 1: {traj1['trajectory_id'].iloc[0]}; Length: {len(traj1)}' "
                f"Trajectory ID 2: {traj2['trajectory_id'].iloc[0]}; Length: {len(traj2)}'"
            )
        else:
            logging.warning("Using asynchronous Euclidean distance, but cannot check IDs.")

        return _async_euclidean_distance(t1, t2, use_haversine)

    # 正常计算欧几里得距离
    return euclidean_distance(t1, t2, use_haversine)


def euclidean_distance(t1: np.ndarray, t2: np.ndarray, use_haversine=True):
    """Compute the euclidean distance between two trajectories."""
    n = len(t1)
    if use_haversine:
        d = haversine_vector(t1, t2, unit=Unit.METERS)
    else:
        d = np.linalg.norm((t2 - t1), axis=1)
    res = 1 / n * np.sum(d)
    return res


def _async_euclidean_distance(t1: np.ndarray, t2: np.ndarray, use_haversine=True) -> float:
    if len(t1) > len(t2):
        t1, t2 = t2, t1
    n = len(t1)
    m = len(t2)
    if m == n:
        raise ValueError("The synchronous definition should have been used.")
    results = []
    for j in range(0, m - n + 1):
        tmp = t2[j:j + n, ]
        if use_haversine:
            d = haversine_vector(t1, tmp, unit=Unit.METERS)
        else:
            d = np.linalg.norm((tmp - t1), axis=1)
        results.append(np.sum(d))
    assert len(results) == m - n + 1
    return 1 / n * min(results)


def hausdorff_distance_pd(t1: pd.DataFrame, t2: pd.DataFrame,
                          use_haversine: bool = True, disable_checks: bool = False) -> float:
    if not disable_checks:
        check_haversine_usable_pd(t1, t2, use_haversine)
    t1 = get_latlon_matrix(t1)
    t2 = get_latlon_matrix(t2)
    return hausdorff_distance(t1, t2, use_haversine)


def hausdorff_distance(t1: np.ndarray, t2: np.ndarray, use_haversine: bool = True) -> float:
    return max([
        _hausdorff_distance_directed(t1, t2, use_haversine=use_haversine),
        _hausdorff_distance_directed(t2, t1, use_haversine=use_haversine),
    ])


def _hausdorff_distance_directed(t1: np.ndarray, t2: np.ndarray, use_haversine: bool = True) -> float:
    results = []
    for v in t1:
        tmp = np.broadcast_to(v, (len(t2), len(v)))
        if use_haversine:
            distances = haversine_vector(t2, tmp, unit=Unit.METERS)
        else:
            distances = np.linalg.norm((t2 - tmp), axis=1)
        results.append(min(distances))
    return max(results)


def check_haversine_usable_pd(t1: np.ndarray, t2: np.ndarray, use_haversine: bool = True) -> None:
    if use_haversine:
        # Check that coordinates are longitude and latitude
        if not is_polar_coord_pd(t1) or not is_polar_coord_pd(t2):
            raise ValueError(
                "Haversine Formula only works with longitude and latitude."
            )
    else:
        if is_polar_coord_pd(t1) or is_polar_coord_pd(t2):
            logging.warning(
                "It looks like you are not using haversine distance although dealing with polar coordinates."
            )


def jaccard_index(t1: np.ndarray, t2: np.ndarray) -> float:
    p1, p2 = Polygon(t1).convex_hull, Polygon(t2).convex_hull
    intersection = p1.intersection(p2)
    union = p1.union(p2)
    
    # 检查 union.area 是否为零，避免除零错误
    if union.area == 0:
        return 0  # 或者返回 np.nan，根据需求来设定

    jaccard_index = intersection.area / union.area
    return jaccard_index

def jaccard_index_pd(t1: pd.DataFrame, t2: pd.DataFrame) -> float:
    return jaccard_index(get_latlon_matrix(t1), get_latlon_matrix(t2))