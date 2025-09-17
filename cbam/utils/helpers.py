import logging
import pickle
import shutil
from pathlib import Path
from typing import List, Any, Dict, Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from timeit import default_timer as timer

from cbam.utils.config import Config

log = logging.getLogger()


def kmh_to_ms(v: float) -> float:
    return v * 1000 / 3600


def ms_to_kmh(v: float) -> float:
    return v / 1000 * 3600


def remove_cache() -> None:
    temp_dir = Config.get_temp_dir()
    for file in Path(temp_dir).glob("*.pickle"):
        log.info(f"Removing {file}.")
        file.unlink()


def get_latlon_arrays(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    return df['latitude'].to_numpy(), df['longitude'].to_numpy()


def get_latlon_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = ['latitude', 'longitude']
    return df[cols].to_numpy()


def set_latlon(df: pd.DataFrame, lat: np.ndarray, lon: np.ndarray,
               lat_label: str = 'latitude', lon_label: str = 'longitude') -> pd.DataFrame:
    df[lon_label] = lon
    df[lat_label] = lat
    return df


def plot_progress(history, filename: str = None) -> None:
    metrics = history.history.keys()
    plt.title('Training Metrics')
    for m in metrics:
        plt.plot(history.history[m], label=m)
    plt.ylim(0, 1)
    plt.legend()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def compute_reference_point(df: pd.DataFrame or Iterable[pd.DataFrame],
                            lat_label: str = 'latitude',
                            lon_label: str = 'longitude') -> (float, float):
    start_time = timer()
    if isinstance(df, Iterable) and not isinstance(df, pd.DataFrame):
        df = pd.concat(df)
    lat0 = df[lat_label].sum() / len(df)
    lon0 = df[lon_label].sum() / len(df)
    log.debug(f"Computed reference point in {timer() - start_time:.2f}s.")
    return lat0, lon0


def find_bbox(
        trajs: List[pd.DataFrame],
        quantile: float = 1,
        x_label: str = 'longitude',
        y_label: str = 'latitude'
) -> (float, float, float, float):
    single_db = pd.concat(trajs)
    upper_quantiles = single_db.quantile(q=quantile)
    lower_quantiles = single_db.quantile(q=(1 - quantile))
    return lower_quantiles[x_label], upper_quantiles[x_label], \
           lower_quantiles[y_label], upper_quantiles[y_label]


def compute_scaling_factor(trajectories: Iterable[pd.DataFrame], lat0: float, lon0: float) -> (float, float):
    start_time = timer()
    df = pd.concat(trajectories)
    scale_lat = max(abs(df['latitude'].max() - lat0),
                    abs(df['latitude'].min() - lat0),
                    )
    scale_lon = max(abs(df['longitude'].max() - lon0),
                    abs(df['longitude'].min() - lon0),
                    )
    log.debug(f"Computed scaling factor in {timer() - start_time:.2f}s.")
    return scale_lat, scale_lon


def clear_tensorboard_logs() -> None:
    tdir = Config.get_tensorboard_dir()
    for file in Path(tdir).glob('*'):
        print(f"Removing {file}.")
        if file.is_file():
            file.unlink(missing_ok=True)
        else:
            shutil.rmtree(file.absolute())


def store(obj: object, filename: str, mute: bool = False) -> None:
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    if not mute:
        log.info(f"Wrote data to file {filename}")


def load(filename: str, mute: bool = False) -> Any:
    if not mute:
        log.info(f"Loading data from {filename}")
    with open(filename, 'rb') as f:
        res = pickle.load(f)
    return res


def dictify_trajectories(lst: List[pd.DataFrame], tid_label: str = 'trajectory_id') -> Dict[int or str, pd.DataFrame]:
    dct = {}
    for t in tqdm(lst, leave=False, desc='Dictify'):
        dct[str(t[tid_label].iloc[0])] = t
    return dct


def load_cached_trajectory_dict(base_filename: str, tid_label: str = 'trajectory_id') -> Dict[int or str, pd.DataFrame]:
    protected_file = base_filename + ".pickle"
    protected_dict_file = base_filename + "_dict.pickle"
    if not Path(protected_dict_file).exists():
        protected_list = load(protected_file)
        protected_dict = dictify_trajectories(protected_list, tid_label=tid_label)
        if Config.is_caching():
            store(protected_dict, protected_dict_file)
    else:
        protected_dict = load(protected_dict_file)
    return protected_dict


def load_trajectory_dict(dataset: str,
                         basename: str,
                         tid_label: str = 'trajectory_id') -> Dict[int or str, pd.DataFrame]:
    if Config.is_caching():
        try:
            return load_cached_trajectory_dict(
                base_filename=Config.get_cache_dir(dataset=dataset) + basename, tid_label=tid_label)
        except FileNotFoundError as e:
            log.warning(f"No cached file exist ({e.filename}). Loading CSV instead.")
    return read_trajectories_from_csv(
        filename=Config.get_csv_dir(dataset=dataset) + basename + '.csv', tid_label=tid_label
    )


def split_set_into_xy(lst: List[tuple]):
    X = np.array([x for x, _ in lst], dtype='object')
    Y = np.array([y for _, y in lst], dtype='object')
    return X, Y


def read_trajectories_from_csv(filename: str,
                               latitude_label: str = 'latitude',
                               longitude_label: str = 'longitude',
                               tid_label: str = 'trajectory_id',
                               user_label: str = 'uid',
                               tid_type: str = 'str',
                               user_type: str = 'int32',
                               date_columns: bool or list = ['timestamp'],
                               as_dict: bool = True
                               ) -> Dict[str or int, pd.DataFrame]:
    log.info(f"Reading Trajectories from {filename}.")
    df = pd.read_csv(
        filename,
        # parse_dates=date_columns,
        dtype={tid_label: tid_type, user_label: user_type}
    )
    if type(date_columns) is list and date_columns[0] in df:
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
    conv = {latitude_label: 'latitude', longitude_label: 'longitude', tid_label: 'trajectory_id', user_label: 'uid'}
    df.rename(columns=conv, inplace=True)
    trajectories: Dict[str or int, pd.DataFrame] = {key: t.reset_index(drop=True) for key,
                                                                                      t in df.groupby('trajectory_id')}
    if as_dict:
        return trajectories
    else:
        return list(trajectories.values())


def trajectories_to_csv(trajectories: List[pd.DataFrame] or dict, filename: str):
    if type(trajectories) is dict:
        key = int if type(next(iter(trajectories.keys()))) is int or next(iter(trajectories.keys())) else None
        trajectories = [trajectories[key] for key in sorted(trajectories.keys(), key=key)]
    Path(filename).parent.mkdir(exist_ok=True, parents=True)
    pd.concat(trajectories).to_csv(filename, index=False)
    log.info(f"Wrote Trajectories to CSV file {filename}.")


def append_trajectory(t: pd.DataFrame or List[pd.DataFrame], filename: str):
    if type(t) is list:
        t = pd.concat(t)
    t.to_csv(filename, mode='a', index=False, header=False)
