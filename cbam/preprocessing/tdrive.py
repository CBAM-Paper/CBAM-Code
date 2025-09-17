import logging
import pickle
import random
from datetime import timedelta
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from typing import List

import multiprocessing as mp
from tqdm import tqdm

from cbam.preprocessing.preprocess import drop_out_of_bounds, drop_duplicate_points, \
    drop_speed_outliers, verify_trajectory, split_based_on_timediff, load_cache
from cbam.utils import logger
from cbam.utils.config import Config

from cbam.utils.helpers import kmh_to_ms, store, trajectories_to_csv, find_bbox

CHUNKSIZE = 1
pdir = Config.get_cache_dir('tdrive')
Path(pdir).mkdir(exist_ok=True, parents=True)
csv_dir = Config.get_csv_dir('tdrive')
_tdrive_cache_filename = pdir + "tdrive_cleaned.pickle"
_hua2015_tmpfile = pdir + "hua2015.pickle"
_ma2021_tmpfile = pdir + "ma2021.pickle"
_tdrive_trajectories = pdir + "originals.pickle"
_tdrive_trajectories_dict = pdir + "originals_dict.pickle"
INTERVAL = float(Config.get('TDRIVE', 'INTERVAL'))  # seconds
MIN_LEN = int(Config.get('TDRIVE', 'MIN_LENGTH'))
MAX_LEN = int(Config.get('TDRIVE', 'MAX_LENGTH'))
SPEED = float(Config.get('TDRIVE', 'OUTLIER_SPEED'))
MAX_DIST = kmh_to_ms(SPEED) * INTERVAL  # in Meter
if __name__ == '__main__':
    log = logger.configure_root_loger(
        logging.DEBUG, Config.get_logdir() + "tdrive.log")
else:
    log = logging.getLogger()


def _read_tdrive_file(filename: str) -> pd.DataFrame:
    _tdrive_datatypes = OrderedDict({
        "id": int,
        "timestamp": str,
        "longitude": float,
        "latitude": float
    })
    taxi = pd.read_csv(
        filename,
        delimiter=',',
        header=None,
        names=list(_tdrive_datatypes.keys()),
        dtype=_tdrive_datatypes,
        parse_dates=['timestamp']
    )
    return taxi


def _drop_speed_outliers_tdrive(df: pd.DataFrame) -> int:
    max_speed = float(Config.get('TDRIVE', 'OUTLIER_SPEED'))
    return drop_speed_outliers(df, max_speed=max_speed)


def _read_tdrive_files(disable_cleaning=False) -> List[pd.DataFrame]:

    directory = Config.get_dataset_dir('tdrive')
    files = (directory + file for file in Config.get_filenames_tdrive())

    with mp.Pool(mp.cpu_count()) as pool:
        taxis: List[pd.DataFrame] = [
            r for r in tqdm(
                pool.imap(_read_tdrive_file, files, chunksize=CHUNKSIZE),
                total=10357,
                desc='Reading Files'
            )
        ]

        if not disable_cleaning:
            bbox = find_bbox(taxis, quantile=0.99)
            log.info(f"Using bounding box: {np.around(bbox, 2)}")
            args = ((taxi, *bbox) for taxi in taxis)
            taxis = [r for r in tqdm(
                pool.imap(drop_out_of_bounds, args, chunksize=CHUNKSIZE),
                total=len(taxis),
                desc='Removing out-of-bounds'
            )]
            taxis = [t for t in tqdm(
                pool.imap(drop_duplicate_points, taxis, chunksize=CHUNKSIZE),
                total=len(taxis),
                desc='Drop Duplicates'
            )]
            taxis = [r for r in tqdm(
                pool.imap(_drop_speed_outliers_tdrive, taxis, chunksize=CHUNKSIZE),
                total=len(taxis),
                desc='Removing speed outliers'
            )]

            taxis = [t for t in taxis if len(t) >= MIN_LEN]

    log.info(f"Generated {len(taxis)} taxi DataFrames.")

    if Config.is_caching():
        store(taxis, _tdrive_cache_filename)
    return taxis


def get_tdrive_data() -> List[pd.DataFrame]:
    trajs = load_cache(_tdrive_cache_filename)
    if trajs is None:
        trajs = _read_tdrive_files()
    return trajs


def _verify_tdrive_trajectory(df: pd.DataFrame) -> bool:
    return verify_trajectory(df, INTERVAL, MAX_DIST, MIN_LEN, MAX_LEN)


def _verify_tdrive_trajectories(dfs: List[pd.DataFrame]) -> bool:
    with mp.Pool(mp.cpu_count()) as pool:
        for _ in tqdm(
                pool.imap_unordered(_verify_tdrive_trajectory, dfs, chunksize=CHUNKSIZE),
                total=len(dfs),
                desc='Verification'
        ):
            pass
    return True


def _split_based_on_timediff_tdrive(df: pd.DataFrame) -> List[pd.DataFrame]:
    return split_based_on_timediff(df, INTERVAL)


def _generate_tdrive_trajs() -> List[pd.DataFrame]:
    trajs: List[pd.DataFrame] = get_tdrive_data()

    log.info("Generating Trajectories:")
    with mp.Pool(mp.cpu_count()) as pool:
        result = []
        for r in tqdm(pool.imap(_split_based_on_timediff_tdrive, trajs, chunksize=CHUNKSIZE),
                      total=len(trajs),
                      leave=True,
                      desc='Splitting Trajectories'):
            result.extend(r)

        result = [r.reset_index(drop=True)
                  for r in tqdm(result, desc='Length Boundaries')
                  if MAX_LEN >= len(r) >= MIN_LEN]

    result_dct = {}
    for i, r in enumerate(result):
        r.rename(columns={"id": "uid"}, inplace=True)
        r.insert(loc=0, column='trajectory_id', value=i)
        result_dct[str(i)] = r

    _verify_tdrive_trajectories(result)

    log.info(f"Generated {len(result)} T-Drive trajectories.")

    if Config.is_caching():
        store(result_dct, _tdrive_trajectories_dict)
    trajectories_to_csv(result, csv_dir + 'originals.csv')

    assert len(result) != len(trajs)

    return result


def get_tdrive_trajs() -> List[pd.DataFrame]:
    trajs = load_cache(_tdrive_trajectories)
    if trajs is None:
        trajs = _generate_tdrive_trajs()
    return trajs


def _generate_ma2021_trajs() -> List[pd.DataFrame]:
    start_time = "2008-02-04 06:00"
    stop_time = "2008-02-04 07:00"
    trajs = []
    dataset = get_tdrive_data()
    for df in dataset:
        df = df.loc[(start_time <= df['timestamp'])
                    & (stop_time > df['timestamp'])]
        if len(df) >= 10:
            trajs.append(df)
    log.info(f"Generated {len(trajs)} MA2021 trajectories.")

    pickle.dump(trajs, open(_ma2021_tmpfile, "wb"))

    return trajs


def get_ma2021_trajs() -> List[pd.DataFrame]:
    sample_size = 1000
    trajs = load_cache(_ma2021_tmpfile)
    if trajs is None:
        trajs = _generate_ma2021_trajs()
    trajs = random.sample(trajs, sample_size)
    return trajs


def _generate_hua2015_trajs(day: str = '2008-02-04') -> List[pd.DataFrame]:
    one_day = True
    if one_day:
        start_time = f'{day} 8:30'
        stop_time = f'{day} 14:30'
    else:
        start_time = '8:30'
        stop_time = '14:30'
    interval = timedelta(minutes=10)
    nodes_per_traj = 32
    trajs = []
    dataset: List[pd.DataFrame] = get_tdrive_data()

    if one_day:
        dataset = [
            df.loc[(start_time <= df['timestamp'])
                   & (stop_time > df['timestamp'])]
            for df in dataset
        ]
    else:
        dataset = [
            df.iloc[pd.DatetimeIndex(df['timestamp']).indexer_between_time(
                start_time, stop_time)]
            for df in dataset
        ]

    if not one_day:
        result = []
        for df in dataset:
            result.extend(split_based_on_timediff(df, 60 * 60))
        dataset = result

    dataset = [df.reset_index(drop=True)
               for df in dataset if len(df) >= nodes_per_traj]

    removed = 0
    for df in tqdm(dataset):

        if df.at[len(df) - 1, 'timestamp'] - df.at[0, 'timestamp'] < timedelta(minutes=310):
            continue

        ind = 0
        st = df.at[0, 'timestamp']
        indices = [0]
        while len(indices) < 32:
            cur_time = st + interval * len(indices)
            candidate = ind + 1
            if len(df) == candidate + 1:
                indices.append(candidate)
                break
            while (candidate < len(df) - 1) and (
                    abs(df.at[candidate, "timestamp"] - cur_time) > abs(df.at[candidate + 1, "timestamp"] - cur_time)):
                candidate += 1
            if abs(df.at[candidate, "timestamp"] - cur_time) > 2 * interval:
                break
            ind = candidate
            indices.append(ind)

        if len(indices) < 32:
            removed += 1
            continue

        df = df.iloc[indices]

        df.reset_index(drop=True, inplace=True)
        if df.at[31, "timestamp"] - df.at[0, 'timestamp'] < timedelta(minutes=300):
            removed += 1
            continue
        trajs.append(df)
    log.info(f"Removed: {removed}")
    log.info(f"Generated # Trajectories: {len(trajs)}")

    for t in trajs:
        assert nodes_per_traj == len(t)
    pickle.dump(trajs, open(_hua2015_tmpfile, 'wb'))

    return trajs


def get_hua2015_trajs() -> List[pd.DataFrame]:
    trajs = load_cache(_hua2015_tmpfile)
    if trajs is None:
        trajs = _generate_hua2015_trajs()
    return trajs


def get_li2017_trajs() -> List[pd.DataFrame]:
    sample_size = 850
    trajs = random.sample(get_hua2015_trajs(), sample_size)
    return trajs


def get_single_tdrive_db() -> pd.DataFrame:
    trajs = pd.concat(get_tdrive_data())
    return trajs


if __name__ == '__main__':
    _generate_tdrive_trajs()
