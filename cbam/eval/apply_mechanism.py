import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from typing import Dict, List

from cbam.dp.sdd import execute_mechanism, sdd, StuckException, cnoise
from cbam.eval.parser import parse_eval
from cbam.utils import logger, helpers
from cbam.utils.config import Config, get_basename
from cbam.utils.helpers import store, load, trajectories_to_csv, load_trajectory_dict


log = logger.configure_root_loger(
        logging.INFO, Config.get_logdir() + "trajectory_generator.log")
mechanisms = {'SDD': sdd, 'CNOISE': cnoise}


def protect_trajectories(
        unprotected_trajectories: List[pd.DataFrame],
        mechanism: str,
        epsilon: float,
        M: float,
        tmp_file: str,
) -> List[pd.DataFrame]:
    # """
    #     使用指定的保护机制对轨迹数据进行保护。
    #     :param unprotected_trajectories: 未保护的轨迹列表
    #     :param mechanism: 使用的保护机制（如 SDD、CNoise）
    #     :param epsilon: 差分隐私参数 epsilon
    #     :param M: 数据集的敏感度
    #     :param tmp_file: 中间结果保存文件（用于长时间运行的机制如 SDD）
    #     :return: 受保护的轨迹列表
    # """
    # Compute reference point
    log.info("Computing Reference Point...")
    lat0, lon0 = helpers.compute_reference_point(unprotected_trajectories)
    log.info(f"Using reference point ({lat0:.2f}, {lon0:.2f}).")
    todo = [(t, mechanism, lat0, lon0, epsilon, M) for t in unprotected_trajectories]
    protected = []
    try:
        # 是否开启并行化
        if Config.parallelization_enabled():
            n_cpu = cpu_count() if Config.use_all_cpus() else int(cpu_count() * 3 / 4)
            with Pool(n_cpu) as pool:
                for i, res in tqdm(
                        pool.imap_unordered(_generate, todo, chunksize=10),
                        total=len(todo),
                        leave=False,
                        ncols=120):
                    if res is not None:  # Execution failed if None
                        protected.append(res)
                        if mechanism.upper() == 'SDD' and len(protected) % 1000 == 0:
                            store(protected, tmp_file, mute=True)
        else:
            protected = [_generate(t) for t in tqdm(todo, leave=False)]
    except KeyboardInterrupt:
        # Quit gracefully and store current results
        pass
    # 将受保护的轨迹列表返回。
    return protected


def apply_mechanism(
        dataset: str,
        mechanism: str,
        epsilon: float,
        sensitivity: float = 0,
        version: int = 1,
        output_prefix: str = '',
        originals: Dict[str or int, pd.DataFrame] = None,
):
    # """
    #    对指定的数据集应用保护机制。
    #    :param dataset: 数据集名称
    #    :param mechanism: 保护机制
    #    :param epsilon: 差分隐私的 epsilon 值
    #    :param sensitivity: 数据集的敏感度 M
    #    :param version: 输出文件的版本号
    #    :param originals: 如果提供，则直接使用此轨迹数据
    #    :param output_prefix: 输出文件的前缀
    #    :return: None
    #    """
    M = Config.get_M(dataset) if sensitivity == 0 else sensitivity
    log.info(f'Using M = {M}m.')  # See dp.sdd for meaning

    pdir = Config.get_cache_dir(dataset)
    Path(pdir).mkdir(parents=True, exist_ok=True)
    basename = get_basename(mechanism, epsilon, M, version)
    dict_pickle_file = pdir + f"{output_prefix}{basename}_dict.pickle"
    tmpfile = pdir + f"{output_prefix}{basename}.pickle"
    csv_out_file = Config.get_csv_dir(dataset) + f"{output_prefix}{basename}.csv"
    log.info(f"Will be saving to {csv_out_file}.")
    # 缓存我呢见是否存在
    if Config.is_caching() and Path(dict_pickle_file).exists():
        log.warning(f"Continue with: {dict_pickle_file}")
        protected = list(load(dict_pickle_file).values())
        done = [t['trajectory_id'][0] for t in protected]
    else:
        protected, done = [], []
    if originals is None:
        # Load originals
        originals: Dict[str or int, pd.DataFrame] = load_trajectory_dict(dataset=dataset, basename='originals')

    todo = [originals[t] for t in originals if t not in done]

    protected.extend(protect_trajectories(todo, mechanism=mechanism, epsilon=epsilon, M=M, tmp_file=tmpfile))

    dct = helpers.dictify_trajectories(protected)
    if Config.is_caching():
        store(dct, dict_pickle_file)
    trajectories_to_csv(dct, csv_out_file)
    log.info(f"Stored {len(protected)} generated trajectories.")


def _generate(args: tuple):
    """
        调用实际的保护机制。
        :param args: (轨迹, 保护机制, 参考点, epsilon, M)
        :return: 受保护的轨迹
        """

    # """Call the actual protection mechanism."""
    t: pd.DataFrame
    t, mechanism, lat0, lon0, epsilon, M = args
    i = (t['trajectory_id'][0])
    if mechanism == 'SDD':
        error_counter = 0
        while True:
            try:
                tp = execute_mechanism(t, sdd, lat0=lat0, lon0=lon0, kwargs={
                    'epsilon': epsilon,
                    'M': M,
                    'noisy_endpoints': True,
                    'show_progress': False,
                    'enforce_line11': True
                })
                break
            except StuckException:
                error_counter += 1
                if error_counter >= 1000:
                    # This trajectory seems to be unusable.
                    log.warning(
                        f"Execution aborted for trajectory {i} because 1000 tries unsuccessful."
                    )
                    return i, None
                # log.warning("Restarting SDD mechanism because the algorithm is stuck at line 11.")
    else:
        tp = execute_mechanism(
            t, mechanisms[mechanism], lat0=lat0, lon0=lon0, kwargs={'epsilon': epsilon, 'M': M}
        )
    return i, tp


if __name__ == '__main__':
    args = parse_eval().parse_args()
    apply_mechanism(args.dataset, args.mechanism, args.epsilon,
                    sensitivity=args.sensitivity, version=args.version)
