import argparse
import copy
import logging
import traceback
import multiprocessing as mp
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Iterable, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold

from cbam.ml.tensorflow_preamble import TensorflowConfig
from cbam.preprocessing.metrics import euclidean_distance_pd, hausdorff_distance_pd, jaccard_index_pd


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='RAoPT Evaluation.')
    parser.add_argument('-c', '--case', metavar='CASE', type=str,
                        help='Execute specific case', default=None)
    parser.add_argument('-g', '--gpu', help="GPU to use", type=int, default=None)
    return parser


CASE = None
if __name__ == '__main__':
    from cbam.utils import logger
    log = logger.configure_root_loger(
        logging.INFO, None)
    args = get_parser().parse_args()
    TensorflowConfig.configure_tensorflow(args.gpu)
    CASE = args.case
else:
    log = logging.getLogger()

from cbam.utils.config import Config, get_basename
n_kfold = int(Config.get('DEFAULT', 'KFOLD'))
kfold_random_state = 7
learning_rate = Config.get_learning_rate()
EPOCHS = Config.get_epochs()
batch_size = Config.get_batch_size()

def get_cases_file() -> str:
    return Config.get_basedir() + 'config/cases.csv'


def read_cases(filename: str) -> pd.DataFrame:
    df = pd.read_csv(
        filename,
        delimiter=',',
        dtype={
            'ID': str,
            'Epsilon Train': float,
            'Epsilon Test': float,
            'M Train': float,
            'M Test': float,
            'Done': bool
        }
    )
    return df


def get_cases() -> List[dict]:
    """Get all evaluation cases represented as dicts."""
    filename = get_cases_file()
    df = read_cases(filename)
    return df.to_dict('records')


def mark_case_complete(case_id: str, mark_as=True, filename=get_cases_file()) -> None:
    """Mark an evaluation case as done within the case file."""
    cases = read_cases(filename)
    cases.loc[cases['ID'] == str(case_id), 'Done'] = mark_as
    cases.to_csv(filename, index=False)


#  过滤掉经度和纬度超出范围的点
def validate_coordinates(data):
    # 过滤掉经度和纬度超出范围的点
    data = data[(data['longitude'] >= -180) & (data['longitude'] <= 180)]
    data = data[(data['latitude'] >= -90) & (data['latitude'] <= 90)]
    return data


def compute_distances(val: (pd.DataFrame, pd.DataFrame, pd.DataFrame, int)) -> dict:
    (o, r, p, fold) = val
    # 验证数据坐标是否在范围内
    o = validate_coordinates(o)
    r = validate_coordinates(r)
    p = validate_coordinates(p)

    res = \
        {
            'Fold': fold,
            'Euclidean Original - Protected':
                euclidean_distance_pd(o, p, use_haversine=True, disable_checks=True),
            'Euclidean Original - Reconstructed':
                euclidean_distance_pd(o, r, use_haversine=True, disable_checks=True),
            'Hausdorff Original - Protected':
                hausdorff_distance_pd(o, p, use_haversine=True, disable_checks=True),
            'Hausdorff Original - Reconstructed':
                hausdorff_distance_pd(o, r, use_haversine=True, disable_checks=True),
            'Jaccard Original - Protected':
                jaccard_index_pd(o, p),
            'Jaccard Original - Reconstructed':
                jaccard_index_pd(o, r),
        }
    return res



def parallelized_distance_computation(
        test_orig: dict, reconstructed: dict, test_p: dict, fold: int = 0) -> List[dict]:
    start = timer()
    parallel_input = [
        (test_orig[id],
         reconstructed[id],
         test_p[id],
         fold)
        for id in reconstructed
    ]
    if Config.parallelization_enabled():
        with mp.Pool(mp.cpu_count()) as pool:
            fold_results = [
                x for x in tqdm(pool.imap(compute_distances, parallel_input, chunksize=10),
                                desc='Computing distances',
                                total=len(parallel_input))]
    else:
        fold_results = [
            compute_distances(x) for x in tqdm(parallel_input, desc='Computing distances', total=len(parallel_input))]
    log.info(f"Completed distance computation in {round(timer() - start)}s.")
    return fold_results


def compute_decrease_percent(before: float or np.ndarray, after: float or np.ndarray) -> float or np.ndarray:
    return 100 * (before - after) / abs(before)


def compute_increase_percent(before: float or np.ndarray, after: float or np.ndarray) -> float or np.ndarray:
    return 100 * (after - before) / abs(before)


def comp_results(df: pd.DataFrame) -> (float, float, float, float):
    ep = df['Euclidean Original - Protected']
    er = df['Euclidean Original - Reconstructed']
    hp = df['Hausdorff Original - Protected']
    hr = df['Hausdorff Original - Reconstructed']
    e_imp = (compute_decrease_percent(ep, er)).mean()
    h_imp = (compute_decrease_percent(hp, hr)).mean()
    jp = df['Jaccard Original - Protected'].mean()
    jr = df['Jaccard Original - Reconstructed'].mean()
    return e_imp, h_imp, jp, jr


def print_results_detailed(df: pd.DataFrame) -> None:
    e_imp, h_imp, jp, jr = comp_results(df)
    print(f"Average Euclidean Distance protected\t\t<-->\toriginal:\t{df['Euclidean Original - Protected'].mean()}")
    print(
        f"Average Euclidean Distance reconstructed\t<-->\toriginal:\t{df['Euclidean Original - Reconstructed'].mean()}")
    print(f"Improvement by {round(e_imp, 1)}%.")
    print(f"Average Hausdorff Distance protected\t\t<-->\toriginal:\t{df['Hausdorff Original - Protected'].mean()}")
    print(
        f"Average Hausdorff Distance reconstructed\t<-->\toriginal:\t{df['Hausdorff Original - Reconstructed'].mean()}")
    print(f"Improvement by {round(h_imp, 1)}%.")
    print(f"Average Jaccard Distance protected\t\t<-->\toriginal:\t{jp}")
    print(f"Average Jaccard Distance reconstructed\t\t<-->\toriginal:\t{jr}")
    j_imp = compute_increase_percent(jp, jr) if jp != 0.0 else np.inf
    print(f"Improvement by {round(j_imp, 1)}%.")


def print_all_results(output_dir: str, res_file: str = None) -> None:
    filename = get_cases_file()
    cases = read_cases(filename)
    all_cases = cases['ID']

    for cid in all_cases:
        filename = output_dir + f"case{cid}/results.csv"
        try:
            df = pd.read_csv(filename)
            e_imp, h_imp, jp, jr = comp_results(df)
            if 'Jaccard Original - Protected' in df:
                j_imp = compute_increase_percent(jp, jr) if jp != 0.0 else np.inf
            else:
                j_imp, jp, jr = "N/A", "N/A", "N/A"

            cases.loc[cases['ID'] == str(cid), 'Euclid'] = f"{e_imp}%"
            cases.loc[cases['ID'] == str(cid), 'Hausdorff'] = f"{h_imp}%"
            cases.loc[cases['ID'] == str(cid), 'Jaccard Before'] = f"{jp}"
            cases.loc[cases['ID'] == str(cid), 'Jaccard After'] = f"{jr}"

            print(f"\033[31m Case {cid} Results: \033[0m\t"
                  f"Improvement by ({round(e_imp, 1)}%;"
                  f"\t{round(h_imp, 1)}%;"
                  f"\t{round(j_imp, 1)}%).")
        except FileNotFoundError:
            print(f"\033[31m Case {cid} not found.\033[0m")

    if filename is not None:
        cases.to_csv(res_file, index=False)


def store_metadata(odir: str, case: dict) -> None:
    metafile = odir + 'metadata.txt'
    with open(metafile, 'w') as f:
        f.write(f"# Case {case['ID']}\n")
        for key in case:
            f.write(f"{key} = {case[key]}\n")
        f.write(f"Number of Splits = {n_kfold}\n")
        f.write(f"Seed for KFold/Split = {kfold_random_state}\n")
    log.info(f"Wrote Metadate to {metafile}.")


def determine_max_length(case: dict) -> int:
    return max([Config.get_max_len(case['Dataset Train']), Config.get_max_len(case['Dataset Test'])])


def run_case(case: dict) -> bool:
    from cbam.utils import helpers
    from cbam.ml.model import AttackModel
    import tensorflow as tf
    cid = case['ID']
    print(f"\033[33m Running Case {cid}.\033[0m")

    same_dataset = True if case['Dataset Train'] == case['Dataset Test'] else False

    odir = Config.get_output_dir() + f"case{cid}/"
    Path(odir).mkdir(parents=True, exist_ok=True)

    log.info("Loading Training/Testing Data")
    train_basename = get_basename(case['Protection Train'], case['Epsilon Train'], case['M Train'], 1)
    test_basename = get_basename(case['Protection Test'], case['Epsilon Test'], case['M Test'], 1)
    try:
        train_originals = helpers.load_trajectory_dict(dataset=case['Dataset Train'], basename='originals')
        train_protected = helpers.load_trajectory_dict(dataset=case['Dataset Train'], basename=train_basename)
        test_originals = copy.deepcopy(train_originals) if same_dataset else helpers.load_trajectory_dict(
            dataset=case['Dataset Test'], basename='originals')
        test_protected = copy.deepcopy(train_protected) if same_dataset and train_basename == test_basename \
            else helpers.load_trajectory_dict(dataset=case['Dataset Test'], basename=test_basename)
    except FileNotFoundError as e:
        log.error(f"Aborting case {cid} because file {e.filename} not found.")
        raise RuntimeError(f"Case {cid}: {e.filename} not found.")

    splits: Iterable[(list, list)]
    if same_dataset:
        log.info(f"Using {n_kfold}-Fold Cross Validation.")
        keys = sorted(train_protected.keys())
        kf = KFold(n_splits=n_kfold, shuffle=True, random_state=kfold_random_state)
        splits = [([keys[i] for i in train_idx], [keys[i] for i in test_idx]) for train_idx, test_idx in kf.split(keys)]
    else:
        splits: List = [(list(train_protected.keys()), list(test_protected.keys()))]
        splits = splits * n_kfold
    store_metadata(odir, case)
    max_length = determine_max_length(case)
    from cbam.ml.encoder import get_encoded_trajectory_dict
    train_originals_encoded = get_encoded_trajectory_dict(
        dataset=case['Dataset Train'], basename="originals", trajectory_dict=train_originals)
    train_protected_encoded = get_encoded_trajectory_dict(
        dataset=case['Dataset Train'], basename=train_basename, trajectory_dict=train_protected)
    fold, results = 0, []
    for train_idx, test_idx in splits:
        tf.keras.backend.clear_session()
        fold += 1
        log.info(f"Processing round {fold}/{len(splits)}")
        parameter_file = odir + f'parameters_fold_{fold}.hdf5'


        lstm = AttackModel(
            max_length=max_length,
            scale_factor=scale_factor,
            learning_rate=learning_rate,
            reference_point=(lat0, lon0),
            parameter_file=parameter_file
        )


        if Config.continue_evaluation() and Path(parameter_file).exists():
            log.warning(f"Loading existing parameters from {parameter_file}.")
            lstm.model.load_weights(parameter_file)
        epochs = EPOCHS


        fold_completion_file = odir + f'fold_{fold}_complete.txt'
        if not Config.continue_evaluation() or (not Path(fold_completion_file).exists() or
                                                helpers.load(fold_completion_file) < epochs):
            trainX = [train_protected_encoded[key]for key in train_idx]
            trainY = [train_originals_encoded[key] for key in train_idx]
            log.info("Start Training")
            history = lstm.train(trainX, trainY, epochs=epochs, batch_size=batch_size,
                                 use_val_loss=True, tensorboard=Config.use_tensorboard())
            n_epochs = len(history.history['loss'])
            log.info(f"Training complete after {n_epochs} epochs.")
            helpers.store(epochs, fold_completion_file)

        log.info(f"Prediction on {len(test_idx)} indices.")
        reconstructed = lstm.predict([test_protected[i] for i in test_idx if i in test_protected])
        reconstructed: Dict[str or int, pd.DataFrame] = {str(p['trajectory_id'][0]): p for p in reconstructed}
        fold_results = parallelized_distance_computation(test_orig=test_originals, reconstructed=reconstructed,
                                                         test_p=test_protected, fold=fold)
        result_file = odir + f'fold_results_{fold}.csv'
        df = pd.DataFrame(fold_results)
        df.to_csv(result_file, index=False)
        log.info(f"Wrote Fold {fold} results to: {result_file}")
        print_results_detailed(df)
        results.extend(fold_results)

    result_file = odir + f'results.csv'
    df = pd.DataFrame(results)
    df.to_csv(result_file, index=False)
    log.info(f"Wrote final results to: {result_file}")
    print_results_detailed(df)
    return True


def run_cases():
    cases = get_cases()
    for case in cases:
        if not case['Todo']:
            log.info(f"Skipping case {case['ID']} because Todo = {case['Todo']}.")
            continue
        case_logfile_handler = logger.add_filehandler(Config.get_logdir() + f"case_{case['ID']}.log", log)
        try:
            success = run_case(case)
            if success:
                mark_case_complete(case['ID'])
        except Exception as e:
            err_file = Config.get_logdir() + f"case_{case['ID']}.err"
            log.error(f"Case {case['ID']}: Aborted due to the following error: {str(e)}")
            with open(err_file, 'a') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
            log.error(f"Full stack trace written to {err_file}.")
        finally:
            # Remove case specific log file
            log.removeHandler(case_logfile_handler)
    log.info("All test cases completed. Terminating.")


if __name__ == '__main__':
    if CASE is None:
        run_cases()
    else:
        cases = get_cases()
        for case in cases:
            if case['ID'] == CASE:
                run_case(case)
