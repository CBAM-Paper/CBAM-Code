
import logging
from typing import Union, Iterable, List, Callable, Any

import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm import tqdm

from cbam.preprocessing.coordinates import latlon_to_xy, xy_to_latlon, check_coordinate_range
from cbam.utils.helpers import get_latlon_arrays, set_latlon

# -----------------------------CONSTANTS---------------------------------------
SDD_THRESHOLD = 1000
# -----------------------------CONSTANTS---------------------------------------

log = logging.getLogger()


def add_laplace_to_value(v: Union[float, Iterable[float]], mean: float, scale: float) -> Any:
    n = None
    if hasattr(v, "__len__"):
        n = len(v)
    return v + np.random.laplace(mean, scale, n)


def diff_private_laplace(
        v: Union[float, Iterable[float]], epsilon: float, sensitivity: float) -> Union[float, Iterable[float]]:
    return add_laplace_to_value(v, 0, sensitivity / epsilon)


def sphere_sampling(n: int, r: float) -> List[float]:
    x = np.random.normal(0, 1, n)
    w = np.sqrt(np.sum(np.square(
        x
    )))
    z = r / w * x
    return z


def gnoise(
        x: np.ndarray, y: np.ndarray, epsilon: float, delta: float, M: float
) -> (np.ndarray, np.ndarray):
    check_coordinate_range(x, y)
    assert len(x) == len(y)
    n = len(x)
    log.debug(f"GNoise - Using M = {round(M)}m.")

    if delta >= 1:
        log.error("Delta has to be smaller than one!")
        raise RuntimeError("Delta has to be smaller than one!")

    b = (2 * M / epsilon) + (2 * M / epsilon) * \
        (2 * n - 1) * (1 / np.log(1. / (1. - delta)))

    log.debug(f"GNoise: b = {b}")

    # Sample noise
    r = np.random.exponential(b)

    z = sphere_sampling(2 * n, r)
    alpha = [z[i] for i in range(len(z)) if i % 2 == 0]
    beta = [z[i] for i in range(len(z)) if i % 2 == 1]

    assert alpha != beta
    assert len(alpha) == len(beta)
    assert len(alpha) == n

    # Add Noise
    res_x = x + alpha
    res_y = y + beta

    return res_x, res_y


def pnoise(
        x: np.ndarray, y: np.ndarray, epsilon: float, delta: float, M: float
) -> (np.ndarray, np.ndarray):
    check_coordinate_range(x, y)
    assert len(x) == len(y)
    n = len(x)
    log.debug(f"PNoise - Using M = {round(M)}m.")

    if delta >= 1:
        log.error("Delta has to be smaller than one!")
        raise RuntimeError("Delta has to be smaller than  one!")

    b = 2 * M / epsilon + 2 * M / (epsilon * np.log(1 / (1 - delta)))

    log.debug(f"PNoise: b = {b}")

    # Sample noise
    alpha, beta = [], []

    r = np.random.exponential(b, size=n)

    for i in range(n):
        z = sphere_sampling(2, r[i])
        alpha.append(z[0])
        beta.append(z[1])

    assert alpha != beta
    assert len(alpha) == len(beta)
    assert len(alpha) == n

    # Add Noise
    res_x = x + alpha
    res_y = y + beta

    return res_x, res_y


def cnoise(
        x: np.ndarray, y: np.ndarray, epsilon: float, M: float
) -> (np.ndarray, np.ndarray):
    check_coordinate_range(x, y)
    assert len(x) == len(y)
    n = len(x)
    log.debug(f"CNoise - Using M = {round(M)}m.")

    b = 2 * np.sqrt(2) * M / epsilon

    log.debug(f"CNoise: b = {b}")

    # Sample noise
    alpha = np.random.laplace(0, b, n)
    beta = np.random.laplace(0, b, n)

    assert len(alpha) == len(beta)
    assert len(alpha) == n

    # Add Noise
    res_x = x + alpha
    res_y = y + beta

    return res_x, res_y


def _compute_C(a: float, B: float, eps: float) -> float:
    """Compute the C value from Jiang2013"""
    if a < B:
        C_inv = (8 * B / eps) * (
            2 - np.exp(- a * eps / (8 * B)) -
            np.exp(- eps * (B - a) / (8 * B))
        )
    else:
        C_inv = 8 * B / eps * \
            (np.exp(- (a - B) * eps / (8 * B)) - np.exp(- eps * a / (8 * B)))
    return 1.0 / C_inv


def get_exp_pdf(C: float, a: float, B: float, eps: float) -> Callable:
    def pdf(x):
        return C * np.exp(- eps * abs(x - a) / (8 * B))

    return pdf


def exponential_mechanism(a: float, B: float, eps: float) -> st.rv_continuous:
    C = _compute_C(a, B, eps)

    pdf = get_exp_pdf(C, a, B, eps)

    class my_pdf(st.rv_continuous):
        def _pdf(self, x):
            return pdf(x)

    my_cv: st.rv_continuous = my_pdf(a=0, b=B, name='PDF Dist')

    return my_cv


def unit_vector(v: np.array) -> float:
    return v / np.linalg.norm(v)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class StuckException(RuntimeError):
    """Raise if SDD mechanism gets stuck"""


def _sdd_step(prev: np.ndarray, destination: np.ndarray, epsilon: float, M: float,
              n: int, i: int, endpoint: np.ndarray, abort_if_stuck=True) -> np.ndarray:
    v = destination - prev
    r_i = np.linalg.norm(v)
    angle = angle_between(destination, prev)
    roh_gen = exponential_mechanism(r_i, M, epsilon)
    alpha_gen = exponential_mechanism(angle, 2 * np.pi, epsilon)
    trials = 0
    guesses = {}
    while True:
        trials += 1
        roh = roh_gen.rvs(size=1)[0]
        alpha = alpha_gen.rvs(size=1)[0]
        new = prev + np.array((roh * np.cos(alpha), roh * np.sin(alpha)))
        distance = np.linalg.norm(new - endpoint)
        if distance <= (n + 1 - i) * M:
            return new
        else:
            guesses[distance] = new
        if trials > SDD_THRESHOLD:
            # Stuck
            if abort_if_stuck:
                raise StuckException(
                    f"SDD mechanism is stuck at i = {i}. "
                    f"Allowed Distance: {n + 1 - i}M, "
                    f"Current Distance: {round(np.linalg.norm(new - endpoint) / M, 2)}M")
            else:
                # Just use the best value, violates definition of SDD
                value = guesses[min(guesses.keys())]
                dist = np.linalg.norm(value - endpoint) / M
                log.warning(
                    f'Using a value violating line 11 '
                    f'(Distance: {round(dist, 2)}M vs {round(n + 1 - i, 2)}M) for i = {i}')
                return value


def sdd(
        x: np.ndarray,
        y: np.ndarray,
        epsilon: float,
        M: float,
        noisy_endpoints: bool = True,
        show_progress=False,
        enforce_line11=False
) -> (np.ndarray, np.ndarray):
    check_coordinate_range(x, y)
    assert len(x) == len(y)

    n = len(x)
    alg_n = n - 2

    log.debug(f"SDD - Using M = {round(M)}m.")

    abort_if_stuck = enforce_line11


    res_x, res_y = np.zeros(n), np.zeros(n)

    startpoint = np.array((x[0], y[0]))
    res_x[0], res_y[0] = startpoint
    endpoint = np.array((x[-1], y[-1]))
    res_x[-1], res_y[-1] = endpoint
    log.debug(f"Startpoint: {str(startpoint)}, Endpoint: {str(endpoint)}")

    loop = range(1, n - 1)
    if show_progress:
        loop = tqdm(loop, leave=True, ncols=120)

    for i in loop:
        orig = np.array((x[i], y[i]))
        prev = np.array((res_x[i - 1], res_y[i - 1]))
        new = _sdd_step(prev=prev, destination=orig, epsilon=epsilon, M=M, n=alg_n, i=i,
                        endpoint=endpoint, abort_if_stuck=abort_if_stuck)
        res_x[i], res_y[i] = new

    if noisy_endpoints:
        prev = np.array((res_x[-2], res_y[-2]))
        new = _sdd_step(prev, endpoint, epsilon, M=M, n=n - 1, i=n - 1,
                        endpoint=endpoint, abort_if_stuck=abort_if_stuck)
        res_x[-1], res_y[-1] = new

        prev = np.array((res_x[1], res_y[1]))
        new = _sdd_step(prev, startpoint, epsilon, M=M, n=alg_n, i=0,
                        endpoint=endpoint, abort_if_stuck=abort_if_stuck)
        res_x[0], res_y[0] = new

    return res_x, res_y


def execute_mechanism(df: pd.DataFrame, mechanism: Callable, lat0: float, lon0: float,
                      convert: bool = True,
                      args: Iterable = (), kwargs: dict = {}) -> pd.DataFrame:
    df = df.copy()
    lat, lon = get_latlon_arrays(df)
    if convert:
        lat, lon = latlon_to_xy(lat, lon, lat0, lon0)
    lat, lon = mechanism(lat, lon, *args, **kwargs)
    if convert:
        lat, lon = xy_to_latlon(lat, lon, lat0, lon0)
    df = set_latlon(df, lat, lon)
    return df
