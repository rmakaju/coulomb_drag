"""Module peaks.py

Provides functions to detect and analyze peaks in 2D data sets.
"""

import logging
import numpy as np
from typing import Tuple, Sequence, Callable, Any
from data_load import load_2d_data, get_gate3_lock1, get_gate1_lock1
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


def _extract_peaks(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    min_width: int,
    x_peak_max: float,
    max_peaks: int,
    height: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic peak extraction using scipy.signal.find_peaks.

    Parameters
    ----------
    x_axis : np.ndarray
        Independent variable (e.g., gate voltage).
    y_axis : np.ndarray
        Dependent variable (e.g., lock-in signal).
    min_width : int
        Minimum width of peaks.
    x_peak_max : float
        Maximum x_axis value to consider.
    max_peaks : int
        Maximum number of peaks to return.
    height : float
        Minimum height of peaks.

    Returns
    -------
    Tuple of arrays: x positions, y values, widths, prominences
    """
    peak_idx, properties = find_peaks(y_axis, width=min_width, height=height)
    mask = x_axis[peak_idx] < x_peak_max
    x_vals = x_axis[peak_idx][mask][:max_peaks]
    y_vals = y_axis[peak_idx][mask][:max_peaks]
    widths = properties["widths"][mask][:max_peaks]
    prominences = properties["prominences"][mask][:max_peaks]
    return x_vals, y_vals, widths, prominences


def peaks_hort(
    gate3: np.ndarray,
    lock1: np.ndarray,
    min_width: int = 3,
    x_peak_max: float = -0.5,
    max_peaks: int = 2,
    height: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find peaks in horizontal scans (gate3 vs lock1).

    Returns:
        x positions, y values, widths, and prominences of peaks.
    """
    return _extract_peaks(gate3, lock1, min_width, x_peak_max, max_peaks, height)


def peaks_vert(
    gate1: np.ndarray,
    lock1: np.ndarray,
    min_width: int = 3,
    x_peak_max: float = -0.5,
    max_peaks: int = 2,
    height: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find peaks in vertical scans (gate1 vs lock1).

    Returns:
        x positions, y values, widths, and prominences of peaks.
    """
    return _extract_peaks(gate1, lock1, min_width, x_peak_max, max_peaks, height)

def _find_peak_evolution(
    file_list: Sequence[str],
    gate_loader: Callable[[Any], Tuple[np.ndarray, np.ndarray]],
    peak_func: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    gaussian_smoothing: bool,
    sigma: float,
    min_width: int,
    height: float,
    x_peak_max: float,
    max_peaks: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic peak evolution analysis over multiple data files.

    Parameters
    ----------
    file_list : Sequence[str]
        Paths to data files.
    gate_loader : Callable
        Function to extract (gate, lock1) arrays from file_data.
    peak_func : Callable
        Peak extraction function (e.g., peaks_hort or peaks_vert).
    gaussian_smoothing : bool
        Apply Gaussian filter to lock1 data.
    sigma : float
        Standard deviation for Gaussian smoothing.
    min_width : int
        Minimum width for peak detection.
    height : float
        Minimum peak height.
    x_peak_max : float
        Maximum x value to include peaks.
    max_peaks : int
        Maximum number of peaks per file.

    Returns
    -------
    Tuple of arrays: x positions, y values, widths, prominences, number of peaks
    """
    num_files = len(file_list)
    x_array = np.full((num_files, max_peaks), np.nan)
    y_array = np.full((num_files, max_peaks), np.nan)
    width_array = np.zeros((num_files, max_peaks))
    prominence_array = np.zeros((num_files, max_peaks))
    num_peaks_array = np.zeros(num_files, dtype=int)
    valid = 0
    for idx, filename in enumerate(file_list):
        try:
            data_file = load_2d_data(filename)
            gate_vals, lock1 = gate_loader(data_file)
            series = gaussian_filter(lock1, sigma) if gaussian_smoothing else lock1
            x_list, y_list, widths, prominences = peak_func(
                gate_vals, series,
                min_width=min_width,
                height=height,
                x_peak_max=x_peak_max,
                max_peaks=max_peaks
            )
            num_peaks = len(x_list)
            # output number of peaks found for this file
            print(f"{num_peaks}", end="")
            logger.debug(f"{filename}: found {num_peaks} peaks")
            num_peaks_array[idx] = num_peaks
            assert num_peaks <= max_peaks
            x_array[idx, :num_peaks] = x_list
            y_array[idx, :num_peaks] = y_list
            width_array[idx, :num_peaks] = widths
            prominence_array[idx, :num_peaks] = prominences
            valid += 1
        except Exception:
            logger.exception(f"Error processing {filename}")

    print("\nValid files #:", valid)
    logger.info("Done. Valid files: %d", valid)
    return x_array, y_array, width_array, prominence_array, num_peaks_array


def find_peak_evol_hort(
    file_list: Sequence[str],
    fixedGate1Val: float = -1.6,
    gaussian_smoothing: bool = True,
    sigma: float = 1.0,
    min_width: int = 3,
    height: float = 0.0,
    x_peak_max: float = -0.5,
    max_peaks: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute peak evolution across horizontal sweeps.

    Returns arrays of x positions, y values, widths, prominences, and peak counts.
    """
    return _find_peak_evolution(
        file_list,
        # use correct parameter name `fixed_gate1` per data_load.get_gate3_lock1 signature
        lambda d: get_gate3_lock1(file_data=d, fixed_gate1=fixedGate1Val),
        peaks_hort,
        gaussian_smoothing,
        sigma,
        min_width,
        height,
        x_peak_max,
        max_peaks
    )


def find_peak_evol_vert(
    file_list: Sequence[str],
    fixedGate3Val: float = -1.6,
    gaussian_smoothing: bool = False,
    sigma: float = 1.0,
    min_width: int = 3,
    height: float = 0.0,
    x_peak_max: float = -0.5,
    max_peaks: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute peak evolution across vertical sweeps.

    Returns arrays of x positions, y values, widths, prominences, and peak counts.
    """
    return _find_peak_evolution(
        file_list,
        # use correct parameter name `fixed_gate3` per data_load.get_gate1_lock1 signature
        lambda d: get_gate1_lock1(file_data=d, fixed_gate3=fixedGate3Val),
        peaks_vert,
        gaussian_smoothing,
        sigma,
        min_width,
        height,
        x_peak_max,
        max_peaks
    )