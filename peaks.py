"""
Comprehensive peak detection and analysis module for Coulomb drag experiments.

This module provides functions for both 1D and 2D peak detection, including:
- 1D peak detection in horizontal and vertical scans
- 2D peak detection with maximum/minimum finding
- Peak evolution tracking across multiple files
- Statistical analysis and utilities for peak characterization

The module combines functionality from the original peaks.py and peaks_2d.py files
into a single, well-organized interface.

For additional utilities like interpolation, adaptive smoothing, and visualization,
see the companion peak_utils.py module.
"""

import logging
import numpy as np
import os
from typing import Tuple, Sequence, Callable, Any, Optional, List, Union, Dict
from data_load import load_2d_data, get_gate3_lock1, get_gate1_lock1
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# =============================================================================
# 1D Peak Detection Functions
# =============================================================================

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


def peaks_horizontal(
    gate3: np.ndarray,
    lock1: np.ndarray,
    min_width: int = 3,
    x_peak_max: float = -0.5,
    max_peaks: int = 2,
    height: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find peaks in horizontal scans (gate3 vs lock1).

    Parameters
    ----------
    gate3 : np.ndarray
        Gate3 voltage values (x-axis).
    lock1 : np.ndarray
        Lock-in amplifier signal values (y-axis).
    min_width : int, optional
        Minimum width of peaks (default: 3).
    x_peak_max : float, optional
        Maximum gate3 value to consider (default: -0.5).
    max_peaks : int, optional
        Maximum number of peaks to return (default: 2).
    height : float, optional
        Minimum height of peaks (default: 0.0).

    Returns
    -------
    tuple
        (x_positions, y_values, widths, prominences) - Arrays of peak properties.
    """
    return _extract_peaks(gate3, lock1, min_width, x_peak_max, max_peaks, height)


def peaks_vertical(
    gate1: np.ndarray,
    lock1: np.ndarray,
    min_width: int = 3,
    x_peak_max: float = -0.5,
    max_peaks: int = 2,
    height: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find peaks in vertical scans (gate1 vs lock1).

    Parameters
    ----------
    gate1 : np.ndarray
        Gate1 voltage values (x-axis).
    lock1 : np.ndarray
        Lock-in amplifier signal values (y-axis).
    min_width : int, optional
        Minimum width of peaks (default: 3).
    x_peak_max : float, optional
        Maximum gate1 value to consider (default: -0.5).
    max_peaks : int, optional
        Maximum number of peaks to return (default: 2).
    height : float, optional
        Minimum height of peaks (default: 0.0).

    Returns
    -------
    tuple
        (x_positions, y_values, widths, prominences) - Arrays of peak properties.
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
        Peak extraction function (e.g., peaks_horizontal or peaks_vertical).
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
    tuple
        (x_array, y_array, width_array, prominence_array, num_peaks_array)
        Arrays containing peak properties for all files.
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


def analyze_peak_evolution_horizontal(
    file_list: Sequence[str],
    fixed_gate1_val: float = -1.6,
    gaussian_smoothing: bool = True,
    sigma: float = 1.0,
    min_width: int = 3,
    height: float = 0.0,
    x_peak_max: float = -0.5,
    max_peaks: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute peak evolution across horizontal sweeps.

    Parameters
    ----------
    file_list : Sequence[str]
        List of data file paths to analyze.
    fixed_gate1_val : float, optional
        Fixed gate1 voltage value (default: -1.6).
    gaussian_smoothing : bool, optional
        Apply Gaussian smoothing to data (default: True).
    sigma : float, optional
        Standard deviation for Gaussian filter (default: 1.0).
    min_width : int, optional
        Minimum peak width (default: 3).
    height : float, optional
        Minimum peak height (default: 0.0).
    x_peak_max : float, optional
        Maximum x value for peak detection (default: -0.5).
    max_peaks : int, optional
        Maximum number of peaks per file (default: 2).

    Returns
    -------
    tuple
        Arrays of x positions, y values, widths, prominences, and peak counts.
    """
    return _find_peak_evolution(
        file_list,
        lambda d: get_gate3_lock1(file_data=d, fixed_gate1=fixed_gate1_val),
        peaks_horizontal,
        gaussian_smoothing,
        sigma,
        min_width,
        height,
        x_peak_max,
        max_peaks
    )


def analyze_peak_evolution_vertical(
    file_list: Sequence[str],
    fixed_gate3_val: float = -1.6,
    gaussian_smoothing: bool = False,
    sigma: float = 1.0,
    min_width: int = 3,
    height: float = 0.0,
    x_peak_max: float = -0.5,
    max_peaks: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute peak evolution across vertical sweeps.

    Parameters
    ----------
    file_list : Sequence[str]
        List of data file paths to analyze.
    fixed_gate3_val : float, optional
        Fixed gate3 voltage value (default: -1.6).
    gaussian_smoothing : bool, optional
        Apply Gaussian smoothing to data (default: False).
    sigma : float, optional
        Standard deviation for Gaussian filter (default: 1.0).
    min_width : int, optional
        Minimum peak width (default: 3).
    height : float, optional
        Minimum peak height (default: 0.0).
    x_peak_max : float, optional
        Maximum x value for peak detection (default: -0.5).
    max_peaks : int, optional
        Maximum number of peaks per file (default: 2).

    Returns
    -------
    tuple
        Arrays of x positions, y values, widths, prominences, and peak counts.
    """
    return _find_peak_evolution(
        file_list,
        lambda d: get_gate1_lock1(file_data=d, fixed_gate3=fixed_gate3_val),
        peaks_vertical,
        gaussian_smoothing,
        sigma,
        min_width,
        height,
        x_peak_max,
        max_peaks
    )


# =============================================================================
# 2D Peak Detection Functions
# =============================================================================

def find_max_2d(
    file: Union[int, str], 
    data_path: Optional[str] = None, 
    global_max: bool = True, 
    x_min: float = -1000, 
    x_max: float = 1000, 
    y_min: float = -1000, 
    y_max: float = 1000, 
    find_troughs: bool = True, 
    gaussian_smoothing: bool = False, 
    sigma_x: float = 0.0003, 
    sigma_y: float = 0.001
) -> Tuple[float, Tuple[float, float], Any, np.ndarray]:
    """
    Find the maximum (or minimum) value in a 2D dataset.
    
    Parameters
    ----------
    file : int or str
        File number or filename to analyze.
    data_path : str, optional
        Base path for data files (used when file is int).
    global_max : bool, optional
        If True, find global max. If False, find max within specified window (default: True).
    x_min, x_max, y_min, y_max : float, optional
        Window boundaries for local max search (default: -1000 to 1000).
    find_troughs : bool, optional
        If True, also consider absolute value of negative peaks (default: True).
    gaussian_smoothing : bool, optional
        Apply Gaussian smoothing to data (default: False).
    sigma_x, sigma_y : float, optional
        Standard deviations for Gaussian smoothing (default: 0.0003, 0.001).
        
    Returns
    -------
    tuple
        (max_value, coordinates, file_data, processed_data)
        - max_value: The maximum value found
        - coordinates: (x, y) coordinates of the maximum
        - file_data: Original loaded data object
        - processed_data: Processed data array (after smoothing if applied)
    """
    # Load data file
    if isinstance(file, int):
        path = data_path or r'P:\ResLabs\LarocheLab\physics-svc-laroche\data_robocopy\VA_204\VA204D'
        filename = os.path.join(path, f"VA204D_{file}.dat")
        file_data = load_2d_data(filename)
    elif isinstance(file, str):
        file_data = load_2d_data(file)
    else:
        raise ValueError("Invalid file type. Expected an integer (e.g., 123) or a string (e.g., 'filename.dat').")
        
    data = file_data.z.copy()
    
    # Apply Gaussian smoothing if requested
    if gaussian_smoothing:
        print("Smoothing Data")
        data = gaussian_filter(data, (sigma_x, sigma_y), mode='nearest')
    
    # Convert data to absolute values if looking for troughs
    if find_troughs:
        data = np.abs(data)
    
    if global_max:
        # Find global maximum
        max_idx = np.unravel_index(np.nanargmax(data), data.shape)
        max_value = data[max_idx]
        coordinates = (file_data.x[max_idx], file_data.y[max_idx])
    else:
        # Find maximum within specified window
        x_mask = (file_data.x >= x_min) & (file_data.x <= x_max)
        y_mask = (file_data.y >= y_min) & (file_data.y <= y_max)
        window_mask = x_mask & y_mask
        
        if not np.any(window_mask):
            raise ValueError("No data points found within specified window")
        
        # Apply mask and find max in the windowed region
        windowed_data = np.where(window_mask, data, -np.inf)
        max_idx = np.unravel_index(np.nanargmax(windowed_data), windowed_data.shape)
        max_value = data[max_idx]
        coordinates = (file_data.x[max_idx], file_data.y[max_idx])
    
    return max_value, coordinates, file_data, data


def peak_distance_calculator(
    data_files: List[Any], 
    smoothed_data: List[np.ndarray], 
    x_min: float = -1000, 
    x_max: float = 1000, 
    y_min: float = -1000, 
    y_max: float = 1000,
    epsilon: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate distances between pairs of peaks found in multiple datasets.
    
    Parameters
    ----------
    data_files : List[Any]
        List of loaded data file objects.
    smoothed_data : List[np.ndarray]
        List of corresponding processed data arrays.
    x_min, x_max, y_min, y_max : float, optional
        Window boundaries for peak search (default: -1000 to 1000).
    epsilon : float, optional
        Small value to avoid division by zero (default: 1e-10).
        
    Returns
    -------
    tuple
        (distances, coordinates) - Arrays of distances between peaks and their coordinates.
    """
    if len(data_files) != len(smoothed_data):
        raise ValueError("Number of data files must match number of smoothed data arrays")
    
    distances = []
    coordinates = []
    
    for i, (file_data, data) in enumerate(zip(data_files, smoothed_data)):
        try:
            # Create window mask
            x_mask = (file_data.x >= x_min) & (file_data.x <= x_max)
            y_mask = (file_data.y >= y_min) & (file_data.y <= y_max)
            window_mask = x_mask & y_mask
            
            if not np.any(window_mask):
                print(f"Warning: No data points in window for file {i}")
                continue
                
            # Find peaks in the windowed region
            windowed_data = np.where(window_mask, data, -np.inf)
            
            # Find top 2 peaks
            flat_data = windowed_data.flatten()
            flat_indices = np.argsort(flat_data)[-2:]  # Get indices of top 2 values
            peak_indices = [np.unravel_index(idx, windowed_data.shape) for idx in flat_indices]
            
            if len(peak_indices) >= 2:
                # Calculate coordinates of the two peaks
                coord1 = (file_data.x[peak_indices[0]], file_data.y[peak_indices[0]])
                coord2 = (file_data.x[peak_indices[1]], file_data.y[peak_indices[1]])
                
                # Calculate distance between peaks
                distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
                
                distances.append(distance)
                coordinates.append((coord1, coord2))
            else:
                print(f"Warning: Less than 2 peaks found in file {i}")
                
        except Exception as e:
            print(f"Error processing file {i}: {e}")
            continue
    
    return np.array(distances), coordinates


def analyze_peak_evolution(
    file_numbers: Sequence[int], 
    data_path: Optional[str] = None,
    gaussian_smoothing: bool = True, 
    sigma_x: float = 0.0003, 
    sigma_y: float = 0.001,
    x_min: float = -1000, 
    x_max: float = 1000, 
    y_min: float = -1000, 
    y_max: float = 1000,
    find_troughs: bool = True
) -> Dict[str, Any]:
    """
    Analyze peak evolution across multiple 2D data files.
    
    Parameters
    ----------
    file_numbers : Sequence[int]
        List of file numbers to analyze.
    data_path : str, optional
        Base path for data files.
    gaussian_smoothing : bool, optional
        Apply Gaussian smoothing (default: True).
    sigma_x, sigma_y : float, optional
        Standard deviations for Gaussian smoothing (default: 0.0003, 0.001).
    x_min, x_max, y_min, y_max : float, optional
        Window boundaries for analysis (default: -1000 to 1000).
    find_troughs : bool, optional
        Consider absolute values for trough detection (default: True).
        
    Returns
    -------
    dict
        Dictionary containing peak evolution data including positions, values, and statistics.
    """
    peak_positions = []
    peak_values = []
    data_files = []
    processed_data = []
    
    for file_num in file_numbers:
        try:
            max_val, coords, file_data, proc_data = find_max_2d(
                file_num, data_path, global_max=False,
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                find_troughs=find_troughs,
                gaussian_smoothing=gaussian_smoothing,
                sigma_x=sigma_x, sigma_y=sigma_y
            )
            
            peak_positions.append(coords)
            peak_values.append(max_val)
            data_files.append(file_data)
            processed_data.append(proc_data)
            
        except Exception as e:
            print(f"Error processing file {file_num}: {e}")
            continue
    
    # Calculate distances between consecutive peaks
    distances = []
    if len(peak_positions) > 1:
        for i in range(1, len(peak_positions)):
            dist = np.sqrt(
                (peak_positions[i][0] - peak_positions[i-1][0])**2 +
                (peak_positions[i][1] - peak_positions[i-1][1])**2
            )
            distances.append(dist)
    
    return {
        'file_numbers': list(file_numbers),
        'peak_positions': peak_positions,
        'peak_values': peak_values,
        'distances': distances,
        'data_files': data_files,
        'processed_data': processed_data,
        'statistics': {
            'mean_peak_value': np.mean(peak_values) if peak_values else 0,
            'std_peak_value': np.std(peak_values) if peak_values else 0,
            'mean_distance': np.mean(distances) if distances else 0,
            'std_distance': np.std(distances) if distances else 0
        }
    }


# =============================================================================
# Utility Functions
# =============================================================================

def find_2d_peaks_in_region(
    data: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    num_peaks: int = 5
) -> List[Tuple[float, float, float]]:
    """
    Find multiple peaks within a specified 2D region.
    
    Parameters
    ----------
    data : np.ndarray
        2D data array.
    x_coords, y_coords : np.ndarray
        Coordinate arrays corresponding to data.
    x_min, x_max, y_min, y_max : float
        Region boundaries.
    num_peaks : int, optional
        Number of peaks to find (default: 5).
        
    Returns
    -------
    List[Tuple[float, float, float]]
        List of (x, y, value) tuples for found peaks.
    """
    # Create region mask
    x_mask = (x_coords >= x_min) & (x_coords <= x_max)
    y_mask = (y_coords >= y_min) & (y_coords <= y_max)
    region_mask = x_mask & y_mask
    
    if not np.any(region_mask):
        return []
    
    # Apply mask and find peaks
    masked_data = np.where(region_mask, data, -np.inf)
    flat_data = masked_data.flatten()
    flat_indices = np.argsort(flat_data)[-num_peaks:]
    
    peaks = []
    for idx in reversed(flat_indices):  # Start with highest peak
        if flat_data[idx] == -np.inf:
            continue
        peak_idx = np.unravel_index(idx, masked_data.shape)
        x_val = x_coords[peak_idx]
        y_val = y_coords[peak_idx]
        z_val = data[peak_idx]
        peaks.append((x_val, y_val, z_val))
    
    return peaks


def validate_peak_coordinates(
    peak_coords: Sequence[Tuple[float, float]],
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float]
) -> List[bool]:
    """
    Validate that peak coordinates fall within specified bounds.
    
    Parameters
    ----------
    peak_coords : Sequence[Tuple[float, float]]
        List of (x, y) coordinate tuples.
    x_bounds, y_bounds : Tuple[float, float]
        (min, max) bounds for x and y coordinates.
        
    Returns
    -------
    List[bool]
        List of boolean values indicating validity of each coordinate pair.
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    
    validity = []
    for x, y in peak_coords:
        valid = (x_min <= x <= x_max) and (y_min <= y <= y_max)
        validity.append(valid)
    
    return validity


def get_peak_statistics(
    peak_values: Sequence[float],
    peak_positions: Optional[Sequence[Tuple[float, float]]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a set of peaks.
    
    Parameters
    ----------
    peak_values : Sequence[float]
        Peak intensity values.
    peak_positions : Sequence[Tuple[float, float]], optional
        Peak (x, y) positions for spatial statistics.
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing statistical measures.
    """
    if not peak_values:
        return {}
    
    values = np.array(peak_values)
    stats = {
        'count': len(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'range': np.max(values) - np.min(values)
    }
    
    if peak_positions:
        positions = np.array(peak_positions)
        if positions.size > 0:
            stats.update({
                'x_mean': np.mean(positions[:, 0]),
                'y_mean': np.mean(positions[:, 1]),
                'x_std': np.std(positions[:, 0]),
                'y_std': np.std(positions[:, 1]),
                'spatial_spread': np.sqrt(np.var(positions[:, 0]) + np.var(positions[:, 1]))
            })
    
    return stats


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Maintain backward compatibility with original function names
peaks_hort = peaks_horizontal
peaks_vert = peaks_vertical
find_peak_evol_hort = analyze_peak_evolution_horizontal
find_peak_evol_vert = analyze_peak_evolution_vertical
peak_evol = analyze_peak_evolution
peak_dist_calculator = peak_distance_calculator
find_max_2D = find_max_2d
