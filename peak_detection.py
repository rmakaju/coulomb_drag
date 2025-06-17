"""
Peak Detection Module for Coulomb Drag Data

This module contains core peak detection functions and basic statistical analysis
for symmetric/antisymmetric components in Coulomb drag measurements.

Functions:
- find_peaks_in_data(): Main peak detection function for any data component
- analyze_peak_statistics(): Generate statistical summaries of detected peaks
- compare_peak_counts(): Quick summary of peak detection results
- compare_peak_detection(): Compare detection results between different datasets
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from collections import Counter


def find_peaks_in_data(all_runs_data, data_component='lock1_1', height_threshold=None, 
                       prominence=None, distance=None, width=None, return_details=False):
    """
    Find peaks in specified data component for all current values
    
    Parameters
    ----------
    all_runs_data : dict
        Data from all current measurements
    data_component : str
        Which data component to analyze for peaks.
        Options: 'lock1_1', 'lock1_2', 'symmetric', 'antisymmetric', 
                'smoothed_1', 'smoothed_2', 'symmetric_smoothed', 'antisymmetric_smoothed'
    height_threshold : float, optional
        Minimum height of peaks (in µV), if None will auto-detect
    prominence : float, optional
        Minimum prominence of peaks, if None will auto-detect  
    distance : int, optional
        Minimum distance between peaks (in data points)
    width : int, optional
        Minimum width of peaks (in data points)
    return_details : bool, optional
        Whether to return detailed peak properties
    
    Returns
    -------
    dict
        Dictionary with peak information for each current value
        
    Examples
    --------
    >>> # Find peaks in symmetric smoothed data
    >>> peaks = find_peaks_in_data(data, data_component='symmetric_smoothed')
    >>> 
    >>> # Find peaks in raw symmetric data
    >>> peaks = find_peaks_in_data(data, data_component='symmetric')
    >>>
    >>> # Find peaks with custom parameters
    >>> peaks = find_peaks_in_data(data, data_component='antisymmetric', 
    ...                           height_threshold=5.0, prominence=2.0)
    """
    
    # Validate data component
    valid_components = ['lock1_1', 'lock1_2', 'symmetric', 'antisymmetric', 
                       'smoothed_1', 'smoothed_2', 'symmetric_smoothed', 'antisymmetric_smoothed']
    
    if data_component not in valid_components:
        raise ValueError(f"data_component must be one of: {valid_components}")
    
    all_peaks_data = {}
    
    for i in range(len(all_runs_data)):
        data = all_runs_data[i]
        current_val = data['current']
        gate1 = data['gate1']
        
        # Get the specified data component
        y_data = data[data_component]
        
        # Convert to µV for peak detection
        y_data_uv = y_data / 1e-6
        
        # Auto-detect parameters if not provided
        if height_threshold is None:
            # Use a fraction of the data range as threshold
            data_range = np.max(y_data_uv) - np.min(y_data_uv)
            auto_height = np.min(y_data_uv) + 0.1 * data_range
        else:
            auto_height = height_threshold
            
        if prominence is None:
            # Auto-detect prominence as a fraction of data range
            auto_prominence = 0.05 * (np.max(y_data_uv) - np.min(y_data_uv))
        else:
            auto_prominence = prominence
        
        # Find peaks
        peak_kwargs = {
            'height': auto_height,
            'prominence': auto_prominence
        }
        
        if distance is not None:
            peak_kwargs['distance'] = distance
        if width is not None:
            peak_kwargs['width'] = width
            
        peaks, properties = find_peaks(y_data_uv, **peak_kwargs)
        
        # Get peak positions in gate voltage
        peak_gate_positions = gate1[peaks]
        peak_heights = y_data_uv[peaks]
        
        # Store results
        peak_info = {
            'current': current_val,
            'peak_indices': peaks,
            'peak_gate_positions': peak_gate_positions,
            'peak_heights': peak_heights,
            'num_peaks': len(peaks),
            'data_type': data_component,
            'gate1': gate1,
            'y_data_uv': y_data_uv
        }
        
        if return_details:
            peak_info['properties'] = properties
            
        all_peaks_data[i] = peak_info
    
    return all_peaks_data


def analyze_peak_statistics(peaks_data):
    """
    Analyze statistics of detected peaks across all currents
    
    Parameters
    ----------
    peaks_data : dict
        Output from find_peaks_in_data
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with peak statistics including:
        - current_nA: Drive current value
        - num_peaks: Number of peaks detected
        - min_peak_position, max_peak_position: Range of peak positions
        - peak_range: Voltage range spanned by peaks
        - mean_peak_height, max_peak_height: Peak amplitude statistics
        - peak_positions, peak_heights: Lists of individual peak data
    """
    
    stats_data = []
    
    for i in range(len(peaks_data)):
        peak_info = peaks_data[i]
        
        if len(peak_info['peak_gate_positions']) > 0:
            stats = {
                'current_nA': peak_info['current'],
                'num_peaks': peak_info['num_peaks'],
                'min_peak_position': np.min(peak_info['peak_gate_positions']),
                'max_peak_position': np.max(peak_info['peak_gate_positions']),
                'peak_range': np.max(peak_info['peak_gate_positions']) - np.min(peak_info['peak_gate_positions']),
                'mean_peak_height': np.mean(peak_info['peak_heights']),
                'max_peak_height': np.max(peak_info['peak_heights']),
                'peak_positions': list(peak_info['peak_gate_positions']),
                'peak_heights': list(peak_info['peak_heights'])
            }
        else:
            stats = {
                'current_nA': peak_info['current'],
                'num_peaks': 0,
                'min_peak_position': np.nan,
                'max_peak_position': np.nan,
                'peak_range': np.nan,
                'mean_peak_height': np.nan,
                'max_peak_height': np.nan,
                'peak_positions': [],
                'peak_heights': []
            }
            
        stats_data.append(stats)
    
    return pd.DataFrame(stats_data)


def compare_peak_counts(component_name, peaks_data):
    """
    Quick summary of peak detection results
    
    Parameters
    ----------
    component_name : str
        Name of the component being analyzed (for display)
    peaks_data : dict
        Output from find_peaks_in_data
        
    Examples
    --------
    >>> peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> compare_peak_counts('Symmetric Smoothed', peaks)
    Symmetric Smoothed:
      Total peaks: 25
      Average peaks per current: 2.1
      Peak count distribution: {0: 1, 1: 3, 2: 7, 3: 1}
    """
    total_peaks = sum([data['num_peaks'] for data in peaks_data.values()])
    avg_peaks = total_peaks / len(peaks_data) if len(peaks_data) > 0 else 0
    
    print(f"\n{component_name}:")
    print(f"  Total peaks: {total_peaks}")
    print(f"  Average peaks per current: {avg_peaks:.1f}")
    
    # Show peak distribution
    peak_counts = [data['num_peaks'] for data in peaks_data.values()]
    count_dist = Counter(peak_counts)
    print(f"  Peak count distribution: {dict(count_dist)}")


def compare_peak_detection(peaks_raw, peaks_smoothed):
    """
    Compare peak detection results between raw and smoothed data
    
    Parameters
    ----------
    peaks_raw : dict
        Peak detection results from raw data
    peaks_smoothed : dict
        Peak detection results from smoothed data
        
    Returns
    -------
    pandas.DataFrame
        Comparison dataframe with columns:
        - current_nA: Drive current
        - peaks_raw, peaks_smoothed: Number of peaks in each dataset
        - peak_diff: Difference in peak count (smoothed - raw)
        - raw_positions, smoothed_positions: Lists of peak positions
        - raw_max_height, smoothed_max_height: Maximum peak heights
        
    Examples
    --------
    >>> raw_peaks = find_peaks_in_data(data, 'symmetric')
    >>> smooth_peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> comparison = compare_peak_detection(raw_peaks, smooth_peaks)
    >>> print(comparison[['current_nA', 'peaks_raw', 'peaks_smoothed', 'peak_diff']])
    """
    
    comparison_data = []
    
    for i in range(len(peaks_raw)):
        raw_data = peaks_raw[i]
        smoothed_data = peaks_smoothed[i]
        
        comparison = {
            'current_nA': raw_data['current'],
            'peaks_raw': raw_data['num_peaks'],
            'peaks_smoothed': smoothed_data['num_peaks'],
            'peak_diff': smoothed_data['num_peaks'] - raw_data['num_peaks'],
            'raw_positions': list(raw_data['peak_gate_positions']),
            'smoothed_positions': list(smoothed_data['peak_gate_positions']),
            'raw_max_height': np.max(raw_data['peak_heights']) if len(raw_data['peak_heights']) > 0 else 0,
            'smoothed_max_height': np.max(smoothed_data['peak_heights']) if len(smoothed_data['peak_heights']) > 0 else 0
        }
        
        comparison_data.append(comparison)
    
    return pd.DataFrame(comparison_data)


# Legacy compatibility function
def find_peaks_in_symmetric_data(all_runs_data, use_smoothed=True, height_threshold=None, 
                                prominence=None, distance=None, width=None, return_details=False):
    """
    DEPRECATED: Use find_peaks_in_data() instead.
    
    This function is kept for backward compatibility. 
    Use find_peaks_in_data(data, data_component='symmetric_smoothed') instead.
    """
    import warnings
    warnings.warn(
        "find_peaks_in_symmetric_data() is deprecated. "
        "Use find_peaks_in_data(data, data_component='symmetric_smoothed') instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    data_component = 'symmetric_smoothed' if use_smoothed else 'symmetric'
    return find_peaks_in_data(all_runs_data, data_component=data_component, 
                             height_threshold=height_threshold,
                             prominence=prominence, 
                             distance=distance, 
                             width=width, 
                             return_details=return_details)
