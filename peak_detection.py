"""
Peak Detection Module for Coulomb Drag Data

This module contains core peak detection functions and basic statistical analysis
for symmetric/antisymmetric components in Coulomb drag measurements.

Functions:
- find_peaks_in_data(): Main peak detection function for any data component
- find_valleys_in_data(): Main valley detection function for any data component
- find_peaks_and_valleys_in_data(): Combined peak and valley detection
- analyze_peak_statistics(): Generate statistical summaries of detected peaks
- analyze_valley_statistics(): Generate statistical summaries of detected valleys
- compare_peak_counts(): Quick summary of peak detection results
- compare_peak_detection(): Compare detection results between different datasets
- merge_nearby_peaks_valleys(): Merge peaks/valleys that are too close together
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from collections import Counter


def find_peaks_in_data(all_runs_data, data_component='lock1_1', height_threshold=None, 
                       prominence=None, distance=None, width=None, return_details=False,
                       merge_nearby=True, merge_tolerance=0.01, gate_range=None):
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
    merge_nearby : bool, optional
        Whether to merge peaks that are too close together (default: True)
    merge_tolerance : float, optional
        Minimum separation between peaks in V (default: 0.01)
    gate_range : tuple, optional
        (min_gate, max_gate) to restrict peak detection to specific Gate 1 voltage range.
        If None, search entire range. Example: (-4.1, -3.8) to search from -4.1V to -3.8V
    
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
    >>>
    >>> # Find peaks only in specific Gate 1 voltage range
    >>> peaks = find_peaks_in_data(data, data_component='symmetric', 
    ...                           gate_range=(-4.1, -3.8))
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
        
        # Apply gate range filter if specified
        if gate_range is not None:
            min_gate, max_gate = gate_range
            # Find indices within the specified gate range
            gate_mask = (gate1 >= min_gate) & (gate1 <= max_gate)
            gate1_filtered = gate1[gate_mask]
            y_data_filtered = y_data[gate_mask]
            
            # Check if we have enough data points in the range
            if len(gate1_filtered) < 3:
                print(f"Warning: Only {len(gate1_filtered)} data points in gate range {gate_range} for current {current_val}nA")
                # Store empty results for this current
                peak_info = {
                    'current': current_val,
                    'peak_indices': np.array([]),
                    'peak_gate_positions': np.array([]),
                    'peak_heights': np.array([]),
                    'num_peaks': 0,
                    'data_type': data_component,
                    'gate1': gate1,
                    'y_data_uv': y_data / 1e-6,
                    'gate_range': gate_range
                }
                all_peaks_data[i] = peak_info
                continue
        else:
            # Use full range
            gate1_filtered = gate1
            y_data_filtered = y_data
            gate_mask = np.ones(len(gate1), dtype=bool)
        
        # Convert to µV for peak detection
        y_data_uv = y_data_filtered / 1e-6
        
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
        
        # Merge nearby peaks if requested
        if merge_nearby and len(peaks) > 1:
            # Create dummy valley indices for the merge function
            dummy_valleys = np.array([])
            merged_peaks, _ = merge_nearby_peaks_valleys(
                peaks, dummy_valleys, gate1_filtered, y_data_uv, merge_tolerance
            )
            peaks = merged_peaks
        
        # Get peak positions in gate voltage (from filtered data)
        peak_gate_positions = gate1_filtered[peaks]
        peak_heights = y_data_uv[peaks]
        
        # Convert peak indices back to original data indices if gate_range was applied
        if gate_range is not None:
            # Find the original indices corresponding to the filtered peaks
            original_indices = np.where(gate_mask)[0]
            peak_indices_original = original_indices[peaks]
        else:
            peak_indices_original = peaks
        
        # Store results
        peak_info = {
            'current': current_val,
            'peak_indices': peak_indices_original,
            'peak_gate_positions': peak_gate_positions,
            'peak_heights': peak_heights,
            'num_peaks': len(peaks),
            'data_type': data_component,
            'gate1': gate1,
            'y_data_uv': data[data_component] / 1e-6,  # Store original full data
            'gate_range': gate_range
        }
        
        if return_details:
            peak_info['properties'] = properties
            
        all_peaks_data[i] = peak_info
    
    return all_peaks_data


def find_valleys_in_data(all_runs_data, data_component='lock1_1', depth_threshold=None, 
                        prominence=None, distance=None, width=None, return_details=False,
                        merge_nearby=True, merge_tolerance=0.01, gate_range=None):
    """
    Find valleys (local minima) in specified data component for all current values
    
    Parameters
    ----------
    all_runs_data : dict
        Data from all current measurements
    data_component : str
        Which data component to analyze for valleys.
        Options: 'lock1_1', 'lock1_2', 'symmetric', 'antisymmetric', 
                'smoothed_1', 'smoothed_2', 'symmetric_smoothed', 'antisymmetric_smoothed'
    depth_threshold : float, optional
        Maximum depth of valleys (in µV), if None will auto-detect
    prominence : float, optional
        Minimum prominence of valleys, if None will auto-detect  
    distance : int, optional
        Minimum distance between valleys (in data points)
    width : int, optional
        Minimum width of valleys (in data points)
    return_details : bool, optional
        Whether to return detailed valley properties
    merge_nearby : bool, optional
        Whether to merge valleys that are too close together (default: True)
    merge_tolerance : float, optional
        Minimum separation between valleys in V (default: 0.01)
    gate_range : tuple, optional
        (min_gate, max_gate) to restrict valley detection to specific Gate 1 voltage range.
        If None, search entire range. Example: (-4.1, -3.8) to search from -4.1V to -3.8V
    
    Returns
    -------
    dict
        Dictionary with valley information for each current value
        
    Examples
    --------
    >>> # Find valleys in symmetric smoothed data
    >>> valleys = find_valleys_in_data(data, data_component='symmetric_smoothed')
    >>> 
    >>> # Find valleys with custom parameters
    >>> valleys = find_valleys_in_data(data, data_component='antisymmetric', 
    ...                               depth_threshold=-5.0, prominence=2.0)
    >>>
    >>> # Find valleys only in specific Gate 1 voltage range
    >>> valleys = find_valleys_in_data(data, data_component='symmetric', 
    ...                               gate_range=(-4.1, -3.8))
    """
    
    # Validate data component
    valid_components = ['lock1_1', 'lock1_2', 'symmetric', 'antisymmetric', 
                       'smoothed_1', 'smoothed_2', 'symmetric_smoothed', 'antisymmetric_smoothed']
    
    if data_component not in valid_components:
        raise ValueError(f"data_component must be one of: {valid_components}")
    
    all_valleys_data = {}
    
    for i in range(len(all_runs_data)):
        data = all_runs_data[i]
        current_val = data['current']
        gate1 = data['gate1']
        
        # Get the specified data component
        y_data = data[data_component]
        
        # Apply gate range filter if specified
        if gate_range is not None:
            min_gate, max_gate = gate_range
            # Find indices within the specified gate range
            gate_mask = (gate1 >= min_gate) & (gate1 <= max_gate)
            gate1_filtered = gate1[gate_mask]
            y_data_filtered = y_data[gate_mask]
            
            # Check if we have enough data points in the range
            if len(gate1_filtered) < 3:
                print(f"Warning: Only {len(gate1_filtered)} data points in gate range {gate_range} for current {current_val}nA")
                # Store empty results for this current
                valley_info = {
                    'current': current_val,
                    'valley_indices': np.array([]),
                    'valley_gate_positions': np.array([]),
                    'valley_depths': np.array([]),
                    'num_valleys': 0,
                    'data_type': data_component,
                    'gate1': gate1,
                    'y_data_uv': y_data / 1e-6,
                    'gate_range': gate_range
                }
                all_valleys_data[i] = valley_info
                continue
        else:
            # Use full range
            gate1_filtered = gate1
            y_data_filtered = y_data
            gate_mask = np.ones(len(gate1), dtype=bool)
        
        # Convert to µV for valley detection
        y_data_uv = y_data_filtered / 1e-6
        
        # Invert the data to find valleys as peaks
        y_data_inverted = -y_data_uv
        
        # Auto-detect parameters if not provided
        if depth_threshold is None:
            # Use a fraction of the data range as threshold (for inverted data)
            data_range = np.max(y_data_inverted) - np.min(y_data_inverted)
            auto_height = np.min(y_data_inverted) + 0.1 * data_range
        else:
            # Convert depth threshold to height threshold for inverted data
            auto_height = -depth_threshold
            
        if prominence is None:
            # Auto-detect prominence as a fraction of data range
            auto_prominence = 0.05 * (np.max(y_data_inverted) - np.min(y_data_inverted))
        else:
            auto_prominence = prominence
        
        # Find valleys (peaks in inverted data)
        valley_kwargs = {
            'height': auto_height,
            'prominence': auto_prominence
        }
        
        if distance is not None:
            valley_kwargs['distance'] = distance
        if width is not None:
            valley_kwargs['width'] = width
            
        valleys, properties = find_peaks(y_data_inverted, **valley_kwargs)
        
        # Merge nearby valleys if requested
        if merge_nearby and len(valleys) > 1:
            # Create dummy peak indices for the merge function
            dummy_peaks = np.array([])
            _, merged_valleys = merge_nearby_peaks_valleys(
                dummy_peaks, valleys, gate1_filtered, y_data_uv, merge_tolerance
            )
            valleys = merged_valleys
        
        # Get valley positions in gate voltage (from filtered data)
        valley_gate_positions = gate1_filtered[valleys]
        valley_depths = y_data_uv[valleys]  # Use original (non-inverted) data for depths
        
        # Convert valley indices back to original data indices if gate_range was applied
        if gate_range is not None:
            # Find the original indices corresponding to the filtered valleys
            original_indices = np.where(gate_mask)[0]
            valley_indices_original = original_indices[valleys]
        else:
            valley_indices_original = valleys
        
        # Store results
        valley_info = {
            'current': current_val,
            'valley_indices': valley_indices_original,
            'valley_gate_positions': valley_gate_positions,
            'valley_depths': valley_depths,
            'num_valleys': len(valleys),
            'data_type': data_component,
            'gate1': gate1,
            'y_data_uv': data[data_component] / 1e-6,  # Store original full data
            'gate_range': gate_range
        }
        
        if return_details:
            valley_info['properties'] = properties
            
        all_valleys_data[i] = valley_info
    
    return all_valleys_data


def find_peaks_and_valleys_in_data(all_runs_data, data_component='lock1_1', 
                                  peak_height_threshold=None, valley_depth_threshold=None,
                                  peak_prominence=None, valley_prominence=None,
                                  distance=None, width=None, return_details=False,
                                  merge_nearby=True, merge_tolerance=0.01, gate_range=None):
    """
    Find both peaks and valleys in specified data component for all current values
    
    Parameters
    ----------
    all_runs_data : dict
        Data from all current measurements
    data_component : str
        Which data component to analyze
    peak_height_threshold : float, optional
        Minimum height of peaks (in µV)
    valley_depth_threshold : float, optional
        Maximum depth of valleys (in µV)
    peak_prominence : float, optional
        Minimum prominence of peaks
    valley_prominence : float, optional
        Minimum prominence of valleys
    distance : int, optional
        Minimum distance between features (in data points)
    width : int, optional
        Minimum width of features (in data points)
    return_details : bool, optional
        Whether to return detailed properties
    merge_nearby : bool, optional
        Whether to merge nearby features (default: True)
    merge_tolerance : float, optional
        Minimum separation between features in V (default: 0.01)
    gate_range : tuple, optional
        (min_gate, max_gate) to restrict detection to specific Gate 1 voltage range.
        If None, search entire range. Example: (-4.1, -3.8) to search from -4.1V to -3.8V
    
    Returns
    -------
    dict
        Dictionary with both peak and valley information for each current value
        
    Examples
    --------
    >>> # Find both peaks and valleys
    >>> features = find_peaks_and_valleys_in_data(data, data_component='symmetric_smoothed')
    >>> for i, info in features.items():
    ...     print(f"Current {info['current']}: {info['num_peaks']} peaks, {info['num_valleys']} valleys")
    """
    
    # Find peaks
    peaks_data = find_peaks_in_data(all_runs_data, data_component, 
                                   height_threshold=peak_height_threshold,
                                   prominence=peak_prominence,
                                   distance=distance, width=width,
                                   return_details=return_details,
                                   merge_nearby=merge_nearby,
                                   merge_tolerance=merge_tolerance,
                                   gate_range=gate_range)
    
    # Find valleys
    valleys_data = find_valleys_in_data(all_runs_data, data_component,
                                       depth_threshold=valley_depth_threshold,
                                       prominence=valley_prominence,
                                       distance=distance, width=width,
                                       return_details=return_details,
                                       merge_nearby=merge_nearby,
                                       merge_tolerance=merge_tolerance,
                                       gate_range=gate_range)
    
    # Combine results
    combined_data = {}
    for i in range(len(all_runs_data)):
        peak_info = peaks_data[i]
        valley_info = valleys_data[i]
        
        combined_info = {
            'current': peak_info['current'],
            'data_type': data_component,
            'gate1': peak_info['gate1'],
            'y_data_uv': peak_info['y_data_uv'],
            
            # Peak information
            'peak_indices': peak_info['peak_indices'],
            'peak_gate_positions': peak_info['peak_gate_positions'],
            'peak_heights': peak_info['peak_heights'],
            'num_peaks': peak_info['num_peaks'],
            
            # Valley information
            'valley_indices': valley_info['valley_indices'],
            'valley_gate_positions': valley_info['valley_gate_positions'],
            'valley_depths': valley_info['valley_depths'],
            'num_valleys': valley_info['num_valleys']
        }
        
        if return_details:
            combined_info['peak_properties'] = peak_info.get('properties', {})
            combined_info['valley_properties'] = valley_info.get('properties', {})
        
        combined_data[i] = combined_info
    
    return combined_data


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


def analyze_valley_statistics(valleys_data):
    """
    Analyze statistics of detected valleys across all currents
    
    Parameters
    ----------
    valleys_data : dict
        Output from find_valleys_in_data
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with valley statistics including:
        - current_nA: Drive current value
        - num_valleys: Number of valleys detected
        - min_valley_position, max_valley_position: Range of valley positions
        - valley_range: Voltage range spanned by valleys
        - mean_valley_depth, min_valley_depth: Valley depth statistics
    
    Examples
    --------
    >>> valleys = find_valleys_in_data(data, 'symmetric_smoothed')
    >>> stats = analyze_valley_statistics(valleys)
    >>> print(stats[['current_nA', 'num_valleys', 'mean_valley_depth']])
    """
    
    stats_list = []
    
    for i, valley_info in valleys_data.items():
        current_val = valley_info['current']
        valley_positions = valley_info['valley_gate_positions']
        valley_depths = valley_info['valley_depths']
        num_valleys = valley_info['num_valleys']
        
        if num_valleys > 0:
            min_valley_pos = np.min(valley_positions)
            max_valley_pos = np.max(valley_positions)
            valley_range = max_valley_pos - min_valley_pos
            mean_valley_depth = np.mean(valley_depths)
            min_valley_depth = np.min(valley_depths)  # Most negative value
        else:
            min_valley_pos = np.nan
            max_valley_pos = np.nan
            valley_range = np.nan
            mean_valley_depth = np.nan
            min_valley_depth = np.nan
        
        stats = {
            'current_nA': current_val,
            'num_valleys': num_valleys,
            'min_valley_position': min_valley_pos,
            'max_valley_position': max_valley_pos,
            'valley_range': valley_range,
            'mean_valley_depth': mean_valley_depth,
            'min_valley_depth': min_valley_depth
        }
        
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list)


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


def merge_nearby_peaks_valleys(peak_indices, valley_indices, gate_positions, y_data, 
                             merge_tolerance=0.01):
    """
    Merge peaks/valleys that are too close together, keeping only the most prominent one
    
    Parameters
    ----------
    peak_indices : array
        Indices of detected peaks
    valley_indices : array  
        Indices of detected valleys
    gate_positions : array
        Gate voltage positions
    y_data : array
        Data values in µV
    merge_tolerance : float
        Minimum separation required between features (in V)
        
    Returns
    -------
    tuple
        (merged_peak_indices, merged_valley_indices)
    """
    
    def merge_features(indices, data_values, is_peak=True):
        """Merge nearby features, keeping the most extreme one"""
        if len(indices) <= 1:
            return indices
            
        # Get positions and values
        positions = gate_positions[indices]
        values = data_values[indices]
        
        # Sort by position
        sort_order = np.argsort(positions)
        sorted_indices = indices[sort_order]
        sorted_positions = positions[sort_order]
        sorted_values = values[sort_order]
        
        merged_indices = []
        i = 0
        
        while i < len(sorted_indices):
            current_idx = sorted_indices[i]
            current_pos = sorted_positions[i]
            current_val = sorted_values[i]
            
            # Find all nearby features
            nearby_indices = [current_idx]
            nearby_values = [current_val]
            
            j = i + 1
            while j < len(sorted_indices) and (sorted_positions[j] - current_pos) <= merge_tolerance:
                nearby_indices.append(sorted_indices[j])
                nearby_values.append(sorted_values[j])
                j += 1
            
            # Keep the most extreme feature in this group
            if is_peak:
                # For peaks, keep the highest value
                best_idx = np.argmax(nearby_values)
            else:
                # For valleys, keep the lowest value  
                best_idx = np.argmin(nearby_values)
                
            merged_indices.append(nearby_indices[best_idx])
            i = j
            
        return np.array(merged_indices)
    
    # Merge peaks and valleys separately
    merged_peaks = merge_features(peak_indices, y_data, is_peak=True)
    merged_valleys = merge_features(valley_indices, y_data, is_peak=False)
    
    return merged_peaks, merged_valleys
