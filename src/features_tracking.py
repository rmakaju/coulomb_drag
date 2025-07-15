"""
Feature Tracking Module for Coulomb Drag Data

This module contains advanced feature tracking and evolution analysis functions
for studying how peaks and valleys evolve with experimental parameters like drive current.
It consolidates functionality from valley_tracking.py and unified_feature_tracking.py.

UNIVERSAL FUNCTIONS (recommended for new code):
- plot_feature_tracking_across_current(): Universal function to track peaks or valleys across currents
- plot_feature_tracking_selective(): Universal function with selective plotting options for peaks or valleys
- plot_feature_tracking_all_files_only(): Universal function to track only persistent features across all files

BACKWARD-COMPATIBLE WRAPPER FUNCTIONS (for existing code):
- plot_peak_tracking_across_current(): Track peaks across different currents
- plot_valley_tracking_across_current(): Track valleys across different currents
- plot_peak_tracking_selective(): Advanced peak tracking with filtering options  
- plot_valley_tracking_selective(): Advanced valley tracking with filtering options
- plot_peak_tracking_all_files_only(): Track peaks only if present in all data files
- plot_valley_tracking_all_files_only(): Track valleys only if present in all data files

SPECIALIZED ANALYSIS FUNCTIONS:
- plot_peak_evolution(): Visualize how peak characteristics evolve with current
- analyze_peak_persistence(): Analyze which peaks persist across multiple measurements
- plot_peak_trajectories(): Plot trajectories showing peak position and amplitude evolution
- plot_combined_peaks_valleys_tracking(): Track and plot peaks and valleys together

USAGE EXAMPLES:
    # New unified approach (recommended)
    from peak_detection import find_peaks_in_data, find_valleys_in_data
    
    peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    valleys = find_valleys_in_data(data, 'symmetric_smoothed')
    
    # Track peaks using universal function
    plot_feature_tracking_across_current(peaks, data, 'peaks')
    
    # Track valleys using universal function
    plot_feature_tracking_across_current(valleys, data, 'valleys')
    
    # Selective tracking with filtering
    plot_feature_tracking_selective(peaks, data, 'peaks', feature_indices_to_plot=[0, 2])
    
    # Track only highly persistent features
    plot_feature_tracking_all_files_only(peaks, data, 'peaks')
    
    # Legacy approach (still supported)
    plot_peak_tracking_across_current(peaks, data)
    plot_valley_tracking_across_current(valleys, data)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from peak_detection import analyze_peak_statistics


def plot_feature_tracking_across_current(features_data, all_runs_data, feature_type='peaks', 
                                        position_tolerance=0.01, figsize=(12, 8)):
    """
    Universal function to track features (peaks or valleys) across different currents
    
    Parameters
    ----------
    features_data : dict
        Output from find_peaks_in_data or find_valleys_in_data
    all_runs_data : dict
        Data from all current measurements
    feature_type : str, optional
        Type of features to track: 'peaks' or 'valleys' (default: 'peaks')
    position_tolerance : float, optional
        Tolerance in V for grouping features at similar positions (default: 0.01)
    figsize : tuple, optional
        Figure size (default: (12, 8))
    
    Returns
    -------
    list
        List of DataFrames containing grouped feature data
        
    Examples
    --------
    >>> from peak_detection import find_peaks_in_data, find_valleys_in_data
    >>> 
    >>> # Track peaks across current
    >>> peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> groups = plot_feature_tracking_across_current(peaks, data, 'peaks')
    >>>
    >>> # Track valleys across current
    >>> valleys = find_valleys_in_data(data, 'symmetric_smoothed')
    >>> groups = plot_feature_tracking_across_current(valleys, data, 'valleys')
    """
    
    # Validate feature type
    if feature_type not in ['peaks', 'valleys']:
        raise ValueError("feature_type must be either 'peaks' or 'valleys'")
    
    # Get data component info from the first entry
    if len(features_data) == 0:
        print(f"No {feature_type} data provided")
        return []
        
    data_component = features_data[0]['data_type']
    
    # Determine which keys to use based on feature type
    if feature_type == 'peaks':
        position_key = 'peak_gate_positions'
        indices_key = 'peak_indices'
        marker_style = 'o-'
        marker_single = 'o'
    else:  # valleys
        position_key = 'valley_gate_positions'
        indices_key = 'valley_indices'
        marker_style = 's-'
        marker_single = 's'
    
    # Collect all feature data with actual lockin values
    all_data = []
    for i, feature_info in features_data.items():
        # Get the actual lockin values at feature positions from all_runs_data
        current_data = all_runs_data[i]
        lockin_values = current_data[data_component] / 1e-6  # Convert to µV
        
        # Get lockin values at feature indices
        feature_lockin_values = lockin_values[feature_info[indices_key]]
        
        for pos, lockin_val in zip(feature_info[position_key], feature_lockin_values):
            all_data.append({
                'current': feature_info['current'],
                'position': pos,
                'lockin_value': lockin_val
            })
    
    if len(all_data) == 0:
        print(f"No {feature_type} found for tracking")
        return []
    
    df = pd.DataFrame(all_data)
    
    # Group features by similar positions
    position_groups = []
    used_features = set()
    
    for idx, row in df.iterrows():
        if idx in used_features:
            continue
            
        # Find all features within tolerance of this position
        similar_features = df[abs(df['position'] - row['position']) <= position_tolerance]
        position_groups.append(similar_features)
        used_features.update(similar_features.index)
    
    # Sort groups by mean position for consistent indexing
    position_groups.sort(key=lambda x: x['position'].mean())
    
    # Plot tracked features
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(position_groups)))
    
    feature_name = feature_type.capitalize()[:-1]  # Remove 's' from 'peaks'/'valleys'
    for i, group in enumerate(position_groups):
        if len(group) > 1:  # Only plot if feature appears in multiple currents
            mean_position = group['position'].mean()
            plt.plot(group['current'], group['lockin_value'], marker_style, 
                    color=colors[i], linewidth=2, markersize=8, alpha=0.8,
                    label=f'{feature_name}: ~{mean_position:.3f}V')
        else:
            # Single feature - plot as isolated point
            mean_position = group['position'].mean()
            plt.scatter(group['current'], group['lockin_value'], 
                       s=50, alpha=0.5, color=colors[i], marker=marker_single,
                       label=f'{feature_name}: ~{mean_position:.3f}V (single)')
    
    plt.xlabel('Current (nA)', fontsize=14)
    plt.ylabel(f'{feature_name} {data_component} Value (µV)', fontsize=14)
    plt.title(f'{data_component.title()} {feature_name} Evolution', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return position_groups


def plot_feature_tracking_selective(features_data, all_runs_data, feature_type='peaks',
                                   position_tolerance=0.05, feature_indices_to_plot=None, 
                                   position_range=None, figsize=(12, 8)):
    """
    Universal function to track features (peaks or valleys) with selective plotting options
    
    Parameters
    ----------
    features_data : dict
        Output from find_peaks_in_data or find_valleys_in_data
    all_runs_data : dict
        Data from all current measurements
    feature_type : str, optional
        Type of features to track: 'peaks' or 'valleys' (default: 'peaks')
    position_tolerance : float, optional
        Tolerance in V for grouping features at similar positions (default: 0.05)
    feature_indices_to_plot : list, optional
        List of feature group indices to plot (None = plot all)
    position_range : tuple, optional
        (min_pos, max_pos) to filter features by position (None = all positions)
    figsize : tuple, optional
        Figure size (default: (12, 8))
    
    Returns
    -------
    list
        List of DataFrames containing grouped feature data
        
    Examples
    --------
    >>> from peak_detection import find_peaks_in_data, find_valleys_in_data
    >>> 
    >>> # Plot only peaks 0 and 2
    >>> peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> groups = plot_feature_tracking_selective(peaks, data, 'peaks', 
    ...                                         feature_indices_to_plot=[0, 2])
    >>>
    >>> # Plot only valleys in voltage range -2.0 to -1.5 V
    >>> valleys = find_valleys_in_data(data, 'symmetric_smoothed')
    >>> groups = plot_feature_tracking_selective(valleys, data, 'valleys',
    ...                                         position_range=(-2.0, -1.5))
    """
    
    # Validate feature type
    if feature_type not in ['peaks', 'valleys']:
        raise ValueError("feature_type must be either 'peaks' or 'valleys'")
    
    # Get data component info from the first entry
    if len(features_data) == 0:
        print(f"No {feature_type} data provided")
        return []
        
    data_component = features_data[0]['data_type']
    
    # Determine which keys to use based on feature type
    if feature_type == 'peaks':
        position_key = 'peak_gate_positions'
        indices_key = 'peak_indices'
        marker_style = 'o-'
        marker_single = 'o'
    else:  # valleys
        position_key = 'valley_gate_positions'
        indices_key = 'valley_indices'
        marker_style = 's-'
        marker_single = 's'
    
    # Collect all feature data with actual lockin values
    all_data = []
    for i, feature_info in features_data.items():
        # Get the actual lockin values at feature positions from all_runs_data
        current_data = all_runs_data[i]
        lockin_values = current_data[data_component] / 1e-6  # Convert to µV
        
        # Get lockin values at feature indices
        feature_lockin_values = lockin_values[feature_info[indices_key]]
        
        for pos, lockin_val in zip(feature_info[position_key], feature_lockin_values):
            all_data.append({
                'current': feature_info['current'],
                'position': pos,
                'lockin_value': lockin_val
            })
    
    if len(all_data) == 0:
        print(f"No {feature_type} found for tracking")
        return []
    
    df = pd.DataFrame(all_data)
    
    # Group features by similar positions
    position_groups = []
    used_features = set()
    
    for idx, row in df.iterrows():
        if idx in used_features:
            continue
            
        # Find all features within tolerance of this position
        similar_features = df[abs(df['position'] - row['position']) <= position_tolerance]
        position_groups.append(similar_features)
        used_features.update(similar_features.index)
    
    # Sort groups by mean position for consistent indexing
    position_groups.sort(key=lambda x: x['position'].mean())
    
    # Print available features for user reference BEFORE filtering
    feature_name = feature_type.capitalize()[:-1]  # Remove 's' from 'peaks'/'valleys'
    print(f"All available {feature_type} groups (total: {len(position_groups)}):")
    for i, group in enumerate(position_groups):
        mean_pos = group['position'].mean()
        num_points = len(group)
        print(f"  {feature_name} {i}: ~{mean_pos:.3f}V ({num_points} data points)")
    
    # Apply position range filter if specified
    if position_range is not None:
        min_pos, max_pos = position_range
        filtered_groups = [group for group in position_groups 
                          if min_pos <= group['position'].mean() <= max_pos]
        print(f"\nFiltered by position range {position_range}: {len(filtered_groups)} groups")
        position_groups = filtered_groups
    
    # Apply feature index filter if specified
    if feature_indices_to_plot is not None:
        original_groups = position_groups.copy()
        position_groups = []
        for i in feature_indices_to_plot:
            if i < len(original_groups):
                position_groups.append(original_groups[i])
            else:
                print(f"Warning: {feature_name} index {i} is out of range (max: {len(original_groups)-1})")
        print(f"\nSelected {feature_type} indices {feature_indices_to_plot}: {len(position_groups)} groups")
    
    if len(position_groups) == 0:
        print(f"No {feature_type} match the selection criteria")
        return position_groups
    
    # Plot tracked features
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(position_groups)))
    
    for i, group in enumerate(position_groups):
        if len(group) > 1:  # Only plot if feature appears in multiple currents
            mean_position = group['position'].mean()
            plt.plot(group['current'], group['lockin_value'], marker_style, 
                    color=colors[i], linewidth=2, markersize=8, alpha=0.8,
                    label=f'{feature_name}: ~{mean_position:.3f}V')
        else:
            # Single feature - plot as isolated point
            mean_position = group['position'].mean()
            plt.scatter(group['current'], group['lockin_value'], 
                       s=50, alpha=0.5, color=colors[i], marker=marker_single,
                       label=f'{feature_name}: ~{mean_position:.3f}V (single)')
    
    plt.xlabel('Current (nA)', fontsize=14)
    plt.ylabel(f'{feature_name} {data_component} Value (µV)', fontsize=14)
    
    # Update title based on selection
    title_suffix = ""
    if feature_indices_to_plot is not None:
        title_suffix += f" (Selected: {feature_indices_to_plot})"
    if position_range is not None:
        title_suffix += f" (Range: {position_range[0]:.3f}-{position_range[1]:.3f}V)"
    
    plt.title(f'{data_component.title()} {feature_name} Evolution{title_suffix}', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return position_groups


def plot_feature_tracking_all_files_only(features_data, all_runs_data, feature_type='peaks', 
                                        position_tolerance=0.01, figsize=(12, 8)):
    """
    Universal function to track features that appear in ALL files (strict persistence requirement)
    
    Parameters
    ----------
    features_data : dict
        Output from find_peaks_in_data or find_valleys_in_data
    all_runs_data : dict
        Data from all current measurements
    feature_type : str, optional
        Type of features to track: 'peaks' or 'valleys' (default: 'peaks')
    position_tolerance : float, optional
        Tolerance in V for grouping features at similar positions (default: 0.01)
    figsize : tuple, optional
        Figure size (default: (12, 8))
    
    Returns
    -------
    list
        List of DataFrames containing grouped feature data for persistent features only
        
    Examples
    --------
    >>> from peak_detection import find_peaks_in_data, find_valleys_in_data
    >>> 
    >>> # Track only highly persistent peaks
    >>> peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> groups = plot_feature_tracking_all_files_only(peaks, data, 'peaks')
    >>>
    >>> # Track only highly persistent valleys
    >>> valleys = find_valleys_in_data(data, 'symmetric_smoothed')
    >>> groups = plot_feature_tracking_all_files_only(valleys, data, 'valleys')
    """
    
    # Validate feature type
    if feature_type not in ['peaks', 'valleys']:
        raise ValueError("feature_type must be either 'peaks' or 'valleys'")
    
    # Get data component info from the first entry
    if len(features_data) == 0:
        print(f"No {feature_type} data provided")
        return []
        
    data_component = features_data[0]['data_type']
    
    # Determine which keys to use based on feature type
    if feature_type == 'peaks':
        position_key = 'peak_gate_positions'
        indices_key = 'peak_indices'
        marker_style = 'o-'
    else:  # valleys
        position_key = 'valley_gate_positions'
        indices_key = 'valley_indices'
        marker_style = 's-'
    
    # Collect all feature data with actual lockin values
    all_data = []
    for i, feature_info in features_data.items():
        # Get the actual lockin values at feature positions from all_runs_data
        current_data = all_runs_data[i]
        lockin_values = current_data[data_component] / 1e-6  # Convert to µV
        
        # Get lockin values at feature indices
        feature_lockin_values = lockin_values[feature_info[indices_key]]
        
        for pos, lockin_val in zip(feature_info[position_key], feature_lockin_values):
            all_data.append({
                'current': feature_info['current'],
                'position': pos,
                'lockin_value': lockin_val
            })
    
    if len(all_data) == 0:
        print(f"No {feature_type} found for tracking")
        return []
    
    df = pd.DataFrame(all_data)
    
    # Group features by similar positions
    position_groups = []
    used_features = set()
    
    for idx, row in df.iterrows():
        if idx in used_features:
            continue
            
        # Find all features within tolerance of this position
        similar_features = df[abs(df['position'] - row['position']) <= position_tolerance]
        position_groups.append(similar_features)
        used_features.update(similar_features.index)
    
    # Filter to only include features that appear in ALL files
    total_files = len(features_data)
    persistent_groups = []
    
    for group in position_groups:
        unique_currents = group['current'].nunique()
        if unique_currents == total_files:  # Feature appears in all files
            persistent_groups.append(group)
    
    # Sort groups by mean position for consistent indexing
    persistent_groups.sort(key=lambda x: x['position'].mean())
    
    feature_name = feature_type.capitalize()[:-1]  # Remove 's' from 'peaks'/'valleys'
    print(f"All {feature_type} groups: {len(position_groups)}")
    print(f"Persistent {feature_type} (appear in all {total_files} files): {len(persistent_groups)}")
    
    if len(persistent_groups) == 0:
        print(f"No {feature_type} appear in all files")
        return persistent_groups
    
    # Plot only persistent features
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(persistent_groups)))
    
    for i, group in enumerate(persistent_groups):
        mean_position = group['position'].mean()
        plt.plot(group['current'], group['lockin_value'], marker_style, 
                color=colors[i], linewidth=2, markersize=8, alpha=0.8,
                label=f'{feature_name}: ~{mean_position:.3f}V')
    
    plt.xlabel('Current (nA)', fontsize=14)
    plt.ylabel(f'{feature_name} {data_component} Value (µV)', fontsize=14)
    plt.title(f'{data_component.title()} {feature_name} Evolution (All Files Only)', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return persistent_groups


def plot_peak_tracking_across_current(peaks_data, all_runs_data, position_tolerance=0.01, figsize=(12, 8)):
    """
    Track individual peaks across different currents by grouping peaks at similar gate positions
    
    Parameters
    ----------
    peaks_data : dict
        Output from find_peaks_in_data
    all_runs_data : dict
        Data from all current measurements
    position_tolerance : float, optional
        Tolerance in V for grouping peaks at similar positions (default: 0.01)
    figsize : tuple, optional
        Figure size (default: (12, 8))
        
    Returns
    -------
    list
        List of DataFrames containing grouped peak data
        
    Examples
    --------
    >>> from peak_detection import find_peaks_in_data
    >>> peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> groups = plot_peak_tracking_across_current(peaks, data, position_tolerance=0.1)
    >>> print(f"Found {len(groups)} peak groups")
    """
    
    return plot_feature_tracking_across_current(peaks_data, all_runs_data, 'peaks', 
                                               position_tolerance, figsize)


def plot_peak_tracking_selective(peaks_data, all_runs_data, position_tolerance=0.05, 
                                peak_indices_to_plot=None, position_range=None, 
                                figsize=(12, 8)):
    """
    Track individual peaks across different currents with selective plotting options
    
    Parameters
    ----------
    peaks_data : dict
        Output from find_peaks_in_data
    all_runs_data : dict
        Data from all current measurements
    position_tolerance : float, optional
        Tolerance in V for grouping peaks at similar positions (default: 0.05)
    peak_indices_to_plot : list, optional
        List of peak group indices to plot (None = plot all)
    position_range : tuple, optional
        (min_pos, max_pos) to filter peaks by position (None = all positions)
    figsize : tuple, optional
        Figure size (default: (12, 8))
    
    Returns
    -------
    list
        List of DataFrames containing grouped peak data
        
    Examples
    --------
    >>> from peak_detection import find_peaks_in_data
    >>> peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> 
    >>> # Plot only peaks 0 and 2
    >>> groups = plot_peak_tracking_selective(peaks, data, peak_indices_to_plot=[0, 2])
    >>>
    >>> # Plot only peaks in voltage range -2.0 to -1.5 V
    >>> groups = plot_peak_tracking_selective(peaks, data, position_range=(-2.0, -1.5))
    """
    return plot_feature_tracking_selective(peaks_data, all_runs_data, 'peaks',
                                          position_tolerance, peak_indices_to_plot,
                                          position_range, figsize)


def plot_peak_evolution(peaks_data):
    """
    Plot how peak characteristics evolve with current
    
    Parameters
    ----------
    peaks_data : dict
        Output from find_peaks_in_data
        
    Returns
    -------
    pandas.DataFrame
        Statistics dataframe for further analysis
        
    Examples
    --------
    >>> from peak_detection import find_peaks_in_data
    >>> peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> stats_df = plot_peak_evolution(peaks)
    >>> print(stats_df[['current_nA', 'num_peaks', 'max_peak_height']])
    """
    
    stats_df = analyze_peak_statistics(peaks_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Number of peaks vs current
    axes[0,0].plot(stats_df['current_nA'], stats_df['num_peaks'], 'bo-', linewidth=2, markersize=8)
    axes[0,0].set_xlabel('Current (nA)')
    axes[0,0].set_ylabel('Number of Peaks')
    axes[0,0].set_title('Peak Count vs Current')
    axes[0,0].grid(True, alpha=0.3)
    
    # Peak range vs current
    axes[0,1].plot(stats_df['current_nA'], stats_df['peak_range'], 'ro-', linewidth=2, markersize=8)
    axes[0,1].set_xlabel('Current (nA)')
    axes[0,1].set_ylabel('Peak Range (V)')
    axes[0,1].set_title('Peak Voltage Range vs Current')
    axes[0,1].grid(True, alpha=0.3)
    
    # Maximum peak height vs current
    axes[1,0].plot(stats_df['current_nA'], stats_df['max_peak_height'], 'go-', linewidth=2, markersize=8)
    axes[1,0].set_xlabel('Current (nA)')
    axes[1,0].set_ylabel('Max Peak Height (µV)')
    axes[1,0].set_title('Maximum Peak Height vs Current')
    axes[1,0].grid(True, alpha=0.3)
    
    # Mean peak height vs current
    axes[1,1].plot(stats_df['current_nA'], stats_df['mean_peak_height'], 'mo-', linewidth=2, markersize=8)
    axes[1,1].set_xlabel('Current (nA)')
    axes[1,1].set_ylabel('Mean Peak Height (µV)')
    axes[1,1].set_title('Mean Peak Height vs Current')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return stats_df


def analyze_peak_persistence(peaks_data, position_tolerance=0.05, min_appearances=3):
    """
    Analyze which peaks persist across multiple current measurements
    
    Parameters
    ----------
    peaks_data : dict
        Output from find_peaks_in_data
    position_tolerance : float, optional
        Tolerance in V for grouping peaks at similar positions (default: 0.05)
    min_appearances : int, optional
        Minimum number of appearances to consider a peak 'persistent' (default: 3)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with persistent peak information
        
    Examples
    --------
    >>> from peak_detection import find_peaks_in_data
    >>> peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> persistent = analyze_peak_persistence(peaks, min_appearances=5)
    >>> print(f"Found {len(persistent)} persistent peaks")
    """
    
    # Collect all peak data
    all_data = []
    for i, peak_info in peaks_data.items():
        for pos, height in zip(peak_info['peak_gate_positions'], peak_info['peak_heights']):
            all_data.append({
                'current': peak_info['current'],
                'position': pos,
                'height': height
            })
    
    if len(all_data) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    
    # Group peaks by similar positions
    position_groups = []
    used_peaks = set()
    
    for idx, row in df.iterrows():
        if idx in used_peaks:
            continue
            
        # Find all peaks within tolerance of this position
        similar_peaks = df[abs(df['position'] - row['position']) <= position_tolerance]
        if len(similar_peaks) >= min_appearances:
            position_groups.append(similar_peaks)
        used_peaks.update(similar_peaks.index)
    
    # Analyze persistent peaks
    persistent_peaks = []
    for i, group in enumerate(position_groups):
        peak_info = {
            'peak_id': i,
            'mean_position': group['position'].mean(),
            'position_std': group['position'].std(),
            'appearances': len(group),
            'mean_height': group['height'].mean(),
            'height_std': group['height'].std(),
            'min_current': group['current'].min(),
            'max_current': group['current'].max(),
            'current_range': group['current'].max() - group['current'].min()
        }
        persistent_peaks.append(peak_info)
    
    return pd.DataFrame(persistent_peaks)


def plot_peak_trajectories(peaks_data, all_runs_data, position_tolerance=0.05, 
                          show_position_evolution=True, figsize=(15, 6)):
    """
    Plot trajectories showing both position and amplitude evolution of peaks
    
    Parameters
    ----------
    peaks_data : dict
        Output from find_peaks_in_data
    all_runs_data : dict
        Data from all current measurements
    position_tolerance : float, optional
        Tolerance in V for grouping peaks at similar positions (default: 0.05)
    show_position_evolution : bool, optional
        Whether to show how peak positions evolve (default: True)
    figsize : tuple, optional
        Figure size (default: (15, 6))
        
    Examples
    --------
    >>> from peak_detection import find_peaks_in_data
    >>> peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> plot_peak_trajectories(peaks, data)
    """
    
    if len(peaks_data) == 0:
        print("No peak data provided")
        return
        
    data_component = peaks_data[0]['data_type']
    
    # Collect and group peak data
    all_data = []
    for i, peak_info in peaks_data.items():
        current_data = all_runs_data[i]
        lockin_values = current_data[data_component] / 1e-6
        peak_lockin_values = lockin_values[peak_info['peak_indices']]
        
        for pos, lockin_val in zip(peak_info['peak_gate_positions'], peak_lockin_values):
            all_data.append({
                'current': peak_info['current'],
                'position': pos,
                'lockin_value': lockin_val
            })
    
    if len(all_data) == 0:
        print("No peaks found")
        return
    
    df = pd.DataFrame(all_data)
    
    # Group peaks by similar positions
    position_groups = []
    used_peaks = set()
    
    for idx, row in df.iterrows():
        if idx in used_peaks:
            continue
        similar_peaks = df[abs(df['position'] - row['position']) <= position_tolerance]
        if len(similar_peaks) > 1:  # Only track peaks that appear multiple times
            position_groups.append(similar_peaks)
        used_peaks.update(similar_peaks.index)
    
    if show_position_evolution:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(position_groups)))
    
    # Plot position evolution if requested
    if show_position_evolution:
        for i, group in enumerate(position_groups):
            group_sorted = group.sort_values('current')
            ax1.plot(group_sorted['current'], group_sorted['position'], 'o-',
                    color=colors[i], linewidth=2, markersize=6, alpha=0.8,
                    label=f'Peak ~{group["position"].mean():.3f}V')
        
        ax1.set_xlabel('Current (nA)', fontsize=12)
        ax1.set_ylabel('Peak Position (V)', fontsize=12)
        ax1.set_title('Peak Position Evolution', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
    
    # Plot amplitude evolution
    for i, group in enumerate(position_groups):
        group_sorted = group.sort_values('current')
        ax2.plot(group_sorted['current'], group_sorted['lockin_value'], 'o-',
                color=colors[i], linewidth=2, markersize=6, alpha=0.8,
                label=f'Peak ~{group["position"].mean():.3f}V')
    
    ax2.set_xlabel('Current (nA)', fontsize=12)
    ax2.set_ylabel(f'Peak {data_component} Value (µV)', fontsize=12)
    ax2.set_title('Peak Amplitude Evolution', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return position_groups


def plot_combined_peaks_valleys_tracking(peaks_data, valleys_data, all_runs_data, 
                                        position_tolerance=0.02, figsize=(14, 8)):
    """
    Track both peaks and valleys together on the same plot
    
    Parameters
    ----------
    peaks_data : dict
        Output from find_peaks_in_data
    valleys_data : dict
        Output from find_valleys_in_data
    all_runs_data : dict
        Data from all current measurements
    position_tolerance : float, optional
        Tolerance in V for grouping features at similar positions (default: 0.05)
    figsize : tuple, optional
        Figure size (default: (14, 8))
        
    Returns
    -------
    tuple
        (peak_groups, valley_groups) - Lists of DataFrames containing grouped data
        
    Examples
    --------
    >>> from peak_detection import find_peaks_in_data, find_valleys_in_data
    >>> peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> valleys = find_valleys_in_data(data, 'symmetric_smoothed')
    >>> plot_combined_peaks_valleys_tracking(peaks, valleys, data)
    """
    
    if len(peaks_data) == 0 and len(valleys_data) == 0:
        print("No peak or valley data provided")
        return [], []
        
    # Determine data component (prefer from peaks, fallback to valleys)
    if len(peaks_data) > 0:
        data_component = peaks_data[0]['data_type']
    else:
        data_component = valleys_data[0]['data_type']
    
    # Process peaks
    peak_groups = []
    if len(peaks_data) > 0:
        peak_data = []
        for i, peak_info in peaks_data.items():
            current_data = all_runs_data[i]
            lockin_values = current_data[data_component] / 1e-6
            peak_lockin_values = lockin_values[peak_info['peak_indices']]
            
            for pos, lockin_val in zip(peak_info['peak_gate_positions'], peak_lockin_values):
                peak_data.append({
                    'current': peak_info['current'],
                    'position': pos,
                    'lockin_value': lockin_val
                })
        
        if peak_data:
            peak_df = pd.DataFrame(peak_data)
            used_peaks = set()
            
            for idx, row in peak_df.iterrows():
                if idx in used_peaks:
                    continue
                similar_peaks = peak_df[abs(peak_df['position'] - row['position']) <= position_tolerance]
                if len(similar_peaks) > 1:
                    peak_groups.append(similar_peaks)
                used_peaks.update(similar_peaks.index)
    
    # Process valleys
    valley_groups = []
    if len(valleys_data) > 0:
        valley_data = []
        for i, valley_info in valleys_data.items():
            current_data = all_runs_data[i]
            lockin_values = current_data[data_component] / 1e-6
            valley_lockin_values = lockin_values[valley_info['valley_indices']]
            
            for pos, lockin_val in zip(valley_info['valley_gate_positions'], valley_lockin_values):
                valley_data.append({
                    'current': valley_info['current'],
                    'position': pos,
                    'lockin_value': lockin_val
                })
        
        if valley_data:
            valley_df = pd.DataFrame(valley_data)
            used_valleys = set()
            
            for idx, row in valley_df.iterrows():
                if idx in used_valleys:
                    continue
                similar_valleys = valley_df[abs(valley_df['position'] - row['position']) <= position_tolerance]
                if len(similar_valleys) > 1:
                    valley_groups.append(similar_valleys)
                used_valleys.update(similar_valleys.index)
    
    # Plot
    plt.figure(figsize=figsize)
    
    # Plot peaks with circle markers and warm colors
    if peak_groups:
        peak_colors = plt.cm.Reds(np.linspace(0.4, 1, len(peak_groups)))
        for i, group in enumerate(peak_groups):
            mean_position = group['position'].mean()
            plt.plot(group['current'], group['lockin_value'], 'o-', 
                    color=peak_colors[i], linewidth=2, markersize=8, alpha=0.8,
                    label=f'Peak ~{mean_position:.3f}V')
    
    # Plot valleys with square markers and cool colors
    if valley_groups:
        valley_colors = plt.cm.Blues(np.linspace(0.4, 1, len(valley_groups)))
        for i, group in enumerate(valley_groups):
            mean_position = group['position'].mean()
            plt.plot(group['current'], group['lockin_value'], 's-', 
                    color=valley_colors[i], linewidth=2, markersize=8, alpha=0.8,
                    label=f'Valley ~{mean_position:.3f}V')
    
    plt.xlabel('Current (nA)', fontsize=14)
    plt.ylabel(f'{data_component} Value (µV)', fontsize=14)
    plt.title(f'{data_component.title()} Peaks & Valleys Evolution with Current', fontsize=16)
    
    # Create custom legend to distinguish peaks and valleys
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Add peak legend entries
    if peak_groups:
        legend_elements.append(Line2D([0], [0], marker='o', color='red', linestyle='-',
                                    markersize=8, label='Peaks', alpha=0.7))
    
    # Add valley legend entries  
    if valley_groups:
        legend_elements.append(Line2D([0], [0], marker='s', color='blue', linestyle='-',
                                    markersize=8, label='Valleys', alpha=0.7))
    
    # Add individual feature legends
    if peak_groups:
        for i, group in enumerate(peak_groups):
            mean_position = group['position'].mean()
            legend_elements.append(Line2D([0], [0], marker='o', color=peak_colors[i], 
                                        linestyle='-', markersize=6, 
                                        label=f'Peak ~{mean_position:.3f}V', alpha=0.8))
    
    if valley_groups:
        for i, group in enumerate(valley_groups):
            mean_position = group['position'].mean()
            legend_elements.append(Line2D([0], [0], marker='s', color=valley_colors[i], 
                                        linestyle='-', markersize=6,
                                        label=f'Valley ~{mean_position:.3f}V', alpha=0.8))
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Tracked {len(peak_groups)} peak groups and {len(valley_groups)} valley groups")
    
    return peak_groups, valley_groups


def plot_valley_tracking_across_current(valleys_data, all_runs_data, position_tolerance=0.01, figsize=(12, 8)):
    """
    Track individual valleys across different currents by grouping valleys at similar gate positions
    
    Parameters
    ----------
    valleys_data : dict
        Output from find_valleys_in_data
    all_runs_data : dict
        Data from all current measurements
    position_tolerance : float, optional
        Tolerance in V for grouping valleys at similar positions (default: 0.01)
    figsize : tuple, optional
        Figure size (default: (12, 8))
        
    Returns
    -------
    list
        List of DataFrames containing grouped valley data
        
    Examples
    --------
    >>> from peak_detection import find_valleys_in_data
    >>> valleys = find_valleys_in_data(data, 'symmetric_smoothed')
    >>> groups = plot_valley_tracking_across_current(valleys, data, position_tolerance=0.1)
    >>> print(f"Found {len(groups)} valley groups")
    """
    
    return plot_feature_tracking_across_current(valleys_data, all_runs_data, 'valleys', 
                                               position_tolerance, figsize)


def plot_peak_tracking_all_files_only(peaks_data, all_runs_data, position_tolerance=0.01, figsize=(12, 8)):
    """
    Track individual peaks across different currents, but only plot peaks that are found in ALL data files
    
    Parameters
    ----------
    peaks_data : dict
        Output from find_peaks_in_data
    all_runs_data : dict
        Data from all current measurements
    position_tolerance : float, optional
        Tolerance in V for grouping peaks at similar positions (default: 0.01)
    figsize : tuple, optional
        Figure size (default: (12, 8))
        
    Returns
    -------
    list
        List of DataFrames containing grouped peak data that appear in all files
        
    Examples
    --------
    >>> from peak_detection import find_peaks_in_data
    >>> peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> groups = plot_peak_tracking_all_files_only(peaks, data, position_tolerance=0.01)
    >>> print(f"Found {len(groups)} peak groups present in all files")
    """
    return plot_feature_tracking_all_files_only(peaks_data, all_runs_data, 'peaks',
                                               position_tolerance, figsize)


def plot_valley_tracking_all_files_only(valleys_data, all_runs_data, position_tolerance=0.01, figsize=(12, 8)):
    """
    Track individual valleys across different currents, but only plot valleys that are found in ALL data files
    
    Parameters
    ----------
    valleys_data : dict
        Output from find_valleys_in_data
    all_runs_data : dict
        Data from all current measurements
    position_tolerance : float, optional
        Tolerance in V for grouping valleys at similar positions (default: 0.01)
    figsize : tuple, optional
        Figure size (default: (12, 8))
        
    Returns
    -------
    list
        List of DataFrames containing grouped valley data that appear in all files
        
    Examples
    --------
    >>> from peak_detection import find_valleys_in_data
    >>> valleys = find_valleys_in_data(data, 'symmetric_smoothed')
    >>> groups = plot_valley_tracking_all_files_only(valleys, data, position_tolerance=0.01)
    >>> print(f"Found {len(groups)} valley groups present in all files")
    """
    return plot_feature_tracking_all_files_only(valleys_data, all_runs_data, 'valleys',
                                               position_tolerance, figsize)


def plot_valley_tracking_selective(valleys_data, all_runs_data, position_tolerance=0.05, 
                                  valley_indices_to_plot=None, position_range=None, 
                                  figsize=(12, 8)):
    """
    Track individual valleys across different currents with selective plotting options
    
    Parameters
    ----------
    valleys_data : dict
        Output from find_valleys_in_data
    all_runs_data : dict
        Data from all current measurements
    position_tolerance : float, optional
        Tolerance in V for grouping valleys at similar positions (default: 0.05)
    valley_indices_to_plot : list, optional
        List of valley group indices to plot (None = plot all)
    position_range : tuple, optional
        (min_pos, max_pos) to filter valleys by position (None = all positions)
    figsize : tuple, optional
        Figure size (default: (12, 8))
    
    Returns
    -------
    list
        List of DataFrames containing grouped valley data
        
    Examples
    --------
    >>> from peak_detection import find_valleys_in_data
    >>> valleys = find_valleys_in_data(data, 'symmetric_smoothed')
    >>> 
    >>> # Plot only valleys 0 and 2
    >>> groups = plot_valley_tracking_selective(valleys, data, valley_indices_to_plot=[0, 2])
    >>>
    >>> # Plot only valleys in voltage range -2.0 to -1.5 V
    >>> groups = plot_valley_tracking_selective(valleys, data, position_range=(-2.0, -1.5))
    """
    return plot_feature_tracking_selective(valleys_data, all_runs_data, 'valleys',
                                          position_tolerance, valley_indices_to_plot,
                                          position_range, figsize)
