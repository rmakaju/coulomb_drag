"""
Peak Tracking Module for Coulomb Drag Data

This module contains advanced peak tracking and evolution analysis functions
for studying how peaks evolve with experimental parameters like drive current.

Functions:
- plot_peak_tracking_across_current(): Track peaks across different currents
- plot_peak_tracking_selective(): Advanced tracking with filtering options  
- plot_peak_evolution(): Visualize how peak characteristics evolve with current
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from peak_detection import analyze_peak_statistics


def plot_peak_tracking_across_current(peaks_data, all_runs_data, position_tolerance=0.05, figsize=(12, 8)):
    """
    Track individual peaks across different currents by grouping peaks at similar gate positions
    
    Parameters
    ----------
    peaks_data : dict
        Output from find_peaks_in_data
    all_runs_data : dict
        Data from all current measurements
    position_tolerance : float, optional
        Tolerance in V for grouping peaks at similar positions (default: 0.05)
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
    
    # Get data component info from the first entry
    if len(peaks_data) == 0:
        print("No peak data provided")
        return
        
    data_component = peaks_data[0]['data_type']
    
    # Collect all peak data with actual lockin values
    all_data = []
    for i, peak_info in peaks_data.items():
        # Get the actual lockin values at peak positions from all_runs_data
        current_data = all_runs_data[i]
        lockin_values = current_data[data_component] / 1e-6  # Convert to µV
        
        # Get lockin values at peak indices
        peak_lockin_values = lockin_values[peak_info['peak_indices']]
        
        for pos, lockin_val in zip(peak_info['peak_gate_positions'], peak_lockin_values):
            all_data.append({
                'current': peak_info['current'],
                'position': pos,
                'lockin_value': lockin_val
            })
    
    if len(all_data) == 0:
        print("No peaks found for tracking")
        return
    
    df = pd.DataFrame(all_data)
    
    # Group peaks by similar positions
    position_groups = []
    used_peaks = set()
    
    for idx, row in df.iterrows():
        if idx in used_peaks:
            continue
            
        # Find all peaks within tolerance of this position
        similar_peaks = df[abs(df['position'] - row['position']) <= position_tolerance]
        position_groups.append(similar_peaks)
        used_peaks.update(similar_peaks.index)
    
    # Plot tracked peaks
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(position_groups)))
    
    for i, group in enumerate(position_groups):
        if len(group) > 1:  # Only plot if peak appears in multiple currents
            mean_position = group['position'].mean()
            plt.plot(group['current'], group['lockin_value'], 'o-', 
                    color=colors[i], linewidth=2, markersize=8, alpha=0.8,
                    label=f'Peak ~{mean_position:.3f}V')
        else:
            # Single peak - plot as isolated point
            plt.scatter(group['current'], group['lockin_value'], 
                       s=50, alpha=0.5, color='gray')
    
    plt.xlabel('Current (nA)', fontsize=14)
    plt.ylabel(f'Peak {data_component} Value (µV)', fontsize=14)
    plt.title(f'{data_component.title()} Peak Values Evolution with Current (Tracked by Position)', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return position_groups


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
    
    # Get data component info from the first entry
    if len(peaks_data) == 0:
        print("No peak data provided")
        return
        
    data_component = peaks_data[0]['data_type']
    
    # Collect all peak data with actual lockin values
    all_data = []
    for i, peak_info in peaks_data.items():
        # Get the actual lockin values at peak positions from all_runs_data
        current_data = all_runs_data[i]
        lockin_values = current_data[data_component] / 1e-6  # Convert to µV
        
        # Get lockin values at peak indices
        peak_lockin_values = lockin_values[peak_info['peak_indices']]
        
        for pos, lockin_val in zip(peak_info['peak_gate_positions'], peak_lockin_values):
            all_data.append({
                'current': peak_info['current'],
                'position': pos,
                'lockin_value': lockin_val
            })
    
    if len(all_data) == 0:
        print("No peaks found for tracking")
        return
    
    df = pd.DataFrame(all_data)
    
    # Group peaks by similar positions
    position_groups = []
    used_peaks = set()
    
    for idx, row in df.iterrows():
        if idx in used_peaks:
            continue
            
        # Find all peaks within tolerance of this position
        similar_peaks = df[abs(df['position'] - row['position']) <= position_tolerance]
        position_groups.append(similar_peaks)
        used_peaks.update(similar_peaks.index)
    
    # Sort groups by mean position for consistent indexing
    position_groups.sort(key=lambda x: x['position'].mean())
    
    # Print available peaks for user reference BEFORE filtering
    print(f"All available peak groups (total: {len(position_groups)}):")
    for i, group in enumerate(position_groups):
        mean_pos = group['position'].mean()
        num_points = len(group)
        print(f"  Peak {i}: ~{mean_pos:.3f}V ({num_points} data points)")
    
    # Apply position range filter if specified
    if position_range is not None:
        min_pos, max_pos = position_range
        filtered_groups = [group for group in position_groups 
                          if min_pos <= group['position'].mean() <= max_pos]
        print(f"\nFiltered by position range {position_range}: {len(filtered_groups)} groups")
        position_groups = filtered_groups
    
    # Apply peak index filter if specified
    if peak_indices_to_plot is not None:
        original_groups = position_groups.copy()
        position_groups = []
        for i in peak_indices_to_plot:
            if i < len(original_groups):
                position_groups.append(original_groups[i])
            else:
                print(f"Warning: Peak index {i} is out of range (max: {len(original_groups)-1})")
        print(f"\nSelected peak indices {peak_indices_to_plot}: {len(position_groups)} groups")
    
    if len(position_groups) == 0:
        print("No peaks match the selection criteria")
        return position_groups
    
    # Plot tracked peaks
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(position_groups)))
    
    for i, group in enumerate(position_groups):
        if len(group) > 1:  # Only plot if peak appears in multiple currents
            mean_position = group['position'].mean()
            plt.plot(group['current'], group['lockin_value'], 'o-', 
                    color=colors[i], linewidth=2, markersize=8, alpha=0.8,
                    label=f'Peak: ~{mean_position:.3f}V')
        else:
            # Single peak - plot as isolated point
            mean_position = group['position'].mean()
            plt.scatter(group['current'], group['lockin_value'], 
                       s=50, alpha=0.5, color=colors[i], 
                       label=f'Peak: ~{mean_position:.3f}V (single)')
    
    plt.xlabel('Current (nA)', fontsize=14)
    plt.ylabel(f'Peak {data_component} Value (µV)', fontsize=14)
    
    # Update title based on selection
    title_suffix = ""
    if peak_indices_to_plot is not None:
        title_suffix += f" (Selected: {peak_indices_to_plot})"
    if position_range is not None:
        title_suffix += f" (Range: {position_range[0]:.3f}-{position_range[1]:.3f}V)"
    
    plt.title(f'{data_component.title()} Peak Evolution{title_suffix}', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return position_groups


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
