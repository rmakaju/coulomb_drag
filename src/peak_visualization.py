"""
Peak Visualization Module for Coulomb Drag Data

This module contains visualization functions for displaying peak detection results
and basic plotting of data components in Coulomb drag measurements.

Functions:
- plot_components(): Plot selected data components for all current measurements
- plot_data_with_peaks(): Visualize data with detected peaks highlighted
- plot_peak_values_vs_positions(): Scatter plot of peak values vs gate positions
- plot_peaks_vs_current(): Plot peak positions and values as function of current
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_components(all_runs_data, components_to_plot=['symmetric'], gate3_val=-3.9):
    """
    Plot selected components for all runs horizontal linecuts
    
    Parameters
    ----------
    all_runs_data : dict
        Data from all current measurements
    components_to_plot : list of str, optional
        Components to plot. Options: 'lock1_1', 'lock1_2', 'symmetric', 'antisymmetric', 
        'smoothed_1', 'smoothed_2', 'symmetric_smoothed', 'antisymmetric_smoothed'
    gate3_val : float, optional
        Fixed gate3 value for the title (default: -3.9)
        
    Examples
    --------
    >>> # Plot symmetric component
    >>> plot_components(data, ['symmetric'])
    >>>
    >>> # Plot multiple components
    >>> plot_components(data, ['symmetric', 'antisymmetric'], gate3_val=-4.0)
    """
    
    # Define all possible plot configurations
    all_plot_configs = {
        'lock1_1': {'data_key': 'lock1_1', 'title': f'Original Data (Bottom to top) - Gate3 = {gate3_val}V', 'ylabel': '$V_{drag}(\mu V)$'},
        'lock1_2': {'data_key': 'lock1_2', 'title': f'Original Data (Top to bottom) - Gate3 = {gate3_val}V', 'ylabel': ' $V_{drag}(\mu V)$'},
        'symmetric': {'data_key': 'symmetric', 'title': f'Symmetric Component - Gate3 = {gate3_val}V', 'ylabel': 'Symmetric $V_{drag}(\mu V)$'},
        'antisymmetric': {'data_key': 'antisymmetric', 'title': f'Antisymmetric Component - Gate3 = {gate3_val}V', 'ylabel': 'Antisymmetric $V_{drag}(\mu V)$'},
        'smoothed_1': {'data_key': 'smoothed_1', 'title': f'Smoothed Data (Bottom to top, σ=0.7) - Gate3 = {gate3_val}V', 'ylabel': 'Smoothed $V_{drag}(\mu V)$'},
        'smoothed_2': {'data_key': 'smoothed_2', 'title': f'Smoothed Data (Top to bottom, σ=0.7) - Gate3 = {gate3_val}V', 'ylabel': 'Smoothed $V_{drag}(\mu V)$'},
        'symmetric_smoothed': {'data_key': 'symmetric_smoothed', 'title': f'Smoothed Symmetric Component (σ=0.7) - Gate3 = {gate3_val}V', 'ylabel': 'Smoothed Symmetric $V_{drag}(\mu V)$'},
        'antisymmetric_smoothed': {'data_key': 'antisymmetric_smoothed', 'title': f'Smoothed Antisymmetric Component (σ=0.7) - Gate3 = {gate3_val}V', 'ylabel': 'Smoothed Antisymmetric $V_{drag}(\mu V)$'}
    }
    
    # Create plots for selected components only
    for component in components_to_plot:
        if component not in all_plot_configs:
            print(f"Warning: '{component}' is not a valid component. Skipping.")
            continue
            
        plot_config = all_plot_configs[component]
        plt.figure(figsize=(15, 10))
        
        # Plot all current values on the same plot
        for i in range(len(all_runs_data)):
            data = all_runs_data[i]
            plt.plot(data['gate1'], data[plot_config['data_key']]/1e-6, 
                    label=f"$I_{{drive}} = {data['current']}nA $", 
                    alpha=0.8, linewidth=2)
        
        plt.xlabel("Gate 1 (V)", fontsize=14)
        plt.ylabel(plot_config['ylabel'], fontsize=14)
        plt.title(plot_config['title'], fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()


def plot_data_with_peaks(peaks_data, all_runs_data, fixed_gate3_val, show_all_currents=True, 
                        current_indices=None, position_tolerance=0.05, figsize=(15, 10)):
    """
    Plot any data component with detected peaks highlighted with consistent colors per peak cluster
    
    Parameters
    ----------
    peaks_data : dict
        Output from find_peaks_in_data
    all_runs_data : dict
        Data from all current measurements
    fixed_gate3_val : float
        Gate3 value for plot title
    show_all_currents : bool, optional
        Whether to show all currents on one plot (default: True)
    current_indices : list, optional
        List of indices to plot (if show_all_currents=False)
    position_tolerance : float, optional
        Tolerance in V for grouping peaks at similar positions (default: 0.05)
    figsize : tuple, optional
        Figure size (default: (15, 10))
        
    Examples
    --------
    >>> from peak_detection import find_peaks_in_data
    >>> peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> plot_data_with_peaks(peaks, data, -4.0)
    >>>
    >>> # Plot individual current traces
    >>> plot_data_with_peaks(peaks, data, -4.0, show_all_currents=False, 
    ...                      current_indices=[0, 1, 2])
    """
    
    # Get data component info from the first entry
    if len(peaks_data) == 0:
        print("No peak data provided")
        return
        
    data_component = peaks_data[0]['data_type']
    
    if show_all_currents:
        # Plot all currents on one figure
        plt.figure(figsize=figsize)
        
        # Plot all data traces
        for i in range(len(peaks_data)):
            peak_info = peaks_data[i]
            plt.plot(peak_info['gate1'], peak_info['y_data_uv'], 
                     label=f"$I_{{drive}} = {peak_info['current']}nA$", 
                     alpha=0.7, linewidth=2)
        
        # Collect all peaks across currents for clustering
        all_peaks = []
        for i, peak_info in peaks_data.items():
            for pos, height in zip(peak_info['peak_gate_positions'], peak_info['peak_heights']):
                all_peaks.append({'pos': pos, 'height': height, 'current': peak_info['current']})
        
        # Group peaks by similar positions
        position_groups = []
        used = set()
        for idx, pk in enumerate(all_peaks):
            if idx in used:
                continue
            group = [pk]
            used.add(idx)
            for jdx in range(idx+1, len(all_peaks)):
                if jdx not in used and abs(pk['pos'] - all_peaks[jdx]['pos']) <= position_tolerance:
                    group.append(all_peaks[jdx])
                    used.add(jdx)
            position_groups.append(group)
        
        # Sort groups by mean position
        position_groups.sort(key=lambda g: np.mean([p['pos'] for p in g]))
        
        # Plot clusters with consistent colors
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(position_groups))) if position_groups else []
        for cid, group in enumerate(position_groups):
            xs = [p['pos'] for p in group]
            ys = [p['height'] for p in group]
            plt.scatter(xs, ys,
                        color=[cluster_colors[cid]] * len(xs),
                        s=60, marker='o', zorder=5, alpha=0.8,
                        label=f'Peak #{cid+1}')
        
        plt.xlabel("Gate 1 (V)", fontsize=14)
        plt.ylabel(f'{data_component} $V_{{drag}}(\mu V)$', fontsize=14)
        plt.title(f'{data_component.title()} Component with Detected Peaks - Gate3 = {fixed_gate3_val}V', fontsize=16)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    else:
        # Plot individual currents
        if current_indices is None:
            current_indices = range(len(peaks_data))
        
        # First, determine the maximum number of peaks across all currents for consistent color assignment
        max_peaks = 0
        for i in range(len(peaks_data)):
            peak_info = peaks_data[i]
            max_peaks = max(max_peaks, len(peak_info['peak_gate_positions']))
        
        # Create colormap for different peak indices
        peak_colors = plt.cm.tab10(np.linspace(0, 1, max_peaks)) if max_peaks > 0 else []
            
        for idx in current_indices:
            peak_info = peaks_data[idx]
            
            plt.figure(figsize=(12, 6))
            
            # Plot the data
            plt.plot(peak_info['gate1'], peak_info['y_data_uv'], 
                    'b-', linewidth=2, label=f'{data_component} data')
            
            # Highlight peaks with consistent colors
            if len(peak_info['peak_gate_positions']) > 0:
                for peak_idx, (pos, height) in enumerate(zip(peak_info['peak_gate_positions'], peak_info['peak_heights'])):
                    color = peak_colors[peak_idx] if peak_idx < len(peak_colors) else 'red'
                    plt.scatter(pos, height, color=color, s=100, marker='o', zorder=5,
                               label=f'Peak {peak_idx+1}')
            
            plt.xlabel("Gate 1 (V)", fontsize=14)
            plt.ylabel(f'{data_component} $V_{{drag}}(\mu V)$', fontsize=14)
            plt.title(f'{data_component.title()} Component - $I_{{drive}} = {peak_info["current"]}nA$ - Gate3 = {fixed_gate3_val}V', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


def plot_peak_values_vs_positions(peaks_data, all_runs_data, figsize=(12, 8)):
    """
    Plot peak lockin values as a function of their gate positions
    Color-coded by peak index (peak #1, peak #2, etc.) rather than by current
    
    Parameters
    ----------
    peaks_data : dict
        Output from find_peaks_in_data
    all_runs_data : dict
        Data from all current measurements
    figsize : tuple, optional
        Figure size (default: (12, 8))
        
    Examples
    --------
    >>> from peak_detection import find_peaks_in_data
    >>> peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> plot_peak_values_vs_positions(peaks, data)
    """
    
    plt.figure(figsize=figsize)
    
    # Get data component info from the first entry
    if len(peaks_data) == 0:
        print("No peak data provided")
        return
        
    data_component = peaks_data[0]['data_type']
    
    # First, determine the maximum number of peaks across all currents
    max_peaks = 0
    for i, peak_info in peaks_data.items():
        max_peaks = max(max_peaks, len(peak_info['peak_gate_positions']))
    
    if max_peaks == 0:
        print("No peaks found to plot")
        return
    
    # Create colormap for different peak indices
    colors = plt.cm.tab10(np.linspace(0, 1, max_peaks))
    
    # Collect data for each peak index
    peak_index_data = {i: {'positions': [], 'lockin_values': [], 'currents': []} for i in range(max_peaks)}
    
    for i, peak_info in peaks_data.items():
        if len(peak_info['peak_gate_positions']) > 0:
            # Get the actual lockin values at peak positions from all_runs_data
            current_data = all_runs_data[i]
            lockin_values = current_data[data_component] / 1e-6  # Convert to µV
            
            # Get lockin values at peak indices
            peak_lockin_values = lockin_values[peak_info['peak_indices']]
            
            # Sort peaks by gate position to maintain consistent indexing
            sorted_indices = np.argsort(peak_info['peak_gate_positions'])
            sorted_positions = peak_info['peak_gate_positions'][sorted_indices]
            sorted_lockin_values = peak_lockin_values[sorted_indices]
            
            # Assign each peak to its index group
            for peak_idx, (pos, lockin_val) in enumerate(zip(sorted_positions, sorted_lockin_values)):
                if peak_idx < max_peaks:  # Safety check
                    peak_index_data[peak_idx]['positions'].append(pos)
                    peak_index_data[peak_idx]['lockin_values'].append(lockin_val)
                    peak_index_data[peak_idx]['currents'].append(peak_info['current'])
    
    # Plot each peak index group with a different color
    for peak_idx in range(max_peaks):
        data = peak_index_data[peak_idx]
        if len(data['positions']) > 0:
            plt.scatter(data['positions'], data['lockin_values'],
                       c=colors[peak_idx], s=80, alpha=0.7,
                       label=f'Peak #{peak_idx + 1}')
    
    plt.xlabel('Peak Gate Position (V)', fontsize=14)
    plt.ylabel(f'Peak {data_component} Value (µV)', fontsize=14)
    plt.title(f'{data_component.title()} Data: Peak Values vs Gate Positions\n(Color-coded by Peak Index)', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print summary of peak distribution
    print(f"\nPeak Index Distribution for {data_component}:")
    for peak_idx in range(max_peaks):
        count = len(peak_index_data[peak_idx]['positions'])
        if count > 0:
            print(f"Peak #{peak_idx + 1}: appears in {count} current measurements")
            avg_pos = np.mean(peak_index_data[peak_idx]['positions'])
            print(f"  Average position: {avg_pos:.3f}V")
            pos_range = np.max(peak_index_data[peak_idx]['positions']) - np.min(peak_index_data[peak_idx]['positions'])
            print(f"  Position range: {pos_range:.3f}V")


def plot_peaks_vs_current(peaks_data, all_runs_data, figsize=(14, 10)):
    """
    Plot each peak position and lockin value as a function of current
    
    Parameters
    ----------
    peaks_data : dict
        Output from find_peaks_in_data
    all_runs_data : dict
        Data from all current measurements
    figsize : tuple, optional
        Figure size (default: (14, 10))
        
    Examples
    --------
    >>> from peak_detection import find_peaks_in_data
    >>> peaks = find_peaks_in_data(data, 'symmetric_smoothed')
    >>> plot_peaks_vs_current(peaks, data)
    """
    
    # Get data component info from the first entry
    if len(peaks_data) == 0:
        print("No peak data provided")
        return
        
    data_component = peaks_data[0]['data_type']
    
    # Organize peaks by approximate position for tracking across currents
    all_peak_positions = []
    all_currents = []
    
    for i, peak_info in peaks_data.items():
        for pos in peak_info['peak_gate_positions']:
            all_peak_positions.append(pos)
            all_currents.append(peak_info['current'])
    
    if len(all_peak_positions) == 0:
        print("No peaks found to plot vs current")
        return
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Peak positions vs current
    # Use a better color scheme that provides distinct colors (not gradients)
    num_currents = len(peaks_data)
    
    # Create distinct colors by cycling through multiple colormaps
    if num_currents <= 10:
        # Use tab10 for small number of currents
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:num_currents]
    elif num_currents <= 20:
        # Combine tab10 and tab20 for up to 20 distinct colors
        colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
        colors2 = plt.cm.tab20(np.linspace(0, 1, 20))[10:]  # Take the second half of tab20
        colors = np.vstack([colors1, colors2[:num_currents-10]])
    else:
        # For more than 20, cycle through tab10, tab20, and Set3
        colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
        colors2 = plt.cm.tab20(np.linspace(0, 1, 20))[10:]  # Second half of tab20
        colors3 = plt.cm.Set3(np.linspace(0, 1, 12))
        
        # Cycle through the combined color sets
        all_colors = np.vstack([colors1, colors2, colors3])
        colors = []
        for i in range(num_currents):
            colors.append(all_colors[i % len(all_colors)])
        colors = np.array(colors)
    
    # Create a list to track which colors are used for legend
    legend_handles = []
    
    for i, peak_info in peaks_data.items():
        if len(peak_info['peak_gate_positions']) > 0:
            # Plot each peak position for this current
            scatter = ax1.scatter([peak_info['current']] * len(peak_info['peak_gate_positions']),
                                 peak_info['peak_gate_positions'],
                                 color=colors[i], s=80, alpha=0.9, 
                                 edgecolors='black', linewidth=0.5,
                                 label=f"$I_{{drive}} = {peak_info['current']}nA$")
            legend_handles.append(scatter)
    
    ax1.set_xlabel('Current (nA)', fontsize=12)
    ax1.set_ylabel('Peak Gate Position (V)', fontsize=12)
    ax1.set_title(f'{data_component.title()} Peak Positions vs Current', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Peak lockin values vs current
    for i, peak_info in peaks_data.items():
        if len(peak_info['peak_heights']) > 0:
            # Get the actual lockin values at peak positions from all_runs_data
            current_data = all_runs_data[i]
            lockin_values = current_data[data_component] / 1e-6  # Convert to µV
            
            # Get lockin values at peak indices
            peak_lockin_values = lockin_values[peak_info['peak_indices']]
            
            # Plot each peak lockin value for this current with the same color scheme
            ax2.scatter([peak_info['current']] * len(peak_lockin_values),
                       peak_lockin_values,
                       color=colors[i], s=80, alpha=0.9,
                       edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('Current (nA)', fontsize=12)
    ax2.set_ylabel(f'Peak {data_component} Value (µV)', fontsize=12)
    ax2.set_title(f'{data_component.title()} Peak Values vs Current', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
