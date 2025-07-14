"""
Enhanced peak detection utilities and analysis tools.

This module provides additional utilities that complement the main peaks.py module,
including advanced filtering, interpolation, and visualization helpers.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


def interpolate_missing_data(
    x: np.ndarray, 
    y: np.ndarray, 
    z: np.ndarray, 
    method: str = 'cubic'
) -> np.ndarray:
    """
    Interpolate missing or NaN values in 2D data.
    
    Parameters
    ----------
    x, y : np.ndarray
        Coordinate arrays (2D meshgrids).
    z : np.ndarray
        Data array with potential missing values.
    method : str, optional
        Interpolation method ('linear', 'nearest', 'cubic').
        
    Returns
    -------
    np.ndarray
        Interpolated data array with missing values filled.
    """
    # Find valid (non-NaN) data points
    valid_mask = ~np.isnan(z)
    
    if not np.any(valid_mask):
        raise ValueError("No valid data points for interpolation")
    
    # Extract valid points
    valid_x = x[valid_mask].flatten()
    valid_y = y[valid_mask].flatten()
    valid_z = z[valid_mask].flatten()
    
    # Create target grid
    target_x = x.flatten()
    target_y = y.flatten()
    
    # Interpolate
    interpolated = griddata(
        (valid_x, valid_y), valid_z, 
        (target_x, target_y), 
        method=method, 
        fill_value=np.nan
    )
    
    return interpolated.reshape(z.shape)


def adaptive_smoothing(
    data: np.ndarray,
    noise_threshold: float = 0.1,
    max_sigma: float = 2.0
) -> np.ndarray:
    """
    Apply adaptive Gaussian smoothing based on local noise levels.
    
    Parameters
    ----------
    data : np.ndarray
        Input 2D data array.
    noise_threshold : float, optional
        Threshold for determining noise level (default: 0.1).
    max_sigma : float, optional
        Maximum smoothing sigma (default: 2.0).
        
    Returns
    -------
    np.ndarray
        Adaptively smoothed data.
    """
    # Estimate local noise using gradient magnitude
    grad_x, grad_y = np.gradient(data)
    noise_estimate = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize noise estimate
    noise_norm = noise_estimate / np.max(noise_estimate)
    
    # Calculate adaptive sigma
    sigma_map = max_sigma * (noise_norm > noise_threshold)
    
    # Apply smoothing with spatially varying sigma
    # For simplicity, use average sigma (could be improved with more sophisticated methods)
    avg_sigma = np.mean(sigma_map[sigma_map > 0]) if np.any(sigma_map > 0) else 0
    
    if avg_sigma > 0:
        return gaussian_filter(data, sigma=avg_sigma)
    else:
        return data.copy()


def peak_clustering(
    peak_positions: List[Tuple[float, float]],
    cluster_radius: float = 0.1
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Cluster nearby peaks based on spatial proximity.
    
    Parameters
    ----------
    peak_positions : List[Tuple[float, float]]
        List of (x, y) peak coordinates.
    cluster_radius : float, optional
        Maximum distance for peaks to be in same cluster (default: 0.1).
        
    Returns
    -------
    Dict[int, List[Tuple[float, float]]]
        Dictionary mapping cluster IDs to lists of peak positions.
    """
    if not peak_positions:
        return {}
    
    positions = np.array(peak_positions)
    n_peaks = len(positions)
    
    # Calculate distance matrix
    distances = np.sqrt(
        (positions[:, np.newaxis, 0] - positions[np.newaxis, :, 0])**2 +
        (positions[:, np.newaxis, 1] - positions[np.newaxis, :, 1])**2
    )
    
    # Simple clustering algorithm
    clusters = {}
    assigned = np.zeros(n_peaks, dtype=bool)
    cluster_id = 0
    
    for i in range(n_peaks):
        if assigned[i]:
            continue
            
        # Start new cluster
        cluster_members = [i]
        assigned[i] = True
        
        # Find all peaks within cluster radius
        nearby = np.where((distances[i] <= cluster_radius) & (~assigned))[0]
        
        for j in nearby:
            cluster_members.append(j)
            assigned[j] = True
        
        # Store cluster
        clusters[cluster_id] = [tuple(positions[idx]) for idx in cluster_members]
        cluster_id += 1
    
    return clusters


def visualize_peak_evolution(
    peak_data: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a comprehensive visualization of peak evolution data.
    
    Parameters
    ----------
    peak_data : Dict[str, Any]
        Peak evolution data from analyze_peak_evolution function.
    save_path : str, optional
        Path to save the figure.
    figsize : Tuple[int, int], optional
        Figure size (default: (12, 8)).
        
    Returns
    -------
    plt.Figure
        The created figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Peak Evolution Analysis', fontsize=16, fontweight='bold')
    
    # Extract data
    positions = peak_data.get('peak_positions', [])
    values = peak_data.get('peak_values', [])
    distances = peak_data.get('distances', [])
    file_numbers = peak_data.get('file_numbers', [])
    
    if not positions:
        fig.text(0.5, 0.5, 'No peak data available', 
                ha='center', va='center', fontsize=14)
        return fig
    
    positions = np.array(positions)
    
    # Plot 1: Peak trajectory
    ax1 = axes[0, 0]
    if len(positions) > 0:
        ax1.plot(positions[:, 0], positions[:, 1], 'bo-', linewidth=2, markersize=6)
        ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=100, 
                   marker='s', label='Start', zorder=5)
        ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, 
                   marker='X', label='End', zorder=5)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Peak Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Peak values over time
    ax2 = axes[0, 1]
    if values:
        ax2.plot(file_numbers, values, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('File Number')
        ax2.set_ylabel('Peak Value')
        ax2.set_title('Peak Value Evolution')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distances between consecutive peaks
    ax3 = axes[1, 0]
    if distances:
        ax3.plot(file_numbers[1:], distances, 'go-', linewidth=2, markersize=6)
        ax3.set_xlabel('File Number')
        ax3.set_ylabel('Distance from Previous')
        ax3.set_title('Peak Movement Distances')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics summary
    ax4 = axes[1, 1]
    stats = peak_data.get('statistics', {})
    if stats:
        stat_names = list(stats.keys())
        stat_values = list(stats.values())
        
        # Create a simple text display of statistics
        stat_text = '\n'.join([f'{name}: {value:.4f}' 
                              for name, value in zip(stat_names, stat_values)])
        ax4.text(0.1, 0.9, stat_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Statistics Summary')
        ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def export_peak_data(
    peak_data: Dict[str, Any],
    filename: str,
    format: str = 'csv'
) -> None:
    """
    Export peak analysis data to file.
    
    Parameters
    ----------
    peak_data : Dict[str, Any]
        Peak evolution data from analyze_peak_evolution function.
    filename : str
        Output filename.
    format : str, optional
        Export format ('csv', 'json', 'npz').
    """
    import pandas as pd
    import json
    
    if format.lower() == 'csv':
        # Create DataFrame
        data_dict = {
            'file_number': peak_data.get('file_numbers', []),
            'peak_x': [pos[0] if pos else np.nan for pos in peak_data.get('peak_positions', [])],
            'peak_y': [pos[1] if pos else np.nan for pos in peak_data.get('peak_positions', [])],
            'peak_value': peak_data.get('peak_values', [])
        }
        
        # Add distances (one less than number of files)
        distances = peak_data.get('distances', [])
        distance_col = [np.nan] + distances  # Pad with NaN for first file
        data_dict['distance_from_previous'] = distance_col[:len(data_dict['file_number'])]
        
        df = pd.DataFrame(data_dict)
        df.to_csv(filename, index=False)
        
    elif format.lower() == 'json':
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, value in peak_data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, list) and value and isinstance(value[0], tuple):
                json_data[key] = [list(item) for item in value]
            else:
                json_data[key] = value
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
            
    elif format.lower() == 'npz':
        # Save as NumPy compressed archive
        save_dict = {}
        for key, value in peak_data.items():
            if isinstance(value, (list, tuple)):
                save_dict[key] = np.array(value)
            else:
                save_dict[key] = value
        
        np.savez_compressed(filename, **save_dict)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Peak data exported to {filename} ({format.upper()} format)")
