"""
Module for analyzing symmetric and anti-symmetric components in Coulomb drag measurements.

This module provides functions to calculate and visualize the symmetric and anti-symmetric
parts of drag voltage measurements from two datasets with opposite current directions.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from typing import Tuple, Optional
from src.data import DatFile, Data2D

# Set default font size for plots
plt.rcParams.update({'font.size': 18})


def get_symmetries(lockin1_2D_1: np.ndarray, lockin1_2D_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate symmetric and anti-symmetric components from two 2D datasets.
    
    For two measurements with opposite current directions, the symmetric part
    represents the common (background) signal, while the anti-symmetric part
    represents the actual drag signal.
    
    Parameters
    ----------
    lockin1_2D_1 : np.ndarray
        2D lock-in data for current in one direction (e.g., left to right)
    lockin1_2D_2 : np.ndarray
        2D lock-in data for current in opposite direction (e.g., right to left)
        
    Returns
    -------
    tuple
        (symmetric_component, anti_symmetric_component)
        symmetric_component : Average of both measurements
        anti_symmetric_component : Half the difference between measurements
    """
    if lockin1_2D_1.shape != lockin1_2D_2.shape:
        raise ValueError("Input arrays must have the same shape")
        
    sym_lockin = (lockin1_2D_1 + lockin1_2D_2) / 2.0
    antisym_lockin = (lockin1_2D_1 - lockin1_2D_2) / 2.0
    
    return sym_lockin, antisym_lockin

def get_sym_antisym_ratio(lockin1_2D_1: np.ndarray, lockin1_2D_2: np.ndarray,
                         epsilon: float = 1e-10) -> np.ndarray:
    """
    Calculate the ratio of symmetric to anti-symmetric components.
    
    This ratio can help identify regions where the symmetric (background) 
    signal dominates over the anti-symmetric (drag) signal.
    
    Parameters
    ----------
    lockin1_2D_1 : np.ndarray
        2D lock-in data for current in one direction
    lockin1_2D_2 : np.ndarray
        2D lock-in data for current in opposite direction
    epsilon : float, optional
        Small value to avoid division by zero (default: 1e-10)
        
    Returns
    -------
    np.ndarray
        Absolute ratio of symmetric to anti-symmetric components
    """
    if lockin1_2D_1.shape != lockin1_2D_2.shape:
        raise ValueError("Input arrays must have the same shape")
    
    sym_lockin, antisym_lockin = get_symmetries(lockin1_2D_1, lockin1_2D_2)
    
    # Avoid division by zero by adding small epsilon
    antisym_safe = np.where(np.abs(antisym_lockin) < epsilon, 
                           np.sign(antisym_lockin) * epsilon, 
                           antisym_lockin)
    
    ratio = np.abs(sym_lockin / antisym_safe)
    return ratio


def plot_symmetries(lockin1_2D_1: np.ndarray, lockin1_2D_2: np.ndarray, 
                   gate1_grid: np.ndarray, gate3_grid: np.ndarray,
                   scale_factor: float = 50.0, figsize: Tuple[int, int] = (15, 6),
                   save_path: Optional[str] = None) -> mpl.figure.Figure:
    """
    Plot symmetric and anti-symmetric components side by side.
    
    Parameters
    ----------
    lockin1_2D_1 : np.ndarray
        2D lock-in data for current in one direction
    lockin1_2D_2 : np.ndarray
        2D lock-in data for current in opposite direction
    gate1_grid : np.ndarray
        2D grid of gate 1 voltages
    gate3_grid : np.ndarray
        2D grid of gate 3 voltages
    scale_factor : float, optional
        Factor to scale color limits (default: 50.0)
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Calculate symmetric and anti-symmetric components
    sym_lockin, antisym_lockin = get_symmetries(lockin1_2D_1, lockin1_2D_2)

    # Calculate color scale parameters
    sym_min, sym_max = np.min(sym_lockin), np.max(sym_lockin)
    antisym_min, antisym_max = np.min(antisym_lockin), np.max(antisym_lockin)
    
    # Overall limits for consistent scaling
    v_min = min(sym_min, antisym_min)
    v_max = max(sym_max, antisym_max)
    v_abs = max(abs(v_min), abs(v_max))
    
    # Calculate ratios for anti-symmetric colormap truncation
    v_max_ratio = (antisym_max + v_abs) / (2 * v_abs)
    v_min_ratio = (antisym_min + v_abs) / (2 * v_abs)

    # Create truncated colormap for anti-symmetric plot
    cmap_anti_full = mpl.colormaps['seismic']
    cmap_anti = _truncate_colormap(cmap_anti_full, minval=v_min_ratio, 
                                  maxval=v_max_ratio, n=100)

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    
    # Plot symmetric component
    img1 = ax1.pcolormesh(gate1_grid, gate3_grid, sym_lockin, 
                          cmap="seismic", shading="nearest",
                          vmin=-v_abs/scale_factor, vmax=v_abs/scale_factor)

    ax1.set_xlabel("Gate 3 (V)")
    ax1.set_ylabel("Gate 1 (V)")
    ax1.set_title("Symmetric Component")
    ax1.set_aspect("auto")

    # Add colorbar for symmetric plot
    cbar_ax1 = fig.add_axes([0.13, -0.08, 0.35, 0.05])
    cbar1 = fig.colorbar(img1, cax=cbar_ax1, orientation="horizontal")
    cbar1.set_label(r"$V_{\mathrm{drag}}$ ($\mu$V)", fontsize=18)

    # Plot anti-symmetric component
    img2 = ax2.pcolormesh(gate1_grid, gate3_grid, antisym_lockin, 
                          cmap=cmap_anti, shading="nearest", 
                          vmin=antisym_min/scale_factor, vmax=antisym_max/scale_factor)
    
    ax2.set_xlabel("Gate 3 (V)")
    ax2.set_title("Anti-symmetric Component")
    ax2.set_aspect("auto")

    # Add colorbar for anti-symmetric plot
    cbar_ax2 = fig.add_axes([0.55, -0.08, 0.35, 0.05])
    cbar2 = fig.colorbar(img2, cax=cbar_ax2, orientation="horizontal")
    cbar2.set_label(r"$V_{\mathrm{drag}}$ ($\mu$V)", fontsize=18)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig

def plot_symmetries_ratio(lockin1_2D_1: np.ndarray, lockin1_2D_2: np.ndarray, 
                         gate1_grid: np.ndarray, gate3_grid: np.ndarray,
                         ratio_vmax: float = 2.0, scale_factor: float = 1.0,
                         figsize: Tuple[int, int] = (15, 6),
                         save_path: Optional[str] = None) -> mpl.figure.Figure:
    """
    Plot the symmetric/anti-symmetric ratio alongside the anti-symmetric component.
    
    This visualization helps identify regions where background signals dominate
    versus regions with significant drag signals.
    
    Parameters
    ----------
    lockin1_2D_1 : np.ndarray
        2D lock-in data for current in one direction
    lockin1_2D_2 : np.ndarray
        2D lock-in data for current in opposite direction
    gate1_grid : np.ndarray
        2D grid of gate 1 voltages
    gate3_grid : np.ndarray
        2D grid of gate 3 voltages
    ratio_vmax : float, optional
        Maximum value for ratio colorscale (default: 2.0)
    scale_factor : float, optional
        Scale factor for anti-symmetric component display (default: 1.0)
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Calculate components and ratio
    sym_lockin, antisym_lockin = get_symmetries(lockin1_2D_1, lockin1_2D_2)
    ratio = get_sym_antisym_ratio(lockin1_2D_1, lockin1_2D_2)
    
    # Calculate color scale parameters for anti-symmetric component
    antisym_abs_max = np.max(np.abs(antisym_lockin))
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    
    # Plot ratio (left subplot)
    img1 = ax1.pcolormesh(gate1_grid, gate3_grid, ratio, 
                          cmap="viridis", shading="nearest",
                          vmin=0, vmax=ratio_vmax)

    ax1.set_xlabel("Gate 3 (V)", fontsize=18)
    ax1.set_ylabel("Gate 1 (V)", fontsize=18)
    ax1.set_title("Symmetric/Anti-symmetric Ratio", fontsize=18)
    ax1.set_aspect("auto")

    # Add colorbar for ratio plot
    cbar_ax1 = fig.add_axes([0.13, -0.08, 0.35, 0.05])
    cbar1 = fig.colorbar(img1, cax=cbar_ax1, orientation="horizontal")
    cbar1.set_label("Ratio", fontsize=18)

    # Plot anti-symmetric component (right subplot)
    img2 = ax2.pcolormesh(gate1_grid, gate3_grid, antisym_lockin, 
                          cmap="seismic", shading="nearest", 
                          vmin=-antisym_abs_max/scale_factor, 
                          vmax=antisym_abs_max/scale_factor)
    
    ax2.set_xlabel("Gate 3 (V)", fontsize=18)
    ax2.set_title("Anti-symmetric Component", fontsize=18)
    ax2.set_aspect("auto")

    # Add colorbar for anti-symmetric plot
    cbar_ax2 = fig.add_axes([0.55, -0.08, 0.35, 0.05])
    cbar2 = fig.colorbar(img2, cax=cbar_ax2, orientation="horizontal")
    cbar2.set_label(r"$V_{\mathrm{drag}}$ ($\mu$V)", fontsize=18)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig
    
    
def _truncate_colormap(cmap: mpl.colors.Colormap, minval: float = 0.0, 
                      maxval: float = 1.0, n: int = 100) -> mpl.colors.LinearSegmentedColormap:
    """
    Create a truncated version of a colormap.
    
    This is useful for adjusting the color range to better represent
    the data distribution in the anti-symmetric component.
    
    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        Input colormap to truncate
    minval : float, optional
        Minimum value for truncation (0.0 to 1.0)
    maxval : float, optional
        Maximum value for truncation (0.0 to 1.0)
    n : int, optional
        Number of color segments in new colormap
        
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Truncated colormap
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def analyze_symmetry_from_files(file1: str, file2: str, 
                               data_attr: str = 'z') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load two data files and extract symmetric/anti-symmetric components.
    
    Parameters
    ----------
    file1 : str
        Path to first data file (e.g., positive current direction)
    file2 : str
        Path to second data file (e.g., negative current direction)
    data_attr : str, optional
        Attribute name for the measurement data (default: 'z')
        
    Returns
    -------
    tuple
        (symmetric_component, anti_symmetric_component, gate1_grid, gate3_grid)
    """
    # Load data files
    data1 = DatFile(file1).get_data()
    data2 = DatFile(file2).get_data()
    
    # Extract measurement data
    lockin1 = getattr(data1, data_attr)
    lockin2 = getattr(data2, data_attr)
    
    # Calculate symmetries
    sym_component, antisym_component = get_symmetries(lockin1, lockin2)
    
    return sym_component, antisym_component, data1.x, data1.y


def get_symmetry_statistics(sym_component: np.ndarray, 
                           antisym_component: np.ndarray) -> dict:
    """
    Calculate statistical properties of symmetric and anti-symmetric components.
    
    Parameters
    ----------
    sym_component : np.ndarray
        Symmetric component data
    antisym_component : np.ndarray
        Anti-symmetric component data
        
    Returns
    -------
    dict
        Dictionary containing statistical measures
    """
    stats = {
        'symmetric': {
            'mean': np.mean(sym_component),
            'std': np.std(sym_component),
            'min': np.min(sym_component),
            'max': np.max(sym_component),
            'rms': np.sqrt(np.mean(sym_component**2))
        },
        'anti_symmetric': {
            'mean': np.mean(antisym_component),
            'std': np.std(antisym_component),
            'min': np.min(antisym_component),
            'max': np.max(antisym_component),
            'rms': np.sqrt(np.mean(antisym_component**2))
        }
    }
    
    # Calculate signal-to-noise ratio (antisym RMS / sym RMS)
    stats['signal_to_noise_ratio'] = (stats['anti_symmetric']['rms'] / 
                                     stats['symmetric']['rms'])
    
    return stats


def analyze_ratio_quality(lockin1_2D_1: np.ndarray, lockin1_2D_2: np.ndarray,
                         ratio_threshold: float = 1.0) -> dict:
    """
    Analyze the quality of the drag signal based on symmetric/anti-symmetric ratio.
    
    Lower ratios indicate regions where the drag signal is stronger relative
    to background noise/symmetric components.
    
    Parameters
    ----------
    lockin1_2D_1 : np.ndarray
        2D lock-in data for current in one direction
    lockin1_2D_2 : np.ndarray
        2D lock-in data for current in opposite direction
    ratio_threshold : float, optional
        Threshold below which the signal is considered "good quality" (default: 1.0)
        
    Returns
    -------
    dict
        Dictionary containing quality analysis results
    """
    ratio = get_sym_antisym_ratio(lockin1_2D_1, lockin1_2D_2)
    sym_lockin, antisym_lockin = get_symmetries(lockin1_2D_1, lockin1_2D_2)
    
    # Find good quality regions (low ratio)
    good_quality_mask = ratio < ratio_threshold
    good_quality_fraction = np.sum(good_quality_mask) / ratio.size
    
    # Statistics for different regions
    analysis = {
        'ratio_stats': {
            'mean': np.mean(ratio),
            'median': np.median(ratio),
            'min': np.min(ratio),
            'max': np.max(ratio),
            'std': np.std(ratio)
        },
        'good_quality_regions': {
            'fraction': good_quality_fraction,
            'total_pixels': np.sum(good_quality_mask),
            'threshold_used': ratio_threshold
        }
    }
    
    # Statistics for good quality regions only
    if np.any(good_quality_mask):
        analysis['good_quality_stats'] = {
            'antisym_mean': np.mean(antisym_lockin[good_quality_mask]),
            'antisym_std': np.std(antisym_lockin[good_quality_mask]),
            'antisym_max': np.max(np.abs(antisym_lockin[good_quality_mask])),
            'ratio_mean': np.mean(ratio[good_quality_mask]),
            'ratio_std': np.std(ratio[good_quality_mask])
        }
    else:
        analysis['good_quality_stats'] = None
        
    return analysis


