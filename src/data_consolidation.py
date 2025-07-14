"""
Data Consolidation Module for Coulomb Drag Analysis

This module provides unified data loading and processing functions to eliminate
code duplication across notebooks and provide consistent data handling.

Functions:
- load_all_run_pairs(): Load and process all run pairs for a given dataset
- load_single_run_pair(): Load and process a single run pair
- get_dataset_info(): Get information about available datasets
- validate_run_data(): Validate loaded run data consistency
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter
from src.data_load import load_2d_data, get_gate1_lock1
from typing import Dict, List, Optional, Union


# Dataset configurations
DATASET_CONFIGS = {
    'VA204G_5um_base': {
        'run1_list': [80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102],
        'run2_list': [104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126],
        'current': [2.02, 5.25, 7.6, 10.05, 12.22, 15.05, 18.06, 20.13, 22.3, 25.01, 28.1, 30.43],
        'description': 'Original 12-current dataset'
    },
    'VA204G_5um_100mK': {
        'run1_list': [609, 613, 615, 619, 621, 623, 625, 627, 629, 631, 633, 635, 637, 639, 641, 643, 645, 647],
        'run2_list': [649, 651, 653, 655, 657, 659, 661, 663, 665, 667, 669, 671, 673, 675, 677, 679, 681, 683],
        'current': [2.02, 5.39, 6.53, 7.78, 9.03, 10.3, 11.5, 12.8, 14, 15.4, 16.7, 18.1, 19.4, 20.8, 22.1, 24.8, 27.5, 30.3],
        'description': '100mK 18-current dataset'
    }
}


def get_dataset_info(dataset_name: Optional[str] = None) -> Union[Dict, List[str]]:
    """
    Get information about available datasets
    
    Parameters
    ----------
    dataset_name : str, optional
        Name of specific dataset to get info for. If None, returns list of available datasets.
        
    Returns
    -------
    dict or list
        Dataset configuration dictionary or list of available dataset names
        
    Examples
    --------
    >>> # Get all available datasets
    >>> datasets = get_dataset_info()
    >>> print(datasets)
    ['VA204G_5um_old', 'VA204G_5um_100mK']
    
    >>> # Get specific dataset info
    >>> config = get_dataset_info('VA204G_5um_100mK')
    >>> print(f"Dataset has {len(config['current'])} current values")
    """
    if dataset_name is None:
        return list(DATASET_CONFIGS.keys())
    
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASET_CONFIGS.keys())}")
    
    return DATASET_CONFIGS[dataset_name].copy()


def validate_run_data(data: Dict, run1: int, run2: int) -> bool:
    """
    Validate that loaded run data is consistent
    
    Parameters
    ----------
    data : dict
        Loaded run data dictionary
    run1, run2 : int
        Run numbers for validation
        
    Returns
    -------
    bool
        True if data is valid, raises exception otherwise
    """
    required_keys = ['current', 'run1', 'run2', 'gate1', 'lock1_1', 'lock1_2', 
                     'symmetric', 'antisymmetric']
    
    # Check required keys
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")
    
    # Validate run numbers
    if data['run1'] != run1 or data['run2'] != run2:
        raise ValueError(f"Run number mismatch: expected {run1}, {run2}, got {data['run1']}, {data['run2']}")
    
    # Validate array shapes
    gate1_len = len(data['gate1'])
    for key in ['lock1_1', 'lock1_2', 'symmetric', 'antisymmetric']:
        if len(data[key]) != gate1_len:
            raise ValueError(f"Array length mismatch for {key}: expected {gate1_len}, got {len(data[key])}")
    
    return True


def load_single_run_pair(path: str, run1: int, run2: int, current_val: float, 
                        fixed_gate3_val: float, smoothing_sigma: float = 0.7) -> Dict:
    """
    Load and process a single run pair
    
    Parameters
    ----------
    path : str
        Path to data directory
    run1, run2 : int
        Run numbers to load
    current_val : float
        Current value for this run pair
    fixed_gate3_val : float
        Fixed gate3 value for horizontal linecut extraction
    smoothing_sigma : float
        Sigma parameter for Gaussian smoothing
        
    Returns
    -------
    dict
        Dictionary containing processed data for this run pair
        
    Examples
    --------
    >>> data = load_single_run_pair('/path/to/data', 609, 649, 2.02, -4.15)
    >>> print(f"Loaded data for current {data['current']}nA")
    >>> print(f"Gate1 range: {data['gate1'].min():.2f} to {data['gate1'].max():.2f}V")
    """
    # Construct file paths
    filepath1 = os.path.join(path, f"VA204G_5um_{run1}.dat")
    filepath2 = os.path.join(path, f"VA204G_5um_{run2}.dat")
    
    # Load 2D data files
    file_1 = load_2d_data(filepath1)
    file_2 = load_2d_data(filepath2)
    
    # Extract horizontal linecuts at fixed gate3 value
    gate1, lock1_1 = get_gate1_lock1(file_1, fixed_gate3_val)
    gate1_check, lock1_2 = get_gate1_lock1(file_2, fixed_gate3_val)
    
    # Validate gate1 arrays are identical
    if not np.allclose(gate1, gate1_check):
        raise ValueError(f"Gate1 arrays don't match for runs {run1}, {run2}")
    
    # Calculate symmetric and antisymmetric components
    symmetric = (lock1_1 + lock1_2) / 2.0
    antisymmetric = (lock1_1 - lock1_2) / 2.0
    
    # Apply smoothing
    smoothed_1 = gaussian_filter(lock1_1, sigma=smoothing_sigma)
    smoothed_2 = gaussian_filter(lock1_2, sigma=smoothing_sigma)
    symmetric_smoothed = gaussian_filter(symmetric, sigma=smoothing_sigma)
    antisymmetric_smoothed = gaussian_filter(antisymmetric, sigma=smoothing_sigma)
    
    # Create data dictionary
    data = {
        'current': current_val,
        'run1': run1,
        'run2': run2,
        'gate1': gate1,
        'lock1_1': lock1_1,
        'lock1_2': lock1_2,
        'symmetric': symmetric,
        'antisymmetric': antisymmetric,
        'smoothed_1': smoothed_1,
        'smoothed_2': smoothed_2,
        'symmetric_smoothed': symmetric_smoothed,
        'antisymmetric_smoothed': antisymmetric_smoothed,
        'fixed_gate3_val': fixed_gate3_val,
        'smoothing_sigma': smoothing_sigma
    }
    
    # Validate the data
    validate_run_data(data, run1, run2)
    
    return data


def load_all_run_pairs(dataset_name: str, path: str, fixed_gate3_val: float,
                      smoothing_sigma: float = 0.7, verbose: bool = True) -> Dict[int, Dict]:
    """
    Load and process all run pairs for a given dataset
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset configuration to use
    path : str
        Path to data directory
    fixed_gate3_val : float
        Fixed gate3 value for horizontal linecut extraction
    smoothing_sigma : float
        Sigma parameter for Gaussian smoothing
    verbose : bool
        Whether to print progress information
        
    Returns
    -------
    dict
        Dictionary mapping run pair indices to processed data dictionaries
        
    Examples
    --------
    >>> # Load the modern 100mK dataset
    >>> all_data = load_all_run_pairs('VA204G_5um_100mK', '/path/to/data', -4.15)
    >>> print(f"Loaded {len(all_data)} run pairs")
    >>> 
    >>> # Access specific run pair
    >>> first_run = all_data[0]
    >>> print(f"First run: {first_run['current']}nA")
    """
    # Get dataset configuration
    config = get_dataset_info(dataset_name)
    run1_list = config['run1_list']
    run2_list = config['run2_list']
    current_list = config['current']
    
    if verbose:
        print(f"Loading dataset: {dataset_name}")
        print(f"Description: {config['description']}")
        print(f"Number of run pairs: {len(run1_list)}")
        print(f"Current range: {min(current_list):.2f} to {max(current_list):.2f} nA")
        print(f"Fixed Gate3 value: {fixed_gate3_val}V")
        print(f"Smoothing sigma: {smoothing_sigma}")
    
    # Load all run pairs
    all_data = {}
    for i, (run1, run2, current_val) in enumerate(zip(run1_list, run2_list, current_list)):
        if verbose:
            print(f"Loading run pair {i+1}/{len(run1_list)}: {run1}, {run2} ({current_val}nA)")
        
        try:
            data = load_single_run_pair(path, run1, run2, current_val, 
                                      fixed_gate3_val, smoothing_sigma)
            all_data[i] = data
        except Exception as e:
            print(f"Error loading run pair {run1}, {run2}: {e}")
            continue
    
    if verbose:
        print(f"Successfully loaded {len(all_data)}/{len(run1_list)} run pairs")
    
    return all_data


def get_data_summary(all_data: Dict[int, Dict]) -> Dict:
    """
    Get summary statistics for loaded dataset
    
    Parameters
    ----------
    all_data : dict
        Dictionary of loaded run pair data
        
    Returns
    -------
    dict
        Summary statistics dictionary
        
    Examples
    --------
    >>> summary = get_data_summary(all_data)
    >>> print(f"Dataset contains {summary['num_runs']} run pairs")
    >>> print(f"Current range: {summary['current_range'][0]:.2f} to {summary['current_range'][1]:.2f} nA")
    """
    if not all_data:
        return {'num_runs': 0, 'error': 'No data loaded'}
    
    currents = [data['current'] for data in all_data.values()]
    gate1_ranges = [(data['gate1'].min(), data['gate1'].max()) for data in all_data.values()]
    
    summary = {
        'num_runs': len(all_data),
        'current_range': (min(currents), max(currents)),
        'current_values': sorted(currents),
        'gate1_range': (min(r[0] for r in gate1_ranges), max(r[1] for r in gate1_ranges)),
        'fixed_gate3_val': all_data[0]['fixed_gate3_val'],
        'smoothing_sigma': all_data[0]['smoothing_sigma'],
        'available_components': [
            'lock1_1', 'lock1_2', 'symmetric', 'antisymmetric',
            'smoothed_1', 'smoothed_2', 'symmetric_smoothed', 'antisymmetric_smoothed'
        ]
    }
    
    return summary


def print_data_summary(all_data: Dict[int, Dict]):
    """
    Print a formatted summary of loaded dataset
    
    Parameters
    ----------
    all_data : dict
        Dictionary of loaded run pair data
        
    Examples
    --------
    >>> print_data_summary(all_data)
    Dataset Summary:
    ===============
    Number of run pairs: 18
    Current range: 2.02 to 30.30 nA
    Gate1 range: -4.50 to -3.50 V
    Fixed Gate3 value: -4.15 V
    Smoothing sigma: 0.7
    Available components: 8
    """
    summary = get_data_summary(all_data)
    
    if 'error' in summary:
        print(f"Error: {summary['error']}")
        return
    
    print("Dataset Summary:")
    print("=" * 15)
    print(f"Number of run pairs: {summary['num_runs']}")
    print(f"Current range: {summary['current_range'][0]:.2f} to {summary['current_range'][1]:.2f} nA")
    print(f"Gate1 range: {summary['gate1_range'][0]:.2f} to {summary['gate1_range'][1]:.2f} V")
    print(f"Fixed Gate3 value: {summary['fixed_gate3_val']} V")
    print(f"Smoothing sigma: {summary['smoothing_sigma']}")
    print(f"Available components: {len(summary['available_components'])}")
    
    print("\nCurrent values:")
    for i, current in enumerate(summary['current_values']):
        print(f"  {i}: {current:.2f} nA")
