from src.data import DatFile, Data2D
from typing import Tuple, Dict, Iterable
import numpy as np

COLUMN_ALIASES: Dict[str, list] = {
    'x': ['Gate 1 V meas', 'gate 1 V meas'],
    'y': ['Gate 3 V meas', 'gate 3 V meas'],
    'z': ['Lockin 1 X raw', 'sr860 x raw', 'DMM1 x raw'],
    'z_current': ['sr830 y raw'],
    'x_order': ['x_parameter (gate_1)', 'x (Left Gate Scan)', 'x (Right Gate Scan)', 'x_parameter'],
    'y_order': ['y_parameter (gate_2)', 'y (Left Gate Scan)', 'y (Right Gate scan)', 'y_parameter'],
    'current': ['sr830 y raw', 'Lockin 1 Y raw'],
}

def _get_gate_coordinates(file_data: Data2D) -> Tuple[np.ndarray, np.ndarray]:
    """
    Automatically detect the correct orientation for gate coordinates.
    
    Args:
        file_data: Data2D instance containing 2D arrays.
        
    Returns:
        Tuple[gate1_values, gate3_values]: 1D arrays of unique gate values.
    """
    # Try row-wise extraction first (standard case)
    gate1_row = file_data.x[0, :]
    gate3_row = file_data.y[:, 0]
    
    # Try column-wise extraction
    gate1_col = file_data.x[:, 0] 
    gate3_col = file_data.y[0, :]
    
    # Check which orientation has more unique values (indicates proper scanning direction)
    row_unique_ratio = (len(np.unique(gate1_row)) + len(np.unique(gate3_row))) / (len(gate1_row) + len(gate3_row))
    col_unique_ratio = (len(np.unique(gate1_col)) + len(np.unique(gate3_col))) / (len(gate1_col) + len(gate3_col))
    
    if row_unique_ratio >= col_unique_ratio:
        return gate1_row, gate3_row
    else:
        return gate1_col, gate3_col

def load_2d_data(filename: str, use_current: bool = False) -> Data2D:
    """

    Load 2D experimental data from a .dat file.

    Args:
        filename: Path to the input .dat file.
        use_current: If True, load current channel ('sr830 y raw') for z axis.

    Returns:
        Data2D: Structured 2D data container with x, y, z arrays.
    """
    df = DatFile(filename)
    cols = df.df.columns.tolist()
    # helper to pick actual column names ignoring case
    def pick_column(aliases: list) -> str:
        for alias in aliases:
            for col in cols:
                if col.lower() == alias.lower():
                    return col
        raise KeyError(f"None of {aliases} found in columns {cols}")

    x = pick_column(COLUMN_ALIASES['x'])
    y = pick_column(COLUMN_ALIASES['y'])
    # choose z-axis based on use_current flag
    z_aliases = COLUMN_ALIASES['current'] if use_current else COLUMN_ALIASES['z']
    z = pick_column(z_aliases)
    x_order = pick_column(COLUMN_ALIASES['x_order'])
    y_order = pick_column(COLUMN_ALIASES['y_order'])
    return df.get_data(x, y, z, x_order, y_order)


def get_grids(file_data: Data2D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract meshgrid arrays from Data2D.

    Args:
        file_data: Data2D instance containing 2D arrays.

    Returns:
        Tuple[x_grid, y_grid, z_values]: Numpy arrays of gate1, gate3, and lockin measurements.
    """
    return file_data.x, file_data.y, file_data.z
 

def get_gate1_lock1(file_data: Data2D, fixed_gate3: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract lock-in trace for varying gate1 at a fixed gate3 value.

    Args:
        file_data: Data2D instance.
        fixed_gate3: The gate3 value at which to fix the cross-section.

    Returns:
        Tuple[gate1_values, lockin_values] arrays along gate1.
    """
    gate1, gate3 = _get_gate_coordinates(file_data)
    idx = np.argmin(np.abs(gate3 - fixed_gate3))
    
    # Determine if we need to slice along rows or columns based on coordinate orientation
    if np.array_equal(gate1, file_data.x[0, :]):
        # Standard orientation: gate1 along columns, gate3 along rows
        lock1 = file_data.z[idx, :]
    else:
        # Alternative orientation: gate1 along rows, gate3 along columns
        lock1 = file_data.z[:, idx]
    
    return gate1, lock1
 
def get_gate3_lock1(file_data: Data2D, fixed_gate1: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract lock-in trace for varying gate3 at a fixed gate1 value.

    Args:
        file_data: Data2D instance.
        fixed_gate1: The gate1 value at which to fix the cross-section.

    Returns:
        Tuple[gate3_values, lockin_values] arrays along gate3.
    """
    gate1, gate3 = _get_gate_coordinates(file_data)
    idx = np.argmin(np.abs(gate1 - fixed_gate1))
    
    # Determine if we need to slice along rows or columns based on coordinate orientation
    if np.array_equal(gate1, file_data.x[0, :]):
        # Standard orientation: gate1 along columns, gate3 along rows
        lock1 = file_data.z[:, idx]
    else:
        # Alternative orientation: gate1 along rows, gate3 along columns
        lock1 = file_data.z[idx, :]
    
    return gate3, lock1
 
def get_lock1(file_data: Data2D, fixed_gate1: float, fixed_gate3: float) -> float:
    """
    Retrieve a single lock-in value at specified gate1 and gate3.

    Args:
        file_data: Data2D instance.
        fixed_gate1: Gate1 coordinate to fix.
        fixed_gate3: Gate3 coordinate to fix.

    Returns:
        float: Lockin reading at the nearest grid point.
    """
    gate1, gate3 = _get_gate_coordinates(file_data)
    i3 = np.argmin(np.abs(gate3 - fixed_gate3))
    i1 = np.argmin(np.abs(gate1 - fixed_gate1))
    
    # Determine indexing based on coordinate orientation
    if np.array_equal(gate1, file_data.x[0, :]):
        # Standard orientation: gate1 along columns, gate3 along rows
        return float(file_data.z[i3, i1])
    else:
        # Alternative orientation: gate1 along rows, gate3 along columns
        return float(file_data.z[i1, i3])
 
def many_lock1_values(
    file_data: Data2D,
    gate1_values: Iterable[float],
    gate3_values: Iterable[float]
) -> Dict[Tuple[float, float], float]:
    """
    Compute lock-in values over multiple gate1 and gate3 combinations.

    Args:
        file_data: Data2D instance.
        gate1_values: Sequence of gate1 values.
        gate3_values: Sequence of gate3 values.

    Returns:
        Dict mapping (gate1, gate3) pairs to lock-in float values.
    """
    results: Dict[Tuple[float, float], float] = {}
    for g1 in gate1_values:
        for g3 in gate3_values:
            results[(g1, g3)] = get_lock1(file_data, g1, g3)
    return results
