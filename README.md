# Coulomb Drag Analysis

This repository contains Python tools and Jupyter notebooks for analyzing Coulomb drag experiments.

## Features
- Load and pivot raw `.dat` files via `data.py`
- **🆕 Unified analysis workflow** in `unified_coulomb_drag_analysis.ipynb`
- **🆕 Consolidated data loading** via `data_consolidation.py` with multi-dataset support
- **🆕 Optimized feature tracking** via `features_tracking.py` - **1098 lines, 40% reduction** from original
- **Modular peak analysis suite**:
  - `peak_detection.py` - Core peak detection and statistical analysis
  - `peak_visualization.py` - Plotting and visualization functions
  - `features_tracking.py` - **Universal feature tracking** for both peaks and valleys
- Advanced peak analysis utilities in `peak_utils.py` 
- Symmetric/anti-symmetric component analysis in `symmetries.py`
- Utilities for filtering, interpolation, gradient, and more in `data.py`
- Comprehensive test suite and performance benchmarks
- **Legacy notebooks** (see [Migration Guide](MIGRATION_GUIDE.md) for transition)

## Getting Started

### Prerequisites
- Python 3.11+ installed
- [uv](https://docs.astral.sh/uv/#projects) (Astral's uv CLI)

### Setup
```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Initialize uv project (if not already)
uv init .

# Install dependencies
uv sync
```

### Quick Start
```bash
# Run main analysis script
uv run python main.py

# Launch IPython REPL
uv run ipython

# Run tests to verify functionality
uv run python test_peaks_consolidation.py

# Run performance benchmarks
uv run python benchmark_peaks.py
```

## Recommended Workflow

### 1. Use the Unified Notebook (Recommended)
```python
# Open the comprehensive analysis notebook
jupyter notebook unified_coulomb_drag_analysis.ipynb
```

This notebook provides a complete analysis workflow with:
- **Unified data loading** with multi-dataset support
- **Comprehensive peak detection** across all data components
- **Valley detection** and analysis
- **Advanced tracking** of peaks/valleys across currents
- **Statistical analysis** and comparisons
- **Gate range analysis** for targeted studies
- **Export functionality** for results

### 2. Or Use the Modular Approach
```python
from src.data_consolidation import load_all_run_pairs
from src.peak_detection import find_peaks_in_data, find_valleys_in_data
from src.peak_visualization import plot_data_with_peaks
from src.features_tracking import (
    plot_feature_tracking_across_current,
    plot_feature_tracking_selective,
    plot_feature_tracking_all_files_only
)

# Load data
all_runs_data = load_all_run_pairs('VA204G_5um_100mK', '/path/to/data', -4.15)

# Find peaks and valleys
peaks = find_peaks_in_data(all_runs_data, 'symmetric', prominence=0.2)
valleys = find_valleys_in_data(all_runs_data, 'symmetric', prominence=0.2)

# Visualize results
plot_data_with_peaks(peaks, all_runs_data, -4.15)

# Universal feature tracking (recommended)
plot_feature_tracking_across_current(peaks, all_runs_data, 'peaks')
plot_feature_tracking_across_current(valleys, all_runs_data, 'valleys')

# Advanced selective tracking
plot_feature_tracking_selective(peaks, all_runs_data, 'peaks', 
                               feature_indices_to_plot=[0, 2])
plot_feature_tracking_all_files_only(peaks, all_runs_data, 'peaks')
```

### 3. Available Datasets
```python
from src.data_consolidation import get_dataset_info

# See available datasets
datasets = get_dataset_info()
print(datasets)
# ['VA204G_5um_old', 'VA204G_5um_100mK']

# Get dataset details
config = get_dataset_info('VA204G_5um_100mK')
print(f"Dataset has {len(config['current'])} current values")
```

## Migration from Legacy Notebooks

If you're using the old individual notebooks:
- `current_symmetries_204_5um.ipynb` 
- `current-symmetric-peaks.ipynb`
- `peaks_valleys_track_current.ipynb`

**→ See the [Migration Guide](MIGRATION_GUIDE.md) for step-by-step transition instructions.**

## Key Improvements

### Before (3 separate notebooks + duplicated tracking modules):
- ❌ Code duplication across notebooks
- ❌ **1800+ lines of duplicated tracking code**
- ❌ Hard-coded dataset configurations
- ❌ Inconsistent data loading
- ❌ Scattered analysis workflows

### After (unified approach):
- ✅ **60% reduction** in code duplication
- ✅ **~1100 lines eliminated** through unified tracking
- ✅ **Centralized dataset management**
- ✅ **Consistent data loading** with validation
- ✅ **Complete analysis workflow** in one place
- ✅ **Better error handling** and documentation
- ✅ **Easy to scale** to new datasets

## Universal Feature Tracking Architecture

The new `features_tracking.py` module provides a clean, unified architecture:

### Universal Functions (Recommended for New Code):
```python
# Track any feature type with a single function
plot_feature_tracking_across_current(data, all_runs, 'peaks')    # or 'valleys'
plot_feature_tracking_selective(data, all_runs, 'peaks', feature_indices_to_plot=[0, 2])
plot_feature_tracking_all_files_only(data, all_runs, 'valleys')
```

### Backward-Compatible Wrappers (For Existing Code):
```python
# Legacy functions still work
plot_peak_tracking_across_current(peaks, all_runs)
plot_valley_tracking_across_current(valleys, all_runs)
plot_peak_tracking_selective(peaks, all_runs, peak_indices_to_plot=[0, 2])
plot_valley_tracking_selective(valleys, all_runs, position_range=(-2.0, -1.5))
```

### Specialized Analysis Functions:
```python
# Advanced analysis capabilities
plot_peak_evolution(peaks)                    # Evolution of peak characteristics
analyze_peak_persistence(peaks)               # Persistence analysis
plot_peak_trajectories(peaks, all_runs)       # Position & amplitude trajectories
plot_combined_peaks_valleys_tracking(peaks, valleys, all_runs)  # Combined tracking
```

## File Structure

```
coulomb_drag_analysis/
├── src/                                     # 🆕 Source code modules
│   ├── features_tracking.py                # 🆕 Universal feature tracking (1098 lines)
│   ├── data_consolidation.py               # 🆕 Unified data loading
│   ├── peak_detection.py                   # Core peak detection
│   ├── peak_visualization.py               # Plotting functions
│   ├── peak_utils.py                       # Additional utilities
│   ├── data.py                             # Data loading utilities
│   ├── symmetries.py                       # Symmetric/antisymmetric analysis
│   └── main.py                             # Main analysis script
├── unified_coulomb_drag_analysis.ipynb     # 🆕 Main analysis notebook
├── MIGRATION_GUIDE.md                      # 🆕 Migration instructions
├── UNIFIED_TRACKING_GUIDE.md               # 🆕 Unified tracking guide
├── CONSOLIDATION_SUMMARY.md                # 🆕 Consolidation summary
├── test_peaks_consolidation.py             # Test suite
├── pyproject.toml                          # Python project configuration
└── legacy_notebooks/                       # Old notebooks (archived)
    ├── current_symmetries_204_5um.ipynb
    ├── current-symmetric-peaks.ipynb
    └── peaks_valleys_track_current.ipynb
```

## Module Architecture

### Core Analysis Modules:
- **`features_tracking.py`** - Universal feature tracking with 40% code reduction
- **`data_consolidation.py`** - Centralized data loading and management
- **`peak_detection.py`** - Core peak/valley detection algorithms
- **`peak_visualization.py`** - Plotting and visualization functions
- **`peak_utils.py`** - Additional analysis utilities
- **`symmetries.py`** - Symmetric/antisymmetric component analysis
- **`data.py`** - Low-level data loading and processing utilities

### Key Benefits:
- **Single source of truth** for feature tracking
- **Consistent API** across all tracking functions
- **Backward compatibility** with existing code
- **Comprehensive documentation** and examples
- **Easy maintenance** and future enhancements

## Contributing
Please open issues or pull requests for any bugs or feature requests. Contributions are welcome!

### Development Setup
```bash
# Install development dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run linting
uv run ruff check
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
