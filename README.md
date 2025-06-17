# Coulomb Drag Analysis

This repository contains Python tools and Jupyter notebooks for analyzing Coulomb drag experiments.

## Features
- Load and pivot raw `.dat` files via `data.py`
- Comprehensive 1D/2D peak detection in consolidated `peaks.py` module
- **Modular peak analysis suite**:
  - `peak_detection.py` - Core peak detection and statistical analysis
  - `peak_visualization.py` - Plotting and visualization functions
  - `peak_tracking.py` - Advanced peak tracking and evolution analysis
- Advanced peak analysis utilities in `peak_utils.py` 
- Symmetric/anti-symmetric component analysis in `symmetries.py`
- Utilities for filtering, interpolation, gradient, and more in `data.py`
- Comprehensive test suite and performance benchmarks
- Example workflows in Jupyter notebooks under `*.ipynb`

## Getting Started

### Prerequisites
- Python 3.11+ installed
- [uv](https://docs.astral.sh/uv/#projects) (Astral’s uv CLI)

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

### Usage
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

### Quick Start Example
```python
# Example using the new modular peak analysis structure
from peak_detection import find_peaks_in_data, analyze_peak_statistics
from peak_visualization import plot_data_with_peaks, plot_peaks_vs_current
from peak_tracking import plot_peak_tracking_across_current

# Load your data
data = {...}  # Your measurement data

# Detect peaks
peaks = find_peaks_in_data(data, 'symmetric_smoothed')

# Analyze statistics
stats = analyze_peak_statistics(peaks)

# Visualize results
plot_data_with_peaks(data, peaks)
plot_peak_tracking_across_current(peaks, data)
```





## Contributing
Please open issues or pull requests for any bugs or feature requests. Contributions are welcome!
