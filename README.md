# Coulomb Drag Analysis

This repository contains Python tools and Jupyter notebooks for analyzing Coulomb drag experiments.

## Features
- Load and pivot raw `.dat` files via `data.py`
- Comprehensive 1D/2D peak detection in consolidated `peaks.py` module
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

## Module Consolidation

This codebase has undergone a comprehensive consolidation to improve organization and maintainability:

### Peak Detection Modules
- **Before**: Separate `peaks.py` and `peaks_2d.py` modules with overlapping functionality
- **After**: Unified `peaks.py` module containing all peak detection capabilities
- **Benefits**: 
  - Eliminated code duplication
  - Improved documentation and type hints
  - Maintained 100% backward compatibility
  - Added comprehensive test suite

### Key Improvements
- **Consolidated Interface**: All peak detection functions available from single import
- **Enhanced Documentation**: Comprehensive docstrings with usage examples
- **Type Safety**: Full type annotations throughout
- **Performance**: Benchmarked and optimized functions
- **Testing**: 5/5 test cases passing with comprehensive coverage
- **Utilities**: Additional analysis tools in `peak_utils.py`

For detailed information about the consolidation process, see `PEAK_CONSOLIDATION_SUMMARY.md`.

## Project Structure
```
analysis/
├── data.py                          # core data loader and transformations
├── data_load.py                     # 2D data loading utilities
├── peaks.py                         # consolidated peak detection module
├── peak_utils.py                    # additional peak analysis utilities
├── symmetries.py                    # symmetric/anti-symmetric analysis
├── main.py                          # example CLI entrypoint
├── test_peaks_consolidation.py      # comprehensive test suite
├── benchmark_peaks.py               # performance benchmarks
├── cleanup_workspace.py             # post-consolidation cleanup script
├── *.ipynb                          # example notebooks
├── PEAK_CONSOLIDATION_SUMMARY.md    # consolidation documentation
├── pyproject.toml                   # project metadata & dependencies
├── README.md                        # project documentation
├── .gitignore                       # ignored files
├── .python-version                  # pinned Python version
└── uv.lock                          # locked dependencies
```

## Contributing
Please open issues or pull requests for any bugs or feature requests. Contributions are welcome!
