# Coulomb Drag Analysis

This repository contains Python tools and Jupyter notebooks for analyzing Coulomb drag experiments.

## Features
- Load and pivot raw `.dat` files via `data.py`
- 1D/2D peak detection and visualization modules
- Utilities for filtering, interpolation, gradient, and more in `data.py`
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
```

## Project Structure
```
analysis/
├── data.py            # core data loader and transformations
├── peaks.py           # peak detection utilities
├── main.py            # example CLI entrypoint
├── *.ipynb            # example notebooks
├── pyproject.toml     # project metadata & dependencies
├── README.md          # project documentation
├── .gitignore         # ignored files
├── .python-version    # pinned Python version
└── uv.lock            # locked dependencies
```

## Contributing
Please open issues or pull requests for any bugs or feature requests. Contributions are welcome!
