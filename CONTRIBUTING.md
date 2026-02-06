# Contributing to Path Finding 3D

Thank you for your interest in contributing to Path Finding 3D! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/path-finding-3d/issues)
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce the bug
   - Expected vs actual behavior
   - Python version and OS
   - Relevant code snippets or error messages

### Suggesting Features

1. Check existing issues for similar suggestions
2. Create a new issue with the `enhancement` label
3. Describe the feature and its use case
4. Explain why it would benefit the project

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes following the code style guidelines
4. Add tests for new functionality
5. Run the test suite:
   ```bash
   pytest tests/ -v
   ```
6. Commit your changes with clear messages:
   ```bash
   git commit -m "feat: add new solver for X problem"
   ```
7. Push to your fork and create a Pull Request

## Development Setup

### Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) (recommended)

### Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/path-finding-3d.git
cd path-finding-3d

# Create virtual environment and install dependencies
uv sync --all-extras

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_cetsp.py -v
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/
```

## Code Style Guidelines

### General

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use descriptive variable and function names

### Documentation

- Add docstrings to all public functions, classes, and modules
- Use Google-style docstrings:

```python
def solve(self, method: str = "greedy") -> Solution:
    """Solve the optimization problem.

    Args:
        method: The solving method to use. Options: 'greedy', 'astar', 'genetic'.

    Returns:
        A Solution object containing the optimized path.

    Raises:
        ValueError: If method is not supported.

    Example:
        >>> problem = CETSP(depot, customers)
        >>> solution = problem.solve(method='genetic')
    """
```

### Commits

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

## Project Structure

```
path-finding-3d/
├── src/modules/          # Main source code
│   ├── vrp_solvers/      # VRP algorithms
│   ├── BMSSP/            # Shortest path algorithms
│   └── CETSP/            # Close-Enough TSP solvers
├── tests/                # Test files
├── examples/             # Example scripts
└── docs/                 # Documentation
```

## Adding New Algorithms

When adding a new solver:

1. Create a new file in the appropriate `solvers/` directory
2. Inherit from the base solver class
3. Implement required methods (`solve()`, etc.)
4. Add comprehensive tests
5. Update `__init__.py` exports
6. Add documentation and examples

Example:

```python
from .base import CETSPSolver

class MyNewSolver(CETSPSolver):
    """My new optimization solver.
    
    This solver uses [algorithm name] to find optimal solutions.
    """
    
    def __init__(self, problem: CETSP, **kwargs):
        super().__init__(problem)
        # Initialize parameters
    
    def solve(self) -> CETSPSolution:
        # Implement solving logic
        pass
```

## Questions?

Feel free to open an issue for any questions or reach out to the maintainers.

Thank you for contributing! 🎉
