# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **2D to 3D CETSP Converter** (`src/cetsp/convert_to_3d.py`)
  - Convert standard 2D CETSP benchmark files to 3D format
  - Multiple z-coordinate generation strategies: `wave`, `random`, `dome`, `layers`, `distance`
  - Single file and batch conversion support
  - CLI interface for command-line usage
  - Comprehensive test suite (21 tests)

- **CETSP Benchmark Datasets** - Added 46 standard CETSP benchmark instances from literature
  - Located in `data/CETSP/Dataset/`
  - Includes 65 known solutions for comparison

### Changed
- Updated solver count to 16 variants across 7 algorithm families
- Added Simulated Annealing family: `SimulatedAnnealingCETSP`, `AdaptiveSimulatedAnnealingCETSP`, `ThresholdAcceptingCETSP`
- Added Grey Wolf family: `GreyWolfCETSP`, `EnhancedGreyWolfCETSP`

### Fixed
- Fixed all linting issues across codebase (ruff check)
- Fixed all formatting issues (ruff format)
- Updated type hints to use modern Python 3.10+ syntax (`list` instead of `List`, `X | None` instead of `Optional[X]`)

## [0.1.0] - 2026-02-05

### Added

#### Core Modules
- **3D Vehicle Routing Problem (VRP)** solver with support for:
  - Capacitated VRP (CVRP)
  - VRP with Time Windows (VRPTW)
  - Multi-Depot VRP (MDVRP)
  - Multiple solving strategies (nearest neighbor, savings algorithm)

- **Close-Enough TSP (CETSP)** module with 11 solver variants:
  - Baseline: `GreedyCETSP`, `InsertionCETSP`
  - Local Search: `AStarCETSP`, `TwoOptCETSP`
  - Genetic Algorithms: `GeneticCETSP`, `AdaptiveGeneticCETSP`
  - Ant Colony: `AntColonyCETSP`, `MaxMinAntSystem`, `AntColonySystemCETSP`
  - Particle Swarm: `ParticleSwarmCETSP`, `AdaptivePSOCETSP`, `DiscretePSOCETSP`
  - Hummingbird: `ArtificialHummingbirdCETSP`, `EnhancedAHACETSP`

- **BMSSP** (Single-Source Shortest Paths) implementation based on:
  - "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths" (arXiv:2504.17033)
  - Heap-based and block-based frontier implementations
  - Dijkstra reference implementation

- **A* VRP Solver** for state-space search approach to VRP

#### Infrastructure
- Comprehensive test suite with 127 tests
- Example scripts with visualizations (Matplotlib & Plotly)
- Benchmark comparison tools
- Professional project structure

### Documentation
- README with quick start guide
- API documentation in docstrings
- Example scripts demonstrating all features

[Unreleased]: https://github.com/yourusername/path-finding-3d/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/path-finding-3d/releases/tag/v0.1.0
