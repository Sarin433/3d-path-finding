# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

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
