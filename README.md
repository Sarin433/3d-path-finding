<p align="center">
  <h1 align="center">🎯 CETSP - Close-Enough Travelling Salesman Problem</h1>
  <p align="center">
    A comprehensive Python library for solving Close-Enough TSP in 3D space with multiple metaheuristic algorithms
  </p>
</p>

## ✨ What is CETSP?

The **Close-Enough Travelling Salesman Problem (CETSP)** is a variant of the classic TSP where the salesman doesn't need to visit the exact location of each customer—they just need to get "close enough" (within a specified radius). This models real-world scenarios like:

- 📡 **Wireless coverage** - A drone providing signal within range
- 🚁 **Drone delivery** - Dropping packages within acceptable zones
- 📷 **Surveillance** - Cameras covering areas from a distance
- 🔧 **Service routing** - Technicians serving areas, not exact points

---

## 🚀 Features

### 16 Solver Variants Across 7 Algorithm Families

| Family | Solvers | Description |
|--------|---------|-------------|
| **Baseline** | `GreedyCETSP`, `InsertionCETSP` | Fast heuristics for initial solutions |
| **Local Search** | `AStarCETSP`, `TwoOptCETSP` | Optimal/near-optimal refinement |
| **Genetic Algorithm** | `GeneticCETSP`, `AdaptiveGeneticCETSP` | Evolution-based optimization |
| **Ant Colony** | `AntColonyCETSP`, `MaxMinAntSystem`, `AntColonySystemCETSP` | Pheromone-based swarm intelligence |
| **Particle Swarm** | `ParticleSwarmCETSP`, `AdaptivePSOCETSP`, `DiscretePSOCETSP` | Velocity-based optimization |
| **Hummingbird** | `ArtificialHummingbirdCETSP`, `EnhancedAHACETSP` | Nature-inspired metaheuristic |
| **Grey Wolf** | `GreyWolfCETSP`, `EnhancedGreyWolfCETSP` | Pack hunting-inspired optimization |
| **Simulated Annealing** | `SimulatedAnnealingCETSP`, `AdaptiveSimulatedAnnealingCETSP`, `ThresholdAcceptingCETSP` | Temperature-based probabilistic search |

### Additional Features

- ✅ **3D Support** - Full 3D coordinate system
- ✅ **Visualization** - Matplotlib & Plotly support
- ✅ **Benchmarking** - Built-in comparison tools
- ✅ **Type Hints** - Fully typed for IDE support
- ✅ **Well Tested** - Comprehensive test suite

---

## 📦 Installation

### Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Using uv (recommended)

```bash
git clone https://github.com/yourusername/cetsp.git
cd cetsp
uv sync
```

### Using pip

```bash
git clone https://github.com/yourusername/cetsp.git
cd cetsp
pip install -e .
```

---

## 🎯 Quick Start

### Basic Usage

```python
from src.cetsp import CETSP, CETSPNode

# Create depot (radius=0 means exact visit required)
depot = CETSPNode(id=0, x=50, y=50, z=25, radius=0)

# Create customers with coverage radii
customers = [
    CETSPNode(id=1, x=20, y=30, z=10, radius=5),
    CETSPNode(id=2, x=80, y=70, z=40, radius=8),
    CETSPNode(id=3, x=40, y=90, z=15, radius=6),
    CETSPNode(id=4, x=70, y=20, z=30, radius=10),
]

# Create problem and solve
cetsp = CETSP(depot, customers)

# Try different solving methods
greedy = cetsp.solve(method='greedy')      # Fast baseline
astar = cetsp.solve(method='astar')        # Optimal for small instances
genetic = cetsp.solve(method='genetic')    # Good for medium instances
pso = cetsp.solve(method='pso')            # Particle swarm optimization

print(f"Greedy:  {greedy.total_distance:.2f}")
print(f"A*:      {astar.total_distance:.2f}")
print(f"Genetic: {genetic.total_distance:.2f}")
print(f"PSO:     {pso.total_distance:.2f}")
```

### Using Specific Solvers

```python
from src.cetsp import CETSP, CETSPNode
from src.cetsp.solvers import (
    GeneticCETSP,
    AntColonyCETSP,
    ParticleSwarmCETSP,
    TwoOptCETSP,
)

cetsp = CETSP(depot, customers)

# Genetic Algorithm with custom parameters
ga = GeneticCETSP(
    cetsp,
    population_size=100,
    generations=200,
    mutation_rate=0.1,
)
ga_solution = ga.solve()

# Ant Colony Optimization
aco = AntColonyCETSP(
    cetsp,
    n_ants=30,
    n_iterations=100,
    alpha=1.0,  # Pheromone importance
    beta=2.0,   # Heuristic importance
)
aco_solution = aco.solve()

# Improve any solution with 2-opt
two_opt = TwoOptCETSP(cetsp)
improved = two_opt.improve(ga_solution)
```

### Visualization

```python
# Built-in visualization
cetsp.visualize(solution, method='plotly')  # Interactive HTML
cetsp.visualize(solution, method='matplotlib')  # Static PNG
```

---

## 📚 All Available Solvers

```python
from src.cetsp.solvers import (
    # Baseline Heuristics
    GreedyCETSP,           # Nearest neighbor heuristic
    InsertionCETSP,        # Cheapest insertion heuristic

    # Local Search
    AStarCETSP,            # A* optimal search (small instances)
    TwoOptCETSP,           # 2-opt local improvement

    # Genetic Algorithms
    GeneticCETSP,          # Standard genetic algorithm
    AdaptiveGeneticCETSP,  # Self-adapting parameters

    # Ant Colony Optimization
    AntColonyCETSP,        # Basic ACO
    MaxMinAntSystem,       # MAX-MIN Ant System
    AntColonySystemCETSP,  # Ant Colony System

    # Particle Swarm Optimization
    ParticleSwarmCETSP,    # Standard PSO
    AdaptivePSOCETSP,      # Adaptive inertia weight
    DiscretePSOCETSP,      # Discrete/combinatorial PSO

    # Artificial Hummingbird Algorithm
    ArtificialHummingbirdCETSP,  # Standard AHA
    EnhancedAHACETSP,            # Enhanced with local search

    # Grey Wolf Optimization
    GreyWolfCETSP,               # Standard GWO
    EnhancedGreyWolfCETSP,       # Enhanced with adaptive strategies

    # Simulated Annealing
    SimulatedAnnealingCETSP,     # Standard SA
    AdaptiveSimulatedAnnealingCETSP,  # Adaptive with multiple operators
    ThresholdAcceptingCETSP,     # Deterministic threshold variant
)
```

---

## 📁 Project Structure

```
cetsp/
├── src/
│   └── cetsp/
│       ├── __init__.py      # Package exports
│       ├── node.py          # CETSPNode class
│       ├── problem.py       # CETSP, CETSPPath, CETSPSolution
│       └── solvers/
│           ├── __init__.py
│           ├── base.py      # Base solver class
│           ├── greedy.py    # Greedy heuristic
│           ├── insertion.py # Insertion heuristic
│           ├── astar.py     # A* search
│           ├── two_opt.py   # 2-opt improvement
│           ├── genetic.py   # Genetic algorithms
│           ├── aco.py       # Ant colony optimization
│           ├── pso.py       # Particle swarm optimization
│           ├── aha.py       # Artificial hummingbird
│           ├── gwo.py       # Grey wolf optimization
│           └── sa.py        # Simulated annealing
├── tests/
│   └── test_cetsp.py        # Comprehensive tests
├── examples/
│   ├── cetsp_example.py
│   ├── cetsp_benchmark_table.py
│   └── cetsp_variant_comparison.py
├── pyproject.toml
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
└── README.md
```

---

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=src --cov-report=html
```

---

## 📊 Algorithm Comparison

| Algorithm | Quality | Speed | Best Use Case |
|-----------|:-------:|:-----:|---------------|
| Greedy | ⭐⭐ | ⭐⭐⭐⭐⭐ | Quick baseline, very large instances |
| A* | ⭐⭐⭐⭐⭐ | ⭐⭐ | Small instances (<15 nodes), optimal needed |
| 2-opt | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Improving existing solutions |
| Genetic | ⭐⭐⭐⭐ | ⭐⭐⭐ | Medium instances, good balance |
| ACO | ⭐⭐⭐⭐ | ⭐⭐⭐ | Complex problem landscapes |
| PSO | ⭐⭐⭐⭐ | ⭐⭐⭐ | Continuous search spaces |
| AHA | ⭐⭐⭐⭐ | ⭐⭐⭐ | Balanced exploration/exploitation |
| Grey Wolf | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Fast convergence, robust |
| Sim. Annealing | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Escaping local optima, high quality |

---

## 📈 Examples

Run the benchmark to compare all algorithms:

```bash
uv run python examples/cetsp_benchmark_table.py
```

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📚 References

- Gulczynski, Heath, and Price. *"The Close Enough Traveling Salesman Problem: A discussion of several heuristics"*
- Dorigo and Gambardella. *"Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem"*
- Kennedy and Eberhart. *"Particle Swarm Optimization"*
- Zhao, Zhang, and Wang. *"Artificial Hummingbird Algorithm: A new bio-inspired optimizer"*
- Mirjalili, Mirjalili, and Lewis. *"Grey Wolf Optimizer"* - Advances in Engineering Software, 2014
- Kirkpatrick, Gelatt, and Vecchi. *"Optimization by Simulated Annealing"* - Science, 1983

---

<p align="center">
  Made with ❤️ for the optimization community
</p>
