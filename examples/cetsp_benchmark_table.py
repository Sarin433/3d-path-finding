"""
CETSP Benchmark Table
=====================
Execute all CETSP algorithms and display results in a formatted table.

This script provides a quick overview of all solver performance on a test problem.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cetsp import CETSP, CETSPNode

# Import all solvers
from src.cetsp.solvers import (
    AdaptiveGeneticCETSP,
    AdaptivePSOCETSP,
    AdaptiveSimulatedAnnealingCETSP,
    AntColonyCETSP,
    AntColonySystemCETSP,
    ArtificialHummingbirdCETSP,
    AStarCETSP,
    DiscretePSOCETSP,
    EnhancedAHACETSP,
    EnhancedGreyWolfCETSP,
    GeneticCETSP,
    GreedyCETSP,
    GreyWolfCETSP,
    MaxMinAntSystem,
    ParticleSwarmCETSP,
    SimulatedAnnealingCETSP,
    ThresholdAcceptingCETSP,
    TwoOptCETSP,
)


@dataclass
class BenchmarkResult:
    """Stores benchmark result for a single algorithm."""

    algorithm: str
    category: str
    distance: float
    time_ms: float
    improvement: float  # % improvement over baseline
    rank: int = 0


def create_test_problem(n_customers: int = 25, seed: int = 42) -> CETSP:
    """Create a random CETSP problem instance."""
    np.random.seed(seed)

    depot = CETSPNode(id=0, x=50, y=50, z=25, radius=0)

    customers = []
    for i in range(1, n_customers + 1):
        customers.append(
            CETSPNode(
                id=i,
                x=np.random.uniform(5, 95),
                y=np.random.uniform(5, 95),
                z=np.random.uniform(5, 45),
                radius=np.random.uniform(3, 12),
            )
        )

    return CETSP(depot, customers)


def run_solver(solver_class, cetsp: CETSP, name: str, category: str, **kwargs) -> BenchmarkResult:
    """Run a solver and return benchmark result."""
    start = time.perf_counter()

    try:
        solver = solver_class(cetsp, **kwargs)
        solution = solver.solve()
        elapsed = (time.perf_counter() - start) * 1000

        return BenchmarkResult(
            algorithm=name,
            category=category,
            distance=solution.total_distance,
            time_ms=elapsed,
            improvement=0.0,
        )
    except Exception:
        return BenchmarkResult(
            algorithm=name, category=category, distance=float("inf"), time_ms=0, improvement=0.0
        )


def run_all_benchmarks(cetsp: CETSP, n_customers: int) -> list[BenchmarkResult]:
    """Run all solvers and collect results."""
    results = []

    # Adaptive parameters based on problem size
    if n_customers <= 15:
        pop_size, iterations = 80, 150
    elif n_customers <= 25:
        pop_size, iterations = 60, 100
    else:
        pop_size, iterations = 40, 80

    # ═══════════════════════════════════════════════════════════════════════
    # BASELINE ALGORITHMS
    # ═══════════════════════════════════════════════════════════════════════
    print("  Running baseline algorithms...")

    # Greedy
    results.append(run_solver(GreedyCETSP, cetsp, "Greedy (NN)", "Baseline"))

    # A*
    results.append(run_solver(AStarCETSP, cetsp, "A*", "Baseline", max_iterations=50000))

    # A* + 2-Opt
    start = time.perf_counter()
    astar = AStarCETSP(cetsp, max_iterations=50000)
    astar_sol = astar.solve()
    two_opt = TwoOptCETSP(cetsp)
    improved = two_opt.improve(astar_sol)
    elapsed = (time.perf_counter() - start) * 1000
    results.append(
        BenchmarkResult(
            algorithm="A* + 2-Opt",
            category="Baseline",
            distance=improved.total_distance,
            time_ms=elapsed,
            improvement=0.0,
        )
    )

    # ═══════════════════════════════════════════════════════════════════════
    # GENETIC ALGORITHMS
    # ═══════════════════════════════════════════════════════════════════════
    print("  Running genetic algorithms...")

    results.append(
        run_solver(
            GeneticCETSP,
            cetsp,
            "Standard GA",
            "Genetic",
            population_size=pop_size,
            generations=iterations,
        )
    )

    results.append(
        run_solver(
            AdaptiveGeneticCETSP,
            cetsp,
            "Adaptive GA",
            "Genetic",
            population_size=pop_size,
            generations=iterations,
        )
    )

    # ═══════════════════════════════════════════════════════════════════════
    # ANT COLONY OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════════
    print("  Running ant colony algorithms...")

    results.append(
        run_solver(
            AntColonyCETSP,
            cetsp,
            "Standard ACO",
            "Ant Colony",
            n_ants=pop_size,
            n_iterations=iterations,
        )
    )

    results.append(
        run_solver(
            MaxMinAntSystem,
            cetsp,
            "MAX-MIN AS",
            "Ant Colony",
            n_ants=pop_size,
            n_iterations=iterations,
        )
    )

    results.append(
        run_solver(
            AntColonySystemCETSP,
            cetsp,
            "Ant Colony System",
            "Ant Colony",
            n_ants=pop_size,
            n_iterations=iterations,
        )
    )

    # ═══════════════════════════════════════════════════════════════════════
    # ARTIFICIAL HUMMINGBIRD ALGORITHM
    # ═══════════════════════════════════════════════════════════════════════
    print("  Running hummingbird algorithms...")

    results.append(
        run_solver(
            ArtificialHummingbirdCETSP,
            cetsp,
            "Standard AHA",
            "Hummingbird",
            n_hummingbirds=pop_size,
            n_iterations=iterations,
        )
    )

    results.append(
        run_solver(
            EnhancedAHACETSP,
            cetsp,
            "Enhanced AHA",
            "Hummingbird",
            n_hummingbirds=pop_size,
            n_iterations=iterations,
        )
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PARTICLE SWARM OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════════
    print("  Running particle swarm algorithms...")

    results.append(
        run_solver(
            ParticleSwarmCETSP,
            cetsp,
            "Standard PSO",
            "PSO",
            n_particles=pop_size,
            n_iterations=iterations,
        )
    )

    results.append(
        run_solver(
            AdaptivePSOCETSP,
            cetsp,
            "Adaptive PSO",
            "PSO",
            n_particles=pop_size,
            n_iterations=iterations,
        )
    )

    results.append(
        run_solver(
            DiscretePSOCETSP,
            cetsp,
            "Discrete PSO",
            "PSO",
            n_particles=pop_size,
            n_iterations=iterations,
        )
    )

    # ═══════════════════════════════════════════════════════════════════════
    # GREY WOLF OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════════
    print("  Running grey wolf algorithms...")

    results.append(
        run_solver(
            GreyWolfCETSP,
            cetsp,
            "Grey Wolf (GWO)",
            "Grey Wolf",
            n_wolves=pop_size,
            n_iterations=iterations,
        )
    )

    results.append(
        run_solver(
            EnhancedGreyWolfCETSP,
            cetsp,
            "Enhanced GWO",
            "Grey Wolf",
            n_wolves=pop_size,
            n_iterations=iterations,
        )
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SIMULATED ANNEALING
    # ═══════════════════════════════════════════════════════════════════════
    print("  Running simulated annealing algorithms...")

    results.append(
        run_solver(SimulatedAnnealingCETSP, cetsp, "Simulated Annealing", "SA", n_iterations=100)
    )

    results.append(
        run_solver(AdaptiveSimulatedAnnealingCETSP, cetsp, "Adaptive SA", "SA", n_iterations=100)
    )

    results.append(
        run_solver(ThresholdAcceptingCETSP, cetsp, "Threshold Accepting", "SA", n_iterations=100)
    )

    return results


def calculate_rankings(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    """Calculate rankings and improvement percentages."""
    # Find baseline (greedy) distance
    baseline_dist = next(r.distance for r in results if r.algorithm == "Greedy (NN)")

    # Sort by distance
    sorted_results = sorted(results, key=lambda r: r.distance)

    # Assign ranks and calculate improvement
    for rank, result in enumerate(sorted_results, 1):
        result.rank = rank
        result.improvement = (baseline_dist - result.distance) / baseline_dist * 100

    return sorted_results


def print_table(results: list[BenchmarkResult], title: str = "CETSP Benchmark Results"):
    """Print results as a formatted ASCII table."""

    # Column widths
    w_rank = 4
    w_algo = 20
    w_cat = 12
    w_dist = 12
    w_time = 12
    w_imp = 10

    total_width = w_rank + w_algo + w_cat + w_dist + w_time + w_imp + 7  # +7 for separators

    # Header
    print()
    print("=" * total_width)
    print(f" {title.center(total_width - 2)} ")
    print("=" * total_width)

    # Column headers
    header = (
        f"{'Rank':^{w_rank}}|"
        f"{'Algorithm':^{w_algo}}|"
        f"{'Category':^{w_cat}}|"
        f"{'Distance':^{w_dist}}|"
        f"{'Time (ms)':^{w_time}}|"
        f"{'Improv.':^{w_imp}}"
    )
    print(header)
    print("-" * total_width)

    # Data rows
    for r in results:
        imp_str = f"+{r.improvement:.1f}%" if r.improvement > 0 else f"{r.improvement:.1f}%"
        row = (
            f"{r.rank:^{w_rank}}|"
            f"{r.algorithm:^{w_algo}}|"
            f"{r.category:^{w_cat}}|"
            f"{r.distance:^{w_dist}.2f}|"
            f"{r.time_ms:^{w_time}.1f}|"
            f"{imp_str:^{w_imp}}"
        )
        print(row)

    print("=" * total_width)

    # Summary stats
    best = results[0]
    worst = results[-1]
    avg_dist = np.mean([r.distance for r in results])
    avg_time = np.mean([r.time_ms for r in results])

    print(f"\n  Best:    {best.algorithm} ({best.category}) - Distance: {best.distance:.2f}")
    print(f"  Worst:   {worst.algorithm} ({worst.category}) - Distance: {worst.distance:.2f}")
    print(f"  Average: Distance = {avg_dist:.2f}, Time = {avg_time:.1f} ms")
    print(
        f"  Gap:     {(worst.distance - best.distance) / best.distance * 100:.1f}% between best and worst"
    )


def print_category_summary(results: list[BenchmarkResult]):
    """Print summary by category."""
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)

    print("\n" + "=" * 70)
    print(" CATEGORY SUMMARY ")
    print("=" * 70)
    print(f"{'Category':<15}{'Best Algorithm':<20}{'Distance':>12}{'Avg Time':>12}")
    print("-" * 70)

    for cat, cat_results in categories.items():
        best = min(cat_results, key=lambda r: r.distance)
        avg_time = np.mean([r.time_ms for r in cat_results])
        print(f"{cat:<15}{best.algorithm:<20}{best.distance:>12.2f}{avg_time:>12.1f} ms")

    print("=" * 70)


def save_html_table(results: list[BenchmarkResult], filepath: str, n_customers: int, seed: int):
    """Save results as an HTML table."""

    # Category colors
    cat_colors = {
        "Baseline": "#95a5a6",
        "Genetic": "#3498db",
        "Ant Colony": "#e74c3c",
        "Hummingbird": "#2ecc71",
        "PSO": "#9b59b6",
    }

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>CETSP Benchmark Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background: #2c3e50;
            color: white;
            padding: 15px;
            text-align: center;
            font-weight: 600;
        }}
        td {{
            padding: 12px 15px;
            text-align: center;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        tr:first-child td {{
            background: #d4edda;
            font-weight: bold;
        }}
        .rank-1 {{ color: #27ae60; font-weight: bold; }}
        .rank-2 {{ color: #2980b9; }}
        .rank-3 {{ color: #8e44ad; }}
        .category {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 15px;
            color: white;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .summary {{
            display: flex;
            justify-content: space-around;
            margin-top: 30px;
            flex-wrap: wrap;
        }}
        .stat-card {{
            background: white;
            padding: 20px 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            min-width: 180px;
            margin: 10px;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <h1>3D-CETSP Benchmark Results</h1>
    <p class="subtitle">{n_customers} Customers | Seed: {seed} | {len(results)} Algorithms</p>

    <table>
        <tr>
            <th>Rank</th>
            <th>Algorithm</th>
            <th>Category</th>
            <th>Distance</th>
            <th>Time (ms)</th>
            <th>vs Baseline</th>
        </tr>
"""

    for r in results:
        rank_class = f"rank-{r.rank}" if r.rank <= 3 else ""
        cat_color = cat_colors.get(r.category, "#95a5a6")
        imp_class = "positive" if r.improvement > 0 else "negative"
        imp_str = f"+{r.improvement:.1f}%" if r.improvement > 0 else f"{r.improvement:.1f}%"

        html += f"""        <tr>
            <td class="{rank_class}">{r.rank}</td>
            <td><strong>{r.algorithm}</strong></td>
            <td><span class="category" style="background:{cat_color}">{r.category}</span></td>
            <td>{r.distance:.2f}</td>
            <td>{r.time_ms:.1f}</td>
            <td class="{imp_class}">{imp_str}</td>
        </tr>
"""

    # Stats
    best = results[0]
    worst = results[-1]
    avg_dist = np.mean([r.distance for r in results])
    total_time = sum(r.time_ms for r in results)

    html += f"""    </table>

    <div class="summary">
        <div class="stat-card">
            <div class="stat-value" style="color:#27ae60">{best.distance:.1f}</div>
            <div class="stat-label">Best Distance<br><small>{best.algorithm}</small></div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_dist:.1f}</div>
            <div class="stat-label">Average Distance</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#e74c3c">{worst.distance:.1f}</div>
            <div class="stat-label">Worst Distance<br><small>{worst.algorithm}</small></div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_time / 1000:.2f}s</div>
            <div class="stat-label">Total Runtime</div>
        </div>
    </div>
</body>
</html>
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  Saved HTML table: {filepath}")


def main():
    """Main benchmark execution."""
    print("=" * 70)
    print(" CETSP ALGORITHM BENCHMARK ".center(70))
    print("=" * 70)

    # Configuration
    N_CUSTOMERS = 50
    SEED = 42

    # Create problem
    print("\n[1] Creating test problem...")
    print(f"    Customers: {N_CUSTOMERS}")
    print(f"    Seed: {SEED}")

    cetsp = create_test_problem(n_customers=N_CUSTOMERS, seed=SEED)
    print(f"    Depot: ({cetsp.depot.x:.0f}, {cetsp.depot.y:.0f}, {cetsp.depot.z:.0f})")

    # Run benchmarks
    print(f"\n[2] Running {14} algorithms...")
    results = run_all_benchmarks(cetsp, N_CUSTOMERS)

    # Calculate rankings
    print("\n[3] Computing rankings...")
    ranked_results = calculate_rankings(results)

    # Print results
    print_table(ranked_results, f"CETSP Benchmark - {N_CUSTOMERS} Customers")
    print_category_summary(ranked_results)

    # Save HTML
    output_path = Path(__file__).parent / "benchmark_table.html"
    save_html_table(ranked_results, str(output_path), N_CUSTOMERS, SEED)

    print("\n" + "=" * 70)
    print(" BENCHMARK COMPLETE ".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
