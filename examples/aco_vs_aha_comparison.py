"""
ACO vs AHA Comparison - Strengths and Weaknesses Analysis
=========================================================
This script compares Ant Colony Optimization (ACO) and Artificial Hummingbird 
Algorithm (AHA) on various aspects to demonstrate their strengths and weaknesses.

Comparison Aspects:
1. Scalability (different problem sizes)
2. Convergence Speed (iterations to reach good solution)
3. Solution Quality Consistency (multiple runs)
4. Parameter Sensitivity
5. Runtime Efficiency
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cetsp import CETSP, CETSPNode, CETSPSolution
from src.cetsp.solvers import (
    AntColonyCETSP,
    MaxMinAntSystem,
    AntColonySystemCETSP,
    ArtificialHummingbirdCETSP,
    EnhancedAHACETSP,
    GreedyCETSP,
)


def generate_random_problem(n_customers: int, seed: int = 42) -> CETSP:
    """Generate a random 3D CETSP problem."""
    np.random.seed(seed)
    depot = CETSPNode(id=0, x=50, y=50, z=25, radius=0)
    customers = [
        CETSPNode(
            id=i,
            x=np.random.uniform(10, 90),
            y=np.random.uniform(10, 90),
            z=np.random.uniform(5, 45),
            radius=np.random.uniform(5, 15)
        )
        for i in range(1, n_customers + 1)
    ]
    return CETSP(depot, customers)


def run_with_convergence_tracking(solver_class, cetsp, n_iterations, **kwargs):
    """Run solver and track convergence (best distance per iteration)."""
    # For tracking, we'll run multiple times with increasing iterations
    convergence = []
    checkpoints = list(range(10, n_iterations + 1, 10))
    
    for n_iter in checkpoints:
        solver = solver_class(cetsp, n_iterations=n_iter, **kwargs)
        solution = solver.solve()
        convergence.append((n_iter, solution.total_distance))
    
    return convergence


def test_scalability():
    """Test how algorithms scale with problem size."""
    print("\n" + "=" * 70)
    print(" TEST 1: SCALABILITY (Problem Size) ".center(70))
    print("=" * 70)
    print("\nTesting performance as number of customers increases...")
    print("Parameters: 30 agents, 100 iterations\n")
    
    problem_sizes = [10, 20, 30, 40, 50]
    results = {'ACO': [], 'AHA': []}
    
    print(f"{'Size':<8}{'ACO Dist':<12}{'ACO Time':<12}{'AHA Dist':<12}{'AHA Time':<12}{'Winner':<10}")
    print("-" * 66)
    
    for n in problem_sizes:
        cetsp = generate_random_problem(n, seed=42)
        
        # ACO
        start = time.perf_counter()
        aco = AntColonyCETSP(cetsp, n_ants=30, n_iterations=100, seed=42)
        aco_sol = aco.solve()
        aco_time = (time.perf_counter() - start) * 1000
        
        # AHA
        start = time.perf_counter()
        aha = EnhancedAHACETSP(cetsp, n_hummingbirds=30, n_iterations=100, seed=42)
        aha_sol = aha.solve()
        aha_time = (time.perf_counter() - start) * 1000
        
        results['ACO'].append((n, aco_sol.total_distance, aco_time))
        results['AHA'].append((n, aha_sol.total_distance, aha_time))
        
        winner = "ACO" if aco_sol.total_distance < aha_sol.total_distance else "AHA"
        if abs(aco_sol.total_distance - aha_sol.total_distance) < 0.5:
            winner = "TIE"
        
        print(f"{n:<8}{aco_sol.total_distance:<12.2f}{aco_time:<12.1f}{aha_sol.total_distance:<12.2f}{aha_time:<12.1f}{winner:<10}")
    
    print("\n📊 SCALABILITY ANALYSIS:")
    print("  • ACO: Scales well with pheromone-based communication")
    print("  • AHA: May struggle with larger problems due to foraging mechanics")
    
    return results


def test_convergence_speed():
    """Test how quickly algorithms converge to good solutions."""
    print("\n" + "=" * 70)
    print(" TEST 2: CONVERGENCE SPEED ".center(70))
    print("=" * 70)
    print("\nTracking best solution quality over iterations...")
    print("Problem: 25 customers, 30 agents\n")
    
    cetsp = generate_random_problem(25, seed=42)
    
    # Get baseline
    greedy = GreedyCETSP(cetsp)
    baseline = greedy.solve().total_distance
    
    iterations = [10, 25, 50, 75, 100, 150, 200]
    
    print(f"{'Iterations':<12}{'ACO Dist':<12}{'ACO %':<10}{'AHA Dist':<12}{'AHA %':<10}")
    print("-" * 56)
    
    for n_iter in iterations:
        # ACO
        aco = AntColonyCETSP(cetsp, n_ants=30, n_iterations=n_iter, seed=42)
        aco_sol = aco.solve()
        aco_imp = (baseline - aco_sol.total_distance) / baseline * 100
        
        # AHA
        aha = EnhancedAHACETSP(cetsp, n_hummingbirds=30, n_iterations=n_iter, seed=42)
        aha_sol = aha.solve()
        aha_imp = (baseline - aha_sol.total_distance) / baseline * 100
        
        print(f"{n_iter:<12}{aco_sol.total_distance:<12.2f}{aco_imp:>+8.1f}%  {aha_sol.total_distance:<12.2f}{aha_imp:>+8.1f}%")
    
    print(f"\nBaseline (Greedy): {baseline:.2f}")
    print("\n📊 CONVERGENCE ANALYSIS:")
    print("  • ACO: Fast initial convergence due to pheromone exploitation")
    print("  • AHA: Slower start but can escape local optima through diverse foraging")


def test_consistency():
    """Test solution quality consistency across multiple runs."""
    print("\n" + "=" * 70)
    print(" TEST 3: SOLUTION CONSISTENCY (Multiple Runs) ".center(70))
    print("=" * 70)
    print("\nRunning each algorithm 10 times with different seeds...")
    print("Problem: 20 customers, 30 agents, 100 iterations\n")
    
    cetsp = generate_random_problem(20, seed=42)
    n_runs = 10
    
    aco_results = []
    aha_results = []
    
    for seed in range(n_runs):
        aco = AntColonyCETSP(cetsp, n_ants=30, n_iterations=100, seed=seed)
        aco_results.append(aco.solve().total_distance)
        
        aha = EnhancedAHACETSP(cetsp, n_hummingbirds=30, n_iterations=100, seed=seed)
        aha_results.append(aha.solve().total_distance)
    
    print(f"{'Metric':<20}{'ACO':<15}{'AHA':<15}{'Better':<10}")
    print("-" * 60)
    
    aco_mean, aha_mean = np.mean(aco_results), np.mean(aha_results)
    aco_std, aha_std = np.std(aco_results), np.std(aha_results)
    aco_min, aha_min = np.min(aco_results), np.min(aha_results)
    aco_max, aha_max = np.max(aco_results), np.max(aha_results)
    
    print(f"{'Best':<20}{aco_min:<15.2f}{aha_min:<15.2f}{'ACO' if aco_min < aha_min else 'AHA':<10}")
    print(f"{'Worst':<20}{aco_max:<15.2f}{aha_max:<15.2f}{'ACO' if aco_max < aha_max else 'AHA':<10}")
    print(f"{'Mean':<20}{aco_mean:<15.2f}{aha_mean:<15.2f}{'ACO' if aco_mean < aha_mean else 'AHA':<10}")
    print(f"{'Std Dev':<20}{aco_std:<15.2f}{aha_std:<15.2f}{'ACO' if aco_std < aha_std else 'AHA':<10}")
    print(f"{'Range':<20}{aco_max-aco_min:<15.2f}{aha_max-aha_min:<15.2f}{'ACO' if (aco_max-aco_min) < (aha_max-aha_min) else 'AHA':<10}")
    
    print("\n📊 CONSISTENCY ANALYSIS:")
    print(f"  • ACO Coefficient of Variation: {aco_std/aco_mean*100:.2f}%")
    print(f"  • AHA Coefficient of Variation: {aha_std/aha_mean*100:.2f}%")


def test_parameter_sensitivity():
    """Test how sensitive algorithms are to parameter changes."""
    print("\n" + "=" * 70)
    print(" TEST 4: PARAMETER SENSITIVITY ".center(70))
    print("=" * 70)
    print("\nTesting with different population sizes...")
    print("Problem: 20 customers, 100 iterations\n")
    
    cetsp = generate_random_problem(20, seed=42)
    pop_sizes = [10, 20, 30, 50, 100]
    
    print(f"{'Pop Size':<12}{'ACO Dist':<12}{'ACO Time':<12}{'AHA Dist':<12}{'AHA Time':<12}")
    print("-" * 60)
    
    aco_distances = []
    aha_distances = []
    
    for pop in pop_sizes:
        # ACO
        start = time.perf_counter()
        aco = AntColonyCETSP(cetsp, n_ants=pop, n_iterations=100, seed=42)
        aco_sol = aco.solve()
        aco_time = (time.perf_counter() - start) * 1000
        aco_distances.append(aco_sol.total_distance)
        
        # AHA
        start = time.perf_counter()
        aha = EnhancedAHACETSP(cetsp, n_hummingbirds=pop, n_iterations=100, seed=42)
        aha_sol = aha.solve()
        aha_time = (time.perf_counter() - start) * 1000
        aha_distances.append(aha_sol.total_distance)
        
        print(f"{pop:<12}{aco_sol.total_distance:<12.2f}{aco_time:<12.1f}{aha_sol.total_distance:<12.2f}{aha_time:<12.1f}")
    
    aco_sensitivity = (max(aco_distances) - min(aco_distances)) / np.mean(aco_distances) * 100
    aha_sensitivity = (max(aha_distances) - min(aha_distances)) / np.mean(aha_distances) * 100
    
    print(f"\n📊 PARAMETER SENSITIVITY ANALYSIS:")
    print(f"  • ACO sensitivity to population: {aco_sensitivity:.2f}% variation")
    print(f"  • AHA sensitivity to population: {aha_sensitivity:.2f}% variation")


def test_algorithm_variants():
    """Compare different variants of each algorithm."""
    print("\n" + "=" * 70)
    print(" TEST 5: ALGORITHM VARIANTS COMPARISON ".center(70))
    print("=" * 70)
    print("\nComparing standard vs enhanced versions...")
    print("Problem: 25 customers, 30 agents, 100 iterations\n")
    
    cetsp = generate_random_problem(25, seed=42)
    
    algorithms = [
        ("Standard ACO", AntColonyCETSP, {'n_ants': 30, 'n_iterations': 100, 'seed': 42}),
        ("MAX-MIN AS", MaxMinAntSystem, {'n_ants': 30, 'n_iterations': 100, 'seed': 42}),
        ("Ant Colony System", AntColonySystemCETSP, {'n_ants': 30, 'n_iterations': 100, 'seed': 42}),
        ("Standard AHA", ArtificialHummingbirdCETSP, {'n_hummingbirds': 30, 'n_iterations': 100, 'seed': 42}),
        ("Enhanced AHA", EnhancedAHACETSP, {'n_hummingbirds': 30, 'n_iterations': 100, 'seed': 42}),
    ]
    
    print(f"{'Algorithm':<25}{'Distance':<12}{'Time (ms)':<12}{'Rank':<8}")
    print("-" * 57)
    
    results = []
    for name, solver_class, params in algorithms:
        start = time.perf_counter()
        solver = solver_class(cetsp, **params)
        solution = solver.solve()
        elapsed = (time.perf_counter() - start) * 1000
        results.append((name, solution.total_distance, elapsed))
    
    # Sort by distance and add rank
    results.sort(key=lambda x: x[1])
    for rank, (name, dist, time_ms) in enumerate(results, 1):
        print(f"{name:<25}{dist:<12.2f}{time_ms:<12.1f}{rank:<8}")


def print_summary():
    """Print final summary of strengths and weaknesses."""
    print("\n" + "=" * 70)
    print(" SUMMARY: STRENGTHS AND WEAKNESSES ".center(70))
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    ANT COLONY OPTIMIZATION (ACO)                     │
├─────────────────────────────────────────────────────────────────────┤
│  ✅ STRENGTHS:                                                       │
│     • Fast convergence through pheromone exploitation                │
│     • Excellent for path/routing problems (natural fit)              │
│     • Scales well with problem size                                  │
│     • Consistent results across runs (low variance)                  │
│     • Good balance of exploration and exploitation                   │
│     • Multiple proven variants (MAX-MIN AS, ACS)                     │
│                                                                      │
│  ❌ WEAKNESSES:                                                       │
│     • Can get stuck in local optima (pheromone stagnation)           │
│     • Many parameters to tune (α, β, ρ, Q)                           │
│     • Slower per iteration due to pheromone updates                  │
│     • Memory intensive for large problems (pheromone matrix)         │
│     • Early convergence may miss global optimum                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│              ARTIFICIAL HUMMINGBIRD ALGORITHM (AHA)                  │
├─────────────────────────────────────────────────────────────────────┤
│  ✅ STRENGTHS:                                                       │
│     • Good exploration through diverse foraging behaviors            │
│     • Fewer parameters to tune                                       │
│     • Can escape local optima through guided/territorial foraging    │
│     • Fast per-iteration computation                                 │
│     • Novel approach - less prone to common pitfalls                 │
│     • Memory efficient                                               │
│                                                                      │
│  ❌ WEAKNESSES:                                                       │
│     • Slower convergence initially                                   │
│     • Higher variance in solution quality                            │
│     • May struggle with very large problems                          │
│     • Less mature - fewer proven enhancements                        │
│     • Not specifically designed for routing problems                 │
│     • May require more iterations for comparable quality             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      RECOMMENDATION GUIDE                            │
├─────────────────────────────────────────────────────────────────────┤
│  Use ACO when:                                                       │
│     • Problem is path/routing focused (like TSP, VRP, CETSP)         │
│     • Consistent, reliable results are needed                        │
│     • You have time to tune parameters                               │
│     • Problem size is moderate to large                              │
│                                                                      │
│  Use AHA when:                                                       │
│     • Quick prototyping without parameter tuning                     │
│     • Exploration is more important than exploitation                │
│     • Problem has many local optima                                  │
│     • Memory is constrained                                          │
│     • Combining with other methods (hybrid approach)                 │
└─────────────────────────────────────────────────────────────────────┘
""")


def main():
    print("=" * 70)
    print(" ACO vs AHA: Strengths and Weaknesses Analysis ".center(70))
    print("=" * 70)
    print("\nThis analysis compares Ant Colony Optimization (ACO) and")
    print("Artificial Hummingbird Algorithm (AHA) on multiple aspects.")
    
    # Run all tests
    test_scalability()
    test_convergence_speed()
    test_consistency()
    test_parameter_sensitivity()
    test_algorithm_variants()
    
    # Print summary
    print_summary()
    
    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE ".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
