"""
CETSP 3D Benchmark - Bubbles1 Dataset
=====================================
Test all CETSP algorithms on the 3D modified bubbles1 dataset.

This example loads the bubbles1_3d.cetsp file and runs all available solvers,
comparing their performance on a real benchmark problem.
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cetsp import CETSP, CETSPNode, CETSPSolution

# Import all solvers
from src.cetsp.solvers import (
    GreedyCETSP,
    AStarCETSP,
    TwoOptCETSP,
    GeneticCETSP,
    AdaptiveGeneticCETSP,
    AntColonyCETSP,
    MaxMinAntSystem,
    AntColonySystemCETSP,
    ArtificialHummingbirdCETSP,
    EnhancedAHACETSP,
    ParticleSwarmCETSP,
    AdaptivePSOCETSP,
    DiscretePSOCETSP,
    GreyWolfCETSP,
    EnhancedGreyWolfCETSP,
    SimulatedAnnealingCETSP,
    AdaptiveSimulatedAnnealingCETSP,
    ThresholdAcceptingCETSP,
)


def load_cetsp_file(filepath: str, depot_index: int = 0) -> CETSP:
    """
    Load a .cetsp file and create a CETSP problem.
    
    File format: x y z radius (one node per line)
    Lines starting with // are comments.
    
    Args:
        filepath: Path to the .cetsp file
        depot_index: Index of node to use as depot (0 = first node)
        
    Returns:
        CETSP problem instance
    """
    nodes = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('//'):
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                x, y, z, radius = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                nodes.append((x, y, z, radius))
    
    # Use specified node as depot (with radius=0)
    depot_data = nodes[depot_index]
    depot = CETSPNode(id=0, x=depot_data[0], y=depot_data[1], z=depot_data[2], radius=0)
    
    # Create customers from remaining nodes
    customers = []
    node_id = 1
    for i, (x, y, z, radius) in enumerate(nodes):
        if i != depot_index:
            customers.append(CETSPNode(id=node_id, x=x, y=y, z=z, radius=radius))
            node_id += 1
    
    return CETSP(depot, customers)


@dataclass
class BenchmarkResult:
    """Stores benchmark result for a single algorithm."""
    algorithm: str
    category: str
    distance: float
    time_ms: float
    improvement: float = 0.0
    rank: int = 0


def run_solver(solver_class, cetsp, name, category, **kwargs) -> BenchmarkResult:
    """Run a solver and return benchmark result."""
    try:
        start = time.perf_counter()
        solver = solver_class(cetsp, **kwargs)
        solution = solver.solve()
        elapsed = (time.perf_counter() - start) * 1000
        
        return BenchmarkResult(
            algorithm=name,
            category=category,
            distance=solution.total_distance,
            time_ms=elapsed
        )
    except Exception as e:
        print(f"    Error in {name}: {e}")
        return BenchmarkResult(
            algorithm=name,
            category=category,
            distance=float('inf'),
            time_ms=0
        )


def run_all_solvers(cetsp: CETSP, seed: int = 42) -> List[BenchmarkResult]:
    """Run all solvers on the CETSP problem."""
    results = []
    
    # Common parameters
    pop_size = 30
    iterations = 100
    
    print("\n[2] Running all algorithms...")
    
    # ═══════════════════════════════════════════════════════════════════════
    # BASELINE ALGORITHMS
    # ═══════════════════════════════════════════════════════════════════════
    print("  Running baseline algorithms...")
    
    results.append(run_solver(GreedyCETSP, cetsp, "Greedy (NN)", "Baseline"))
    
    # A* only for small instances
    if len(cetsp.customers) <= 15:
        results.append(run_solver(AStarCETSP, cetsp, "A*", "Baseline"))
        
        # A* + 2-opt
        start = time.perf_counter()
        astar = AStarCETSP(cetsp)
        astar_sol = astar.solve()
        two_opt = TwoOptCETSP(cetsp)
        improved = two_opt.improve(astar_sol)
        elapsed = (time.perf_counter() - start) * 1000
        results.append(BenchmarkResult(
            algorithm="A* + 2-Opt",
            category="Baseline",
            distance=improved.total_distance,
            time_ms=elapsed
        ))
    else:
        print("    Skipping A* (too many customers)")
    
    # ═══════════════════════════════════════════════════════════════════════
    # GENETIC ALGORITHMS
    # ═══════════════════════════════════════════════════════════════════════
    print("  Running genetic algorithms...")
    
    results.append(run_solver(
        GeneticCETSP, cetsp, "Standard GA", "Genetic",
        population_size=pop_size, generations=iterations, seed=seed
    ))
    
    results.append(run_solver(
        AdaptiveGeneticCETSP, cetsp, "Adaptive GA", "Genetic",
        population_size=pop_size, generations=iterations, seed=seed
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # ANT COLONY OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════════
    print("  Running ant colony algorithms...")
    
    results.append(run_solver(
        AntColonyCETSP, cetsp, "Standard ACO", "Ant Colony",
        n_ants=pop_size, n_iterations=iterations, seed=seed
    ))
    
    results.append(run_solver(
        MaxMinAntSystem, cetsp, "MAX-MIN AS", "Ant Colony",
        n_ants=pop_size, n_iterations=iterations, seed=seed
    ))
    
    results.append(run_solver(
        AntColonySystemCETSP, cetsp, "Ant Colony System", "Ant Colony",
        n_ants=pop_size, n_iterations=iterations, seed=seed
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # ARTIFICIAL HUMMINGBIRD ALGORITHM
    # ═══════════════════════════════════════════════════════════════════════
    print("  Running hummingbird algorithms...")
    
    results.append(run_solver(
        ArtificialHummingbirdCETSP, cetsp, "Standard AHA", "Hummingbird",
        n_hummingbirds=pop_size, n_iterations=iterations, seed=seed
    ))
    
    results.append(run_solver(
        EnhancedAHACETSP, cetsp, "Enhanced AHA", "Hummingbird",
        n_hummingbirds=pop_size, n_iterations=iterations, seed=seed
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # PARTICLE SWARM OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════════
    print("  Running particle swarm algorithms...")
    
    results.append(run_solver(
        ParticleSwarmCETSP, cetsp, "Standard PSO", "PSO",
        n_particles=pop_size, n_iterations=iterations, seed=seed
    ))
    
    results.append(run_solver(
        AdaptivePSOCETSP, cetsp, "Adaptive PSO", "PSO",
        n_particles=pop_size, n_iterations=iterations, seed=seed
    ))
    
    results.append(run_solver(
        DiscretePSOCETSP, cetsp, "Discrete PSO", "PSO",
        n_particles=pop_size, n_iterations=iterations, seed=seed
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # GREY WOLF OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════════
    print("  Running grey wolf algorithms...")
    
    results.append(run_solver(
        GreyWolfCETSP, cetsp, "Grey Wolf (GWO)", "Grey Wolf",
        n_wolves=pop_size, n_iterations=iterations, seed=seed
    ))
    
    results.append(run_solver(
        EnhancedGreyWolfCETSP, cetsp, "Enhanced GWO", "Grey Wolf",
        n_wolves=pop_size, n_iterations=iterations, seed=seed
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # SIMULATED ANNEALING
    # ═══════════════════════════════════════════════════════════════════════
    print("  Running simulated annealing algorithms...")
    
    results.append(run_solver(
        SimulatedAnnealingCETSP, cetsp, "Simulated Annealing", "SA",
        n_iterations=100, seed=seed
    ))
    
    results.append(run_solver(
        AdaptiveSimulatedAnnealingCETSP, cetsp, "Adaptive SA", "SA",
        n_iterations=100, seed=seed
    ))
    
    results.append(run_solver(
        ThresholdAcceptingCETSP, cetsp, "Threshold Accepting", "SA",
        n_iterations=100, seed=seed
    ))
    
    return results


def print_results(results: List[BenchmarkResult], title: str, n_customers: int):
    """Print results as a formatted table."""
    # Calculate rankings and improvement
    baseline_dist = next((r.distance for r in results if r.algorithm == "Greedy (NN)"), results[0].distance)
    
    sorted_results = sorted(results, key=lambda r: r.distance)
    for rank, result in enumerate(sorted_results, 1):
        result.rank = rank
        result.improvement = (baseline_dist - result.distance) / baseline_dist * 100
    
    # Print table
    print("\n" + "=" * 85)
    print(f" {title} - {n_customers} Customers ".center(85))
    print("=" * 85)
    print(f"{'Rank':<5}|{'Algorithm':<25}|{'Category':<12}|{'Distance':<12}|{'Time (ms)':<12}|{'Improv.':<8}")
    print("-" * 85)
    
    for r in sorted_results:
        imp_str = f"+{r.improvement:.1f}%" if r.improvement > 0 else f"{r.improvement:.1f}%"
        print(f" {r.rank:<4}|{r.algorithm:<25}|{r.category:<12}|{r.distance:<12.2f}|{r.time_ms:<12.1f}|{imp_str:<8}")
    
    print("=" * 85)
    
    # Summary
    best = sorted_results[0]
    worst = sorted_results[-1]
    avg_dist = np.mean([r.distance for r in sorted_results if r.distance < float('inf')])
    
    print(f"\n  Best:    {best.algorithm} ({best.category}) - Distance: {best.distance:.2f}")
    print(f"  Worst:   {worst.algorithm} ({worst.category}) - Distance: {worst.distance:.2f}")
    print(f"  Average: Distance = {avg_dist:.2f}")
    print(f"  Gap:     {(worst.distance - best.distance) / best.distance * 100:.1f}% between best and worst")
    
    # Category Summary Table
    print("\n" + "=" * 70)
    print(" CATEGORY SUMMARY ".center(70))
    print("=" * 70)
    print(f"{'Category':<15}{'Best Algorithm':<25}{'Distance':<12}{'Avg Time':<12}")
    print("-" * 70)
    
    categories = {}
    for r in sorted_results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)
    
    for cat in sorted(categories.keys(), key=lambda c: min(r.distance for r in categories[c])):
        cat_results = categories[cat]
        best_in_cat = min(cat_results, key=lambda r: r.distance)
        avg_time = np.mean([r.time_ms for r in cat_results])
        print(f"{cat:<15}{best_in_cat.algorithm:<25}{best_in_cat.distance:<12.2f}{avg_time:<12.1f} ms")
    
    print("=" * 70)
    
    return sorted_results


def visualize_best_solution(cetsp: CETSP, solution: CETSPSolution, title: str, save_path: str, results: List[BenchmarkResult] = None, total_time_ms: float = 0):
    """Visualize the best solution using Plotly Express with results summary table."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        
        # Prepare data for plotly express
        data = []
        
        # Add depot
        depot = cetsp.depot
        data.append({
            'x': depot.x, 'y': depot.y, 'z': depot.z,
            'type': 'Depot', 'size': 12, 'info': f'Depot<br>({depot.x:.1f}, {depot.y:.1f}, {depot.z:.1f})'
        })
        
        # Add customers
        for c in cetsp.customers:
            data.append({
                'x': c.x, 'y': c.y, 'z': c.z,
                'type': 'Customer', 'size': 6, 'info': f'Customer {c.id}<br>Radius: {c.radius:.1f}'
            })
        
        # Add waypoints from route
        waypoints = solution.path.get_waypoint_positions()
        if len(waypoints) > 0:
            for i, wp in enumerate(waypoints):
                data.append({
                    'x': wp[0], 'y': wp[1], 'z': wp[2],
                    'type': 'Waypoint', 'size': 5, 'info': f'Waypoint {i}'
                })
        
        df = pd.DataFrame(data)
        
        # Create 3D scatter plot using plotly express
        color_map = {'Depot': 'red', 'Customer': 'blue', 'Waypoint': 'darkgreen'}
        symbol_map = {'Depot': 'square', 'Customer': 'circle', 'Waypoint': 'diamond'}
        
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='type',
            symbol='type',
            size='size',
            hover_data={'info': True, 'x': ':.2f', 'y': ':.2f', 'z': ':.2f', 'size': False, 'type': False},
            color_discrete_map=color_map,
            symbol_map=symbol_map,
            labels={'x': 'X', 'y': 'Y', 'z': 'Z', 'type': 'Type'}
        )
        
        # Add route path as a line (using graph_objects for line)
        if len(waypoints) > 0:
            fig.add_trace(go.Scatter3d(
                x=waypoints[:, 0], y=waypoints[:, 1], z=waypoints[:, 2],
                mode='lines',
                line=dict(color='limegreen', width=6),
                name='Route Path',
                showlegend=True
            ))
        
        # Add coverage spheres (semi-transparent)
        for customer in cetsp.customers:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 15)
            r = customer.radius
            x = customer.x + r * np.outer(np.cos(u), np.sin(v))
            y = customer.y + r * np.outer(np.sin(u), np.sin(v))
            z = customer.z + r * np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                opacity=0.2,
                colorscale=[[0, 'cyan'], [1, 'cyan']],
                showscale=False,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=1100,
            height=700,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        # Get plotly div with CDN script included
        plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Build the full HTML page with styled table like reference image
        if results:
            best = results[0]
            worst = results[-1]
            avg_dist = np.mean([r.distance for r in results])
            total_runtime = total_time_ms / 1000  # Convert to seconds
            
            # Category colors
            cat_colors = {
                'SA': '#e74c3c',
                'Ant Colony': '#e67e22',
                'Hummingbird': '#27ae60',
                'Genetic': '#3498db',
                'PSO': '#9b59b6',
                'Grey Wolf': '#1abc9c',
                'Baseline': '#95a5a6'
            }
            
            # Build table rows
            table_rows = ""
            for r in results:
                cat_color = cat_colors.get(r.category, '#666')
                imp_color = '#27ae60' if r.improvement > 0 else '#e74c3c'
                imp_str = f"+{r.improvement:.1f}%" if r.improvement > 0 else f"{r.improvement:.1f}%"
                rank_color = '#27ae60' if r.rank <= 3 else '#333'
                
                table_rows += f"""
                <tr>
                    <td style="color: {rank_color}; font-weight: {'bold' if r.rank <= 3 else 'normal'};">{r.rank}</td>
                    <td style="font-weight: 600;">{r.algorithm}</td>
                    <td><span style="background: {cat_color}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px;">{r.category}</span></td>
                    <td style="text-align: right;">{r.distance:.2f}</td>
                    <td style="text-align: right;">{r.time_ms:.1f}</td>
                    <td style="text-align: right; color: {imp_color}; font-weight: 500;">{imp_str}</td>
                </tr>
                """
            
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3D-CETSP Benchmark Results</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f6fa; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        
        .header {{ text-align: center; padding: 30px 0; }}
        .header h1 {{ font-size: 32px; color: #2c3e50; margin-bottom: 10px; }}
        .header .subtitle {{ color: #7f8c8d; font-size: 14px; }}
        
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 30px 0; }}
        .stat-card {{ background: white; border-radius: 10px; padding: 25px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .stat-value {{ font-size: 36px; font-weight: 700; margin-bottom: 5px; }}
        .stat-label {{ color: #7f8c8d; font-size: 13px; }}
        .stat-sublabel {{ color: #95a5a6; font-size: 11px; margin-top: 3px; }}
        .stat-best {{ color: #27ae60; }}
        .stat-avg {{ color: #2c3e50; }}
        .stat-worst {{ color: #e74c3c; }}
        .stat-time {{ color: #3498db; }}
        
        .table-container {{ background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        thead {{ background: #34495e; color: white; }}
        th {{ padding: 15px 20px; text-align: left; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }}
        td {{ padding: 12px 20px; border-bottom: 1px solid #ecf0f1; font-size: 14px; }}
        tbody tr:hover {{ background: #f8f9fa; }}
        
        .plot-container {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .plot-title {{ font-size: 18px; font-weight: 600; color: #2c3e50; margin-bottom: 15px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>3D-CETSP Benchmark Results</h1>
            <p class="subtitle">{len(cetsp.customers)} Customers | Seed: 42 | {len(results)} Algorithms</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value stat-best">{best.distance:.1f}</div>
                <div class="stat-label">Best Distance</div>
                <div class="stat-sublabel">{best.algorithm}</div>
            </div>
            <div class="stat-card">
                <div class="stat-value stat-avg">{avg_dist:.1f}</div>
                <div class="stat-label">Average Distance</div>
            </div>
            <div class="stat-card">
                <div class="stat-value stat-worst">{worst.distance:.1f}</div>
                <div class="stat-label">Worst Distance</div>
                <div class="stat-sublabel">{worst.algorithm}</div>
            </div>
            <div class="stat-card">
                <div class="stat-value stat-time">{total_runtime:.2f}s</div>
                <div class="stat-label">Total Runtime</div>
            </div>
        </div>
        
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th style="width: 80px;">Rank</th>
                        <th>Algorithm</th>
                        <th style="width: 130px;">Category</th>
                        <th style="width: 120px; text-align: right;">Distance</th>
                        <th style="width: 120px; text-align: right;">Time (ms)</th>
                        <th style="width: 100px; text-align: right;">vs Baseline</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        
        <div class="plot-container">
            <div class="plot-title">Best Route Visualization - {best.algorithm} (Distance: {best.distance:.2f})</div>
            {plot_div}
        </div>
    </div>
</body>
</html>"""
        else:
            html_content = fig.to_html(full_html=True, include_plotlyjs=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n  Saved visualization: {save_path}")
        
    except ImportError:
        print("\n  Plotly not available, skipping visualization")


def main():
    print("=" * 70)
    print(" CETSP 3D BENCHMARK - Bubbles1 Dataset ".center(70))
    print("=" * 70)
    
    # Load the 3D dataset
    data_file = Path(__file__).parent.parent / "data" / "bubbles1_3d.cetsp"
    
    if not data_file.exists():
        print(f"Error: Dataset not found at {data_file}")
        return
    
    print(f"\n[1] Loading dataset: {data_file.name}")
    cetsp = load_cetsp_file(str(data_file), depot_index=0)  # Use first node as depot
    print(f"    Customers: {len(cetsp.customers)}")
    print(f"    Depot: ({cetsp.depot.x}, {cetsp.depot.y}, {cetsp.depot.z}) - from first node")
    
    # Run all solvers
    results = run_all_solvers(cetsp, seed=42)
    
    # Print results
    print("\n[3] Computing rankings...")
    sorted_results = print_results(results, "Bubbles1 3D Benchmark", len(cetsp.customers))
    
    # Visualize best solution
    print("\n[4] Generating visualization...")
    best_result = sorted_results[0]
    
    # Re-run best solver to get solution for visualization
    if best_result.algorithm == "Simulated Annealing":
        solver = SimulatedAnnealingCETSP(cetsp, seed=42)
    elif best_result.algorithm == "Threshold Accepting":
        solver = ThresholdAcceptingCETSP(cetsp, seed=42)
    elif best_result.algorithm == "Standard ACO":
        solver = AntColonyCETSP(cetsp, n_ants=30, n_iterations=100, seed=42)
    elif best_result.algorithm == "Enhanced GWO":
        solver = EnhancedGreyWolfCETSP(cetsp, n_wolves=30, n_iterations=100, seed=42)
    else:
        solver = SimulatedAnnealingCETSP(cetsp, seed=42)
    
    best_solution = solver.solve()
    
    # Calculate total runtime
    total_time_ms = sum(r.time_ms for r in sorted_results)
    
    save_path = Path(__file__).parent / "bubbles1_3d_best_route.html"
    visualize_best_solution(
        cetsp, best_solution,
        f"Bubbles1 3D - Best: {best_result.algorithm}",
        str(save_path),
        results=sorted_results,  # Pass results for summary table
        total_time_ms=total_time_ms
    )
    
    print("\n" + "=" * 70)
    print(" BENCHMARK COMPLETE ".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
