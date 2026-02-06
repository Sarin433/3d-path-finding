"""
CETSP Algorithm Variant Comparison

This example evaluates all variants of each metaheuristic algorithm family:
- Genetic Algorithm: Standard GA, Adaptive GA
- Ant Colony Optimization: Standard ACO, MAX-MIN Ant System (MMAS), Ant Colony System (ACS)
- Artificial Hummingbird Algorithm: Standard AHA, Enhanced AHA
- Particle Swarm Optimization: Standard PSO, Adaptive PSO, Discrete PSO

All variants are tested on the same problem instance for fair comparison.
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CETSP imports
from src.cetsp import CETSP, CETSPNode, CETSPSolution


@dataclass
class VariantResult:
    """Result from a solver variant."""
    family: str
    variant: str
    distance: float
    time_ms: float
    solution: CETSPSolution


def create_problem(n_customers: int = 30, seed: int = 42) -> CETSP:
    """Create a random 3D CETSP problem."""
    np.random.seed(seed)
    
    # Depot at center
    depot = CETSPNode(id=0, x=50, y=50, z=25, radius=0)
    
    # Random customers with coverage radii
    customers = [
        CETSPNode(
            id=i,
            x=np.random.uniform(5, 95),
            y=np.random.uniform(5, 95),
            z=np.random.uniform(5, 45),
            radius=np.random.uniform(4, 10)
        )
        for i in range(1, n_customers + 1)
    ]
    
    return CETSP(depot, customers)


def run_variant(cetsp: CETSP, method: str, family: str, variant_name: str, 
                params: dict, seed: int) -> VariantResult:
    """Run a single variant and return result."""
    params_with_seed = {**params, 'seed': seed}
    
    start = time.perf_counter()
    solution = cetsp.solve(method=method, **params_with_seed)
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    return VariantResult(
        family=family,
        variant=variant_name,
        distance=solution.total_distance,
        time_ms=elapsed_ms,
        solution=solution
    )


def run_all_variants(cetsp: CETSP, n_customers: int, seed: int = 42) -> List[VariantResult]:
    """Run all algorithm variants on the problem."""
    results = []
    
    # Determine parameters based on problem size
    small = n_customers <= 20
    
    # ==========================================================================
    # Genetic Algorithm Family
    # ==========================================================================
    print("\n" + "=" * 60)
    print("GENETIC ALGORITHM FAMILY")
    print("=" * 60)
    
    ga_params = {
        'population_size': 50 if small else 80,
        'generations': 150 if small else 250,
    }
    
    # Standard GA
    print("\n[GA-1] Standard Genetic Algorithm...")
    result = run_variant(cetsp, 'genetic', 'Genetic Algorithm', 'Standard GA', ga_params, seed)
    print(f"       Distance: {result.distance:.2f}, Time: {result.time_ms:.2f} ms")
    results.append(result)
    
    # Adaptive GA
    print("[GA-2] Adaptive Genetic Algorithm...")
    result = run_variant(cetsp, 'adaptive_genetic', 'Genetic Algorithm', 'Adaptive GA', ga_params, seed)
    print(f"       Distance: {result.distance:.2f}, Time: {result.time_ms:.2f} ms")
    results.append(result)
    
    # ==========================================================================
    # Ant Colony Optimization Family
    # ==========================================================================
    print("\n" + "=" * 60)
    print("ANT COLONY OPTIMIZATION FAMILY")
    print("=" * 60)
    
    aco_params = {
        'n_ants': 20 if small else 30,
        'n_iterations': 80 if small else 120,
    }
    
    # Standard ACO
    print("\n[ACO-1] Standard ACO...")
    result = run_variant(cetsp, 'aco', 'Ant Colony', 'Standard ACO', aco_params, seed)
    print(f"        Distance: {result.distance:.2f}, Time: {result.time_ms:.2f} ms")
    results.append(result)
    
    # MAX-MIN Ant System
    print("[ACO-2] MAX-MIN Ant System (MMAS)...")
    result = run_variant(cetsp, 'mmas', 'Ant Colony', 'MMAS', aco_params, seed)
    print(f"        Distance: {result.distance:.2f}, Time: {result.time_ms:.2f} ms")
    results.append(result)
    
    # Ant Colony System
    print("[ACO-3] Ant Colony System (ACS)...")
    result = run_variant(cetsp, 'acs', 'Ant Colony', 'ACS', aco_params, seed)
    print(f"        Distance: {result.distance:.2f}, Time: {result.time_ms:.2f} ms")
    results.append(result)
    
    # ==========================================================================
    # Artificial Hummingbird Algorithm Family
    # ==========================================================================
    print("\n" + "=" * 60)
    print("ARTIFICIAL HUMMINGBIRD ALGORITHM FAMILY")
    print("=" * 60)
    
    aha_params = {
        'n_hummingbirds': 25 if small else 35,
        'n_iterations': 120 if small else 180,
    }
    
    # Standard AHA
    print("\n[AHA-1] Standard AHA...")
    result = run_variant(cetsp, 'aha', 'Hummingbird', 'Standard AHA', aha_params, seed)
    print(f"        Distance: {result.distance:.2f}, Time: {result.time_ms:.2f} ms")
    results.append(result)
    
    # Enhanced AHA
    print("[AHA-2] Enhanced AHA...")
    result = run_variant(cetsp, 'enhanced_aha', 'Hummingbird', 'Enhanced AHA', aha_params, seed)
    print(f"        Distance: {result.distance:.2f}, Time: {result.time_ms:.2f} ms")
    results.append(result)
    
    # ==========================================================================
    # Particle Swarm Optimization Family
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PARTICLE SWARM OPTIMIZATION FAMILY")
    print("=" * 60)
    
    pso_params = {
        'n_particles': 25 if small else 35,
        'n_iterations': 120 if small else 180,
    }
    
    # Standard PSO
    print("\n[PSO-1] Standard PSO...")
    result = run_variant(cetsp, 'pso', 'Particle Swarm', 'Standard PSO', pso_params, seed)
    print(f"        Distance: {result.distance:.2f}, Time: {result.time_ms:.2f} ms")
    results.append(result)
    
    # Adaptive PSO
    print("[PSO-2] Adaptive PSO...")
    result = run_variant(cetsp, 'adaptive_pso', 'Particle Swarm', 'Adaptive PSO', pso_params, seed)
    print(f"        Distance: {result.distance:.2f}, Time: {result.time_ms:.2f} ms")
    results.append(result)
    
    # Discrete PSO
    print("[PSO-3] Discrete PSO...")
    result = run_variant(cetsp, 'discrete_pso', 'Particle Swarm', 'Discrete PSO', pso_params, seed)
    print(f"        Distance: {result.distance:.2f}, Time: {result.time_ms:.2f} ms")
    results.append(result)
    
    # ==========================================================================
    # Baseline: Greedy
    # ==========================================================================
    print("\n" + "=" * 60)
    print("BASELINE")
    print("=" * 60)
    
    print("\n[BASE] Greedy (Nearest Neighbor)...")
    result = run_variant(cetsp, 'greedy', 'Baseline', 'Greedy', {}, seed)
    print(f"       Distance: {result.distance:.2f}, Time: {result.time_ms:.2f} ms")
    results.append(result)
    
    return results


def print_summary(results: List[VariantResult]) -> None:
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("SUMMARY - ALL VARIANTS")
    print("=" * 80)
    
    # Sort by distance
    sorted_results = sorted(results, key=lambda r: r.distance)
    best_distance = sorted_results[0].distance
    
    print(f"\n{'Rank':<5} {'Family':<18} {'Variant':<15} {'Distance':>10} {'Time (ms)':>12} {'Gap':>8}")
    print("-" * 70)
    
    for rank, r in enumerate(sorted_results, 1):
        gap = (r.distance - best_distance) / best_distance * 100
        print(f"{rank:<5} {r.family:<18} {r.variant:<15} {r.distance:>10.2f} {r.time_ms:>12.2f} {gap:>7.1f}%")
    
    # Per-family best
    print("\n" + "-" * 70)
    print("BEST PER FAMILY:")
    print("-" * 70)
    
    families = {}
    for r in results:
        if r.family not in families or r.distance < families[r.family].distance:
            families[r.family] = r
    
    for family, r in sorted(families.items(), key=lambda x: x[1].distance):
        gap = (r.distance - best_distance) / best_distance * 100
        print(f"  {family:<18}: {r.variant:<15} - Distance: {r.distance:.2f} (Gap: {gap:.1f}%)")
    
    print(f"\n🏆 OVERALL BEST: {sorted_results[0].variant} ({sorted_results[0].family})")
    print(f"   Distance: {sorted_results[0].distance:.2f}")


def plot_matplotlib(results: List[VariantResult], save_path: str = None) -> plt.Figure:
    """Create matplotlib visualization."""
    # Define colors for each family
    family_colors = {
        'Genetic Algorithm': '#3498db',
        'Ant Colony': '#e74c3c',
        'Hummingbird': '#2ecc71',
        'Particle Swarm': '#9b59b6',
        'Baseline': '#95a5a6'
    }
    
    # Sort by distance
    sorted_results = sorted(results, key=lambda r: r.distance)
    best_distance = sorted_results[0].distance
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Distance comparison (grouped by family)
    ax1 = axes[0]
    
    variants = [r.variant for r in sorted_results]
    distances = [r.distance for r in sorted_results]
    colors = [family_colors[r.family] for r in sorted_results]
    
    bars = ax1.barh(variants, distances, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Total Distance', fontsize=11)
    ax1.set_title('Tour Distance by Variant (sorted)', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # Add value labels
    for bar, dist in zip(bars, distances):
        gap = (dist - best_distance) / best_distance * 100
        label = f'{dist:.1f} ({gap:+.1f}%)' if gap > 0 else f'{dist:.1f} (best)'
        ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9)
    
    # Add legend
    legend_patches = [Patch(color=color, label=family) 
                     for family, color in family_colors.items() if family != 'Baseline']
    legend_patches.append(Patch(color=family_colors['Baseline'], label='Baseline'))
    ax1.legend(handles=legend_patches, loc='lower right', fontsize=9)
    
    ax1.set_xlim(0, max(distances) * 1.25)
    
    # Right: Time vs Quality scatter plot
    ax2 = axes[1]
    
    for family, color in family_colors.items():
        family_results = [r for r in results if r.family == family]
        if family_results:
            times = [r.time_ms for r in family_results]
            dists = [r.distance for r in family_results]
            labels = [r.variant for r in family_results]
            
            ax2.scatter(times, dists, c=color, s=150, label=family, 
                       edgecolors='black', linewidth=0.5, alpha=0.8)
            
            for t, d, lbl in zip(times, dists, labels):
                ax2.annotate(lbl.replace(' ', '\n'), (t, d), 
                           textcoords='offset points', xytext=(5, 5),
                           fontsize=8, alpha=0.9)
    
    ax2.set_xlabel('Computation Time (ms)', fontsize=11)
    ax2.set_ylabel('Tour Distance', fontsize=11)
    ax2.set_title('Time vs Quality Trade-off', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Add Pareto frontier approximation
    pareto_points = []
    sorted_by_time = sorted(results, key=lambda r: r.time_ms)
    min_dist = float('inf')
    for r in sorted_by_time:
        if r.distance < min_dist:
            pareto_points.append(r)
            min_dist = r.distance
    
    if len(pareto_points) > 1:
        pareto_times = [r.time_ms for r in pareto_points]
        pareto_dists = [r.distance for r in pareto_points]
        ax2.plot(pareto_times, pareto_dists, 'k--', alpha=0.5, linewidth=1.5, 
                label='Pareto frontier')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_plotly(results: List[VariantResult], save_path: str = None) -> go.Figure:
    """Create interactive Plotly visualization."""
    family_colors = {
        'Genetic Algorithm': '#3498db',
        'Ant Colony': '#e74c3c',
        'Hummingbird': '#2ecc71',
        'Particle Swarm': '#9b59b6',
        'Baseline': '#95a5a6'
    }
    
    sorted_results = sorted(results, key=lambda r: r.distance)
    best_distance = sorted_results[0].distance
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'bar'}, {'type': 'scatter'}]],
        subplot_titles=['Tour Distance by Variant', 'Time vs Quality Trade-off'],
        column_widths=[0.5, 0.5]
    )
    
    # Left: Bar chart
    for family, color in family_colors.items():
        family_results = [r for r in sorted_results if r.family == family]
        if family_results:
            variants = [r.variant for r in family_results]
            distances = [r.distance for r in family_results]
            gaps = [(r.distance - best_distance) / best_distance * 100 for r in family_results]
            
            hover_text = [f"{v}<br>Distance: {d:.2f}<br>Gap: {g:.1f}%<br>Time: {r.time_ms:.1f}ms" 
                         for v, d, g, r in zip(variants, distances, gaps, family_results)]
            
            fig.add_trace(
                go.Bar(
                    y=variants,
                    x=distances,
                    orientation='h',
                    name=family,
                    marker_color=color,
                    text=[f'{d:.1f}' for d in distances],
                    textposition='outside',
                    hovertext=hover_text,
                    hoverinfo='text'
                ),
                row=1, col=1
            )
    
    # Right: Scatter plot
    for family, color in family_colors.items():
        family_results = [r for r in results if r.family == family]
        if family_results:
            times = [r.time_ms for r in family_results]
            distances = [r.distance for r in family_results]
            variants = [r.variant for r in family_results]
            gaps = [(r.distance - best_distance) / best_distance * 100 for r in family_results]
            
            hover_text = [f"<b>{v}</b><br>Distance: {d:.2f}<br>Gap: {g:.1f}%<br>Time: {t:.1f}ms" 
                         for v, d, g, t in zip(variants, distances, gaps, times)]
            
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=distances,
                    mode='markers+text',
                    name=family,
                    marker=dict(size=15, color=color, line=dict(width=1, color='black')),
                    text=variants,
                    textposition='top center',
                    textfont=dict(size=9),
                    hovertext=hover_text,
                    hoverinfo='text'
                ),
                row=1, col=2
            )
    
    fig.update_layout(
        title=dict(text='CETSP Algorithm Variant Comparison', font=dict(size=16)),
        height=500,
        width=1200,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    fig.update_xaxes(title_text='Total Distance', row=1, col=1)
    fig.update_yaxes(autorange='reversed', row=1, col=1)
    fig.update_xaxes(title_text='Computation Time (ms)', row=1, col=2)
    fig.update_yaxes(title_text='Tour Distance', row=1, col=2)
    
    if save_path:
        fig.write_html(save_path)
        print(f"  Saved: {save_path}")
    
    return fig


def plot_family_comparison(results: List[VariantResult], save_path: str = None) -> go.Figure:
    """Create family comparison with box plots showing variant ranges."""
    family_colors = {
        'Genetic Algorithm': '#3498db',
        'Ant Colony': '#e74c3c',
        'Hummingbird': '#2ecc71',
        'Particle Swarm': '#9b59b6',
    }
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Genetic Algorithm', 'Ant Colony Optimization', 
                       'Artificial Hummingbird', 'Particle Swarm Optimization'],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    family_positions = {
        'Genetic Algorithm': (1, 1),
        'Ant Colony': (1, 2),
        'Hummingbird': (2, 1),
        'Particle Swarm': (2, 2),
    }
    
    best_distance = min(r.distance for r in results)
    
    for family, (row, col) in family_positions.items():
        family_results = [r for r in results if r.family == family]
        if family_results:
            variants = [r.variant for r in family_results]
            distances = [r.distance for r in family_results]
            times = [r.time_ms for r in family_results]
            gaps = [(d - best_distance) / best_distance * 100 for d in distances]
            
            color = family_colors.get(family, '#95a5a6')
            
            # Bar chart for distances
            fig.add_trace(
                go.Bar(
                    x=variants,
                    y=distances,
                    marker_color=color,
                    text=[f'{d:.1f}<br>({g:.1f}%)' for d, g in zip(distances, gaps)],
                    textposition='outside',
                    showlegend=False,
                    hovertemplate='%{x}<br>Distance: %{y:.2f}<extra></extra>'
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title=dict(text='Variant Comparison by Algorithm Family', font=dict(size=16)),
        height=600,
        width=1000,
    )
    
    fig.update_yaxes(title_text='Distance', row=1, col=1)
    fig.update_yaxes(title_text='Distance', row=2, col=1)
    
    if save_path:
        fig.write_html(save_path)
        print(f"  Saved: {save_path}")
    
    return fig


def create_summary_table(results: List[VariantResult], save_path: str = None) -> go.Figure:
    """Create summary table as Plotly figure."""
    sorted_results = sorted(results, key=lambda r: r.distance)
    best_distance = sorted_results[0].distance
    
    # Prepare table data
    ranks = list(range(1, len(sorted_results) + 1))
    families = [r.family for r in sorted_results]
    variants = [r.variant for r in sorted_results]
    distances = [f'{r.distance:.2f}' for r in sorted_results]
    times = [f'{r.time_ms:.1f}' for r in sorted_results]
    gaps = [f'{(r.distance - best_distance) / best_distance * 100:.1f}%' for r in sorted_results]
    
    # Color coding
    fill_colors = []
    for i, r in enumerate(sorted_results):
        if i == 0:
            fill_colors.append('#d5f5e3')  # Best - green
        elif (r.distance - best_distance) / best_distance < 0.02:
            fill_colors.append('#d5f5e3')  # Within 2% - light green
        elif (r.distance - best_distance) / best_distance < 0.05:
            fill_colors.append('#fcf3cf')  # Within 5% - yellow
        else:
            fill_colors.append('white')
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Rank</b>', '<b>Family</b>', '<b>Variant</b>', 
                   '<b>Distance</b>', '<b>Time (ms)</b>', '<b>Gap</b>'],
            fill_color='#3498db',
            font=dict(color='white', size=12),
            align='center',
            height=35
        ),
        cells=dict(
            values=[ranks, families, variants, distances, times, gaps],
            fill_color=[fill_colors] * 6,
            font=dict(size=11),
            align='center',
            height=30
        )
    )])
    
    fig.update_layout(
        title=dict(text='Algorithm Variant Rankings', font=dict(size=16)),
        height=400 + len(results) * 25,
        width=800
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"  Saved: {save_path}")
    
    return fig


def plot_individual_tours_matplotlib(cetsp: CETSP, results: List[VariantResult], 
                                      save_path: str = None) -> plt.Figure:
    """Create a grid of 3D tour plots for each variant."""
    family_colors = {
        'Genetic Algorithm': '#3498db',
        'Ant Colony': '#e74c3c',
        'Hummingbird': '#2ecc71',
        'Particle Swarm': '#9b59b6',
        'Baseline': '#95a5a6'
    }
    
    # Sort by distance
    sorted_results = sorted(results, key=lambda r: r.distance)
    best_distance = sorted_results[0].distance
    
    # Create grid layout (4 columns)
    n_results = len(sorted_results)
    n_cols = 4
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    
    for idx, result in enumerate(sorted_results):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        
        color = family_colors.get(result.family, '#95a5a6')
        gap = (result.distance - best_distance) / best_distance * 100
        
        # Plot customer coverage spheres (simplified)
        for c in cetsp.customers:
            # Plot just the center point
            ax.scatter(c.x, c.y, c.z, c='blue', s=30, alpha=0.6)
            
            # Draw a small circle to represent coverage
            u = np.linspace(0, 2 * np.pi, 20)
            x_circle = c.x + c.radius * np.cos(u)
            y_circle = c.y + c.radius * np.sin(u)
            z_circle = np.full_like(u, c.z)
            ax.plot(x_circle, y_circle, z_circle, 'b-', alpha=0.2, linewidth=0.5)
        
        # Plot depot
        ax.scatter(cetsp.depot.x, cetsp.depot.y, cetsp.depot.z, 
                  c='red', s=150, marker='D', zorder=5, label='Depot')
        
        # Plot tour path
        solution = result.solution
        if solution.path.waypoints:
            wps = solution.path.waypoints
            xs = [w[0] for w in wps]
            ys = [w[1] for w in wps]
            zs = [w[2] for w in wps]
            
            ax.plot(xs, ys, zs, c=color, linewidth=2.5, alpha=0.9)
            ax.scatter(xs[1:-1], ys[1:-1], zs[1:-1], c=color, s=40, zorder=4)
        
        # Set labels and title
        rank_str = "[1st] " if idx == 0 else f"#{idx+1} "
        gap_str = "(BEST)" if gap == 0 else f"(+{gap:.1f}%)"
        ax.set_title(f"{rank_str}{result.variant}\nDist: {result.distance:.1f} {gap_str}", 
                    fontsize=10, fontweight='bold' if idx == 0 else 'normal')
        
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 50)
        ax.tick_params(labelsize=7)
    
    plt.suptitle('Tour Visualization for Each Variant (sorted by distance)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_individual_tours_plotly(cetsp: CETSP, results: List[VariantResult],
                                  save_path: str = None) -> go.Figure:
    """Create interactive Plotly 3D tour plots for each variant."""
    family_colors = {
        'Genetic Algorithm': '#3498db',
        'Ant Colony': '#e74c3c',
        'Hummingbird': '#2ecc71',
        'Particle Swarm': '#9b59b6',
        'Baseline': '#95a5a6'
    }
    
    # Sort by distance
    sorted_results = sorted(results, key=lambda r: r.distance)
    best_distance = sorted_results[0].distance
    
    # Create grid layout
    n_results = len(sorted_results)
    n_cols = 4
    n_rows = (n_results + n_cols - 1) // n_cols
    
    # Create subplot titles
    titles = []
    for idx, r in enumerate(sorted_results):
        gap = (r.distance - best_distance) / best_distance * 100
        gap_str = "(best)" if gap == 0 else f"(+{gap:.1f}%)"
        rank = "🏆" if idx == 0 else f"#{idx+1}"
        titles.append(f"{rank} {r.variant}<br>Dist: {r.distance:.1f} {gap_str}")
    
    # Pad with empty titles if needed
    while len(titles) < n_rows * n_cols:
        titles.append("")
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[[{'type': 'scatter3d'} for _ in range(n_cols)] for _ in range(n_rows)],
        subplot_titles=titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.08
    )
    
    for idx, result in enumerate(sorted_results):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        color = family_colors.get(result.family, '#95a5a6')
        
        # Customer points
        fig.add_trace(
            go.Scatter3d(
                x=[c.x for c in cetsp.customers],
                y=[c.y for c in cetsp.customers],
                z=[c.z for c in cetsp.customers],
                mode='markers',
                marker=dict(size=4, color='blue', opacity=0.6),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        # Coverage spheres (simplified as points with size)
        for c in cetsp.customers:
            # Create a simple sphere approximation
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 10)
            x_sphere = c.x + c.radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = c.y + c.radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = c.z + c.radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig.add_trace(
                go.Surface(
                    x=x_sphere, y=y_sphere, z=z_sphere,
                    colorscale=[[0, 'rgba(100,150,255,0.1)'], [1, 'rgba(100,150,255,0.1)']],
                    showscale=False,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
        
        # Depot
        fig.add_trace(
            go.Scatter3d(
                x=[cetsp.depot.x],
                y=[cetsp.depot.y],
                z=[cetsp.depot.z],
                mode='markers',
                marker=dict(size=10, color='red', symbol='diamond'),
                showlegend=False,
                hovertemplate='Depot<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Tour path
        solution = result.solution
        if solution.path.waypoints:
            wps = solution.path.waypoints
            xs = [w[0] for w in wps]
            ys = [w[1] for w in wps]
            zs = [w[2] for w in wps]
            
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='lines+markers',
                    line=dict(color=color, width=5),
                    marker=dict(size=4, color=color),
                    showlegend=False,
                    hovertemplate='%{x:.1f}, %{y:.1f}, %{z:.1f}<extra></extra>'
                ),
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        title=dict(text='Tour Visualization for Each Variant', font=dict(size=18)),
        height=400 * n_rows,
        width=350 * n_cols,
        showlegend=False
    )
    
    # Update all scenes
    for i in range(n_results):
        scene_name = 'scene' if i == 0 else f'scene{i+1}'
        fig.update_layout(**{
            scene_name: dict(
                xaxis=dict(range=[0, 100], title='X'),
                yaxis=dict(range=[0, 100], title='Y'),
                zaxis=dict(range=[0, 50], title='Z'),
                aspectmode='manual',
                aspectratio=dict(x=2, y=2, z=1)
            )
        })
    
    if save_path:
        fig.write_html(save_path)
        print(f"  Saved: {save_path}")
    
    return fig


def plot_top_tours_comparison(cetsp: CETSP, results: List[VariantResult],
                              top_n: int = 4, save_path: str = None) -> go.Figure:
    """Create side-by-side comparison of top N tours."""
    family_colors = {
        'Genetic Algorithm': '#3498db',
        'Ant Colony': '#e74c3c',
        'Hummingbird': '#2ecc71',
        'Particle Swarm': '#9b59b6',
        'Baseline': '#95a5a6'
    }
    
    # Get top N results
    sorted_results = sorted(results, key=lambda r: r.distance)[:top_n]
    best_distance = sorted_results[0].distance
    
    # Create subplot titles
    titles = []
    for idx, r in enumerate(sorted_results):
        gap = (r.distance - best_distance) / best_distance * 100
        gap_str = "BEST" if gap == 0 else f"+{gap:.1f}%"
        titles.append(f"#{idx+1} {r.variant} ({r.family})<br>Distance: {r.distance:.1f} ({gap_str})")
    
    fig = make_subplots(
        rows=1, cols=top_n,
        specs=[[{'type': 'scatter3d'} for _ in range(top_n)]],
        subplot_titles=titles,
        horizontal_spacing=0.02
    )
    
    for idx, result in enumerate(sorted_results):
        col = idx + 1
        color = family_colors.get(result.family, '#95a5a6')
        
        # Customer spheres
        for c in cetsp.customers:
            u = np.linspace(0, 2 * np.pi, 12)
            v = np.linspace(0, np.pi, 8)
            x_sphere = c.x + c.radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = c.y + c.radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = c.z + c.radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig.add_trace(
                go.Surface(
                    x=x_sphere, y=y_sphere, z=z_sphere,
                    colorscale=[[0, 'rgba(52,152,219,0.15)'], [1, 'rgba(52,152,219,0.15)']],
                    showscale=False, showlegend=False, hoverinfo='skip'
                ),
                row=1, col=col
            )
        
        # Customer centers
        fig.add_trace(
            go.Scatter3d(
                x=[c.x for c in cetsp.customers],
                y=[c.y for c in cetsp.customers],
                z=[c.z for c in cetsp.customers],
                mode='markers+text',
                marker=dict(size=4, color='blue'),
                text=[str(c.id) for c in cetsp.customers],
                textposition='top center',
                textfont=dict(size=8),
                showlegend=False
            ),
            row=1, col=col
        )
        
        # Depot
        fig.add_trace(
            go.Scatter3d(
                x=[cetsp.depot.x], y=[cetsp.depot.y], z=[cetsp.depot.z],
                mode='markers',
                marker=dict(size=12, color='red', symbol='diamond'),
                name='Depot' if col == 1 else None,
                showlegend=(col == 1)
            ),
            row=1, col=col
        )
        
        # Tour path
        solution = result.solution
        if solution.path.waypoints:
            wps = solution.path.waypoints
            
            fig.add_trace(
                go.Scatter3d(
                    x=[w[0] for w in wps],
                    y=[w[1] for w in wps],
                    z=[w[2] for w in wps],
                    mode='lines+markers',
                    line=dict(color=color, width=6),
                    marker=dict(size=5, color=color),
                    name=result.variant if col == 1 else None,
                    showlegend=False
                ),
                row=1, col=col
            )
    
    fig.update_layout(
        title=dict(text=f'Top {top_n} Tour Comparison', font=dict(size=18)),
        height=600,
        width=400 * top_n,
        legend=dict(x=0.01, y=0.99)
    )
    
    for i in range(top_n):
        scene_name = 'scene' if i == 0 else f'scene{i+1}'
        fig.update_layout(**{
            scene_name: dict(
                xaxis=dict(range=[0, 100], title='X'),
                yaxis=dict(range=[0, 100], title='Y'),
                zaxis=dict(range=[0, 50], title='Z'),
                aspectmode='data'
            )
        })
    
    if save_path:
        fig.write_html(save_path)
        print(f"  Saved: {save_path}")
    
    return fig


def main():
    print("=" * 80)
    print("CETSP ALGORITHM VARIANT COMPARISON")
    print("=" * 80)
    
    # Configuration
    N_CUSTOMERS = 30
    SEED = 42
    
    # Create problem
    print(f"\n[SETUP] Creating problem with {N_CUSTOMERS} customers (seed={SEED})...")
    cetsp = create_problem(n_customers=N_CUSTOMERS, seed=SEED)
    print(f"        Depot: ({cetsp.depot.x:.1f}, {cetsp.depot.y:.1f}, {cetsp.depot.z:.1f})")
    print(f"        Customers: {len(cetsp.customers)}")
    print(f"        Avg radius: {np.mean([c.radius for c in cetsp.customers]):.2f}")
    
    # Run all variants
    results = run_all_variants(cetsp, N_CUSTOMERS, SEED)
    
    # Print summary
    print_summary(results)
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("VISUALIZATIONS")
    print("=" * 80)
    
    output_dir = Path(__file__).parent
    
    print("\n[Matplotlib - Comparison]")
    plot_matplotlib(results, save_path=output_dir / 'variant_comparison.png')
    
    print("\n[Matplotlib - All Tours]")
    plot_individual_tours_matplotlib(cetsp, results, 
                                     save_path=output_dir / 'variant_all_tours.png')
    
    print("\n[Plotly - Main]")
    fig_main = plot_plotly(results, save_path=output_dir / 'variant_comparison.html')
    
    print("\n[Plotly - Family]")
    plot_family_comparison(results, save_path=output_dir / 'variant_family_comparison.html')
    
    print("\n[Plotly - Table]")
    create_summary_table(results, save_path=output_dir / 'variant_ranking_table.html')
    
    print("\n[Plotly - All Tours]")
    plot_individual_tours_plotly(cetsp, results, 
                                 save_path=output_dir / 'variant_all_tours.html')
    
    print("\n[Plotly - Top 4 Tours]")
    fig_tours = plot_top_tours_comparison(cetsp, results, top_n=4,
                                          save_path=output_dir / 'variant_top_tours.html')
    
    # Show plots
    print("\n" + "=" * 80)
    print("Opening interactive plots...")
    fig_tours.show()
    fig_main.show()
    plt.show()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
