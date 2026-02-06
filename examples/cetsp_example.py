"""
Close-Enough Travelling Salesman Problem (CETSP) Demo

This script demonstrates solving a 3D CETSP using different algorithms
and visualizing the results with both Matplotlib and Plotly.

In CETSP, the salesman doesn't need to visit exact locations - they only
need to get within a specified "coverage radius" of each customer.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.cetsp import CETSPNode, CETSP, GreedyCETSP, AStarCETSP
from src.cetsp.solvers import TwoOptCETSP


def create_sample_problem(num_customers: int = 12, seed: int = 42):
    """Create a sample 3D CETSP problem."""
    np.random.seed(seed)
    
    # Create depot at center (no coverage radius - must start/end exactly here)
    depot = CETSPNode(id=0, x=50, y=50, z=25, radius=0)
    
    # Create customers with random 3D positions and coverage radii
    customers = []
    for i in range(1, num_customers + 1):
        customer = CETSPNode(
            id=i,
            x=np.random.uniform(0, 100),
            y=np.random.uniform(0, 100),
            z=np.random.uniform(0, 50),
            radius=np.random.uniform(3, 8),  # Coverage radius between 3-8 units
            priority=np.random.uniform(0.5, 1.5)
        )
        customers.append(customer)
    
    return depot, customers


def print_customer_info(customers: list[CETSPNode]):
    """Print customer information."""
    print("\n   Customer Details:")
    print("   " + "-" * 50)
    print(f"   {'ID':>3} | {'Position':^20} | {'Radius':>6}")
    print("   " + "-" * 50)
    for c in customers[:5]:  # Show first 5
        print(f"   {c.id:>3} | ({c.x:5.1f}, {c.y:5.1f}, {c.z:5.1f}) | {c.radius:6.2f}")
    if len(customers) > 5:
        print(f"   ... and {len(customers) - 5} more customers")


def create_sphere_wireframe(center, radius, n_points=20):
    """Create wireframe sphere coordinates for matplotlib."""
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def plot_matplotlib_3d(problem, solutions_dict, depot, customers, save_path=None):
    """
    Plot all solutions using Matplotlib 3D.
    
    Args:
        problem: CETSP problem instance
        solutions_dict: Dictionary of {name: solution}
        depot: Depot node
        customers: List of customer nodes
        save_path: Optional path to save the figure
    """
    n_solutions = len(solutions_dict)
    fig = plt.figure(figsize=(6 * n_solutions, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    
    for idx, (name, solution) in enumerate(solutions_dict.items()):
        ax = fig.add_subplot(1, n_solutions, idx + 1, projection='3d')
        
        # Plot customer coverage spheres (wireframe)
        for c in customers:
            x, y, z = create_sphere_wireframe(c.position, c.radius, n_points=15)
            ax.plot_surface(x, y, z, alpha=0.1, color='blue', linewidth=0)
            # Plot customer center
            ax.scatter(*c.position, c='blue', s=30, marker='o', alpha=0.7)
            ax.text(c.x, c.y, c.z + 2, str(c.id), fontsize=8, ha='center')
        
        # Plot depot
        ax.scatter(*depot.position, c='red', s=150, marker='D', label='Depot', zorder=5)
        
        # Plot tour path
        waypoints = solution.path.waypoints
        if len(waypoints) > 1:
            path_x = [wp[0] for wp in waypoints]
            path_y = [wp[1] for wp in waypoints]
            path_z = [wp[2] for wp in waypoints]
            ax.plot(path_x, path_y, path_z, c=colors[idx % len(colors)], 
                    linewidth=2.5, label=f'Tour ({solution.total_distance:.1f})')
            
            # Plot waypoints
            ax.scatter(path_x[1:-1], path_y[1:-1], path_z[1:-1], 
                      c=colors[idx % len(colors)], s=50, marker='o', alpha=0.8)
        
        # Plot coverage connections (waypoint to customer center)
        covered = solution.path.covered_nodes
        for i, (wp, node) in enumerate(zip(waypoints[1:], covered)):
            if node:
                ax.plot([wp[0], node.x], [wp[1], node.y], [wp[2], node.z],
                       'orange', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{name}\nDistance: {solution.total_distance:.2f}')
        ax.legend(loc='upper left', fontsize=8)
        
        # Set equal aspect ratio
        max_range = max(100, 50, 50) / 2
        mid_x, mid_y, mid_z = 50, 50, 25
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(0, 50)
    
    plt.suptitle('CETSP Solutions Comparison (Matplotlib)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Matplotlib figure saved: {save_path}")
    
    return fig


def plot_plotly_3d(problem, solutions_dict, depot, customers, save_path=None):
    """
    Plot all solutions using Plotly 3D.
    
    Args:
        problem: CETSP problem instance
        solutions_dict: Dictionary of {name: solution}
        depot: Depot node
        customers: List of customer nodes
        save_path: Optional path to save HTML file
    """
    n_solutions = len(solutions_dict)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=n_solutions,
        specs=[[{'type': 'scatter3d'} for _ in range(n_solutions)]],
        subplot_titles=list(solutions_dict.keys()),
        horizontal_spacing=0.02
    )
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    
    for idx, (name, solution) in enumerate(solutions_dict.items()):
        col = idx + 1
        show_legend = (idx == 0)  # Only show legend for first subplot
        
        # Create sphere surfaces for customers
        for c in customers:
            # Create sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = c.x + c.radius * np.outer(np.cos(u), np.sin(v))
            y = c.y + c.radius * np.outer(np.sin(u), np.sin(v))
            z = c.z + c.radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig.add_trace(
                go.Surface(
                    x=x, y=y, z=z,
                    colorscale=[[0, 'rgba(52, 152, 219, 0.3)'], [1, 'rgba(52, 152, 219, 0.3)']],
                    showscale=False,
                    name=f'Customer {c.id}',
                    hoverinfo='name',
                    showlegend=False
                ),
                row=1, col=col
            )
        
        # Customer centers
        fig.add_trace(
            go.Scatter3d(
                x=[c.x for c in customers],
                y=[c.y for c in customers],
                z=[c.z for c in customers],
                mode='markers+text',
                marker=dict(size=5, color='blue', symbol='circle'),
                text=[str(c.id) for c in customers],
                textposition='top center',
                name='Customers',
                showlegend=show_legend
            ),
            row=1, col=col
        )
        
        # Depot
        fig.add_trace(
            go.Scatter3d(
                x=[depot.x], y=[depot.y], z=[depot.z],
                mode='markers',
                marker=dict(size=12, color='red', symbol='diamond'),
                name='Depot',
                showlegend=show_legend
            ),
            row=1, col=col
        )
        
        # Tour path
        waypoints = solution.path.waypoints
        if len(waypoints) > 1:
            fig.add_trace(
                go.Scatter3d(
                    x=[wp[0] for wp in waypoints],
                    y=[wp[1] for wp in waypoints],
                    z=[wp[2] for wp in waypoints],
                    mode='lines+markers',
                    line=dict(color=colors[idx % len(colors)], width=5),
                    marker=dict(size=4, color=colors[idx % len(colors)]),
                    name=f'Tour ({solution.total_distance:.1f})',
                    showlegend=show_legend
                ),
                row=1, col=col
            )
        
        # Coverage connections
        covered = solution.path.covered_nodes
        for i, (wp, node) in enumerate(zip(waypoints[1:], covered)):
            if node:
                fig.add_trace(
                    go.Scatter3d(
                        x=[wp[0], node.x],
                        y=[wp[1], node.y],
                        z=[wp[2], node.z],
                        mode='lines',
                        line=dict(color='orange', width=2, dash='dash'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=col
                )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='CETSP Solutions Comparison (Plotly)',
            font=dict(size=20)
        ),
        height=600,
        width=400 * n_solutions,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    # Update all scene axes
    for i in range(n_solutions):
        scene_name = 'scene' if i == 0 else f'scene{i+1}'
        fig.update_layout(**{
            scene_name: dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            )
        })
    
    if save_path:
        fig.write_html(save_path)
        print(f"   Plotly figure saved: {save_path}")
    
    return fig


def plot_comparison_bar_charts(solutions_dict, save_matplotlib=None, save_plotly=None):
    """
    Create comparison bar charts for distance and time.
    """
    names = list(solutions_dict.keys())
    distances = [sol.total_distance for sol in solutions_dict.values()]
    times = [sol.computation_time * 1000 for sol in solutions_dict.values()]  # ms
    
    # Matplotlib version
    fig_mpl, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Distance bar chart
    bars1 = axes[0].bar(names, distances, color=colors[:len(names)], edgecolor='black')
    axes[0].set_ylabel('Distance')
    axes[0].set_title('Total Tour Distance')
    axes[0].tick_params(axis='x', rotation=15)
    for bar, dist in zip(bars1, distances):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{dist:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Time bar chart
    bars2 = axes[1].bar(names, times, color=colors[:len(names)], edgecolor='black')
    axes[1].set_ylabel('Time (ms)')
    axes[1].set_title('Computation Time')
    axes[1].tick_params(axis='x', rotation=15)
    for bar, t in zip(bars2, times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{t:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('CETSP Solutions Comparison', fontsize=14)
    plt.tight_layout()
    
    if save_matplotlib:
        plt.savefig(save_matplotlib, dpi=150, bbox_inches='tight')
        print(f"   Matplotlib comparison saved: {save_matplotlib}")
    
    # Plotly version
    fig_plotly = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Total Tour Distance', 'Computation Time (ms)']
    )
    
    fig_plotly.add_trace(
        go.Bar(x=names, y=distances, marker_color=colors[:len(names)],
               text=[f'{d:.1f}' for d in distances], textposition='outside',
               name='Distance'),
        row=1, col=1
    )
    
    fig_plotly.add_trace(
        go.Bar(x=names, y=times, marker_color=colors[:len(names)],
               text=[f'{t:.1f}' for t in times], textposition='outside',
               name='Time (ms)'),
        row=1, col=2
    )
    
    fig_plotly.update_layout(
        title='CETSP Solutions Comparison',
        height=500,
        width=900,
        showlegend=False
    )
    
    if save_plotly:
        fig_plotly.write_html(save_plotly)
        print(f"   Plotly comparison saved: {save_plotly}")
    
    return fig_mpl, fig_plotly


def main():
    print("=" * 60)
    print("Close-Enough Travelling Salesman Problem (CETSP) Solver")
    print("=" * 60)
    
    # Create sample problem
    print("\n📦 Creating sample problem...")
    depot, customers = create_sample_problem(num_customers=30, seed=123)
    
    print(f"   Depot: ({depot.x:.1f}, {depot.y:.1f}, {depot.z:.1f})")
    print(f"   Customers: {len(customers)}")
    print(f"   Average coverage radius: {np.mean([c.radius for c in customers]):.2f}")
    
    print_customer_info(customers)
    
    # Create CETSP problem
    problem = CETSP(depot, customers, return_to_depot=True)
    
    # Calculate TSP distance (no coverage) vs potential CETSP savings
    total_center_dist = sum(
        problem.get_center_distance(problem.all_nodes[i], problem.all_nodes[i+1])
        for i in range(len(problem.all_nodes) - 1)
    )
    total_radii = sum(c.radius for c in customers) * 2  # Entry and exit savings
    
    print(f"\n   Potential savings from coverage radii: ~{total_radii:.1f} units")
    
    # Solve with different methods
    print("\n🔧 Solving with different algorithms...")
    
    # 1. Greedy (Nearest Neighbor)
    print("\n   [1] Greedy Nearest Neighbor:")
    greedy_solution = problem.solve(method="greedy")
    print(f"       Total distance: {greedy_solution.total_distance:.2f}")
    print(f"       Waypoints: {greedy_solution.path.num_waypoints}")
    print(f"       Time: {greedy_solution.computation_time*1000:.2f} ms")
    
    # 2. A* Search
    print("\n   [2] A* Search Algorithm:")
    astar_solution = problem.solve(method="astar")
    print(f"       Total distance: {astar_solution.total_distance:.2f}")
    print(f"       Waypoints: {astar_solution.path.num_waypoints}")
    print(f"       Time: {astar_solution.computation_time*1000:.2f} ms")
    
    # 3. Improve best solution with 2-opt
    print("\n🔄 Improving best solution with 2-opt...")
    best_solution = min([greedy_solution, astar_solution], key=lambda s: s.total_distance)
    
    two_opt = TwoOptCETSP(problem)
    improved_solution = two_opt.improve(best_solution)
    
    print(f"   Before: {best_solution.total_distance:.2f}")
    print(f"   After:  {improved_solution.total_distance:.2f}")
    improvement = best_solution.total_distance - improved_solution.total_distance
    print(f"   Improvement: {improvement:.2f} ({improvement/best_solution.total_distance*100:.1f}%)")
    
    # Print path details
    print("\n📋 Solution Path:")
    waypoints = improved_solution.path.waypoints
    covered = improved_solution.path.covered_nodes
    
    print(f"   Start: Depot ({depot.x:.1f}, {depot.y:.1f}, {depot.z:.1f})")
    
    for i, (wp, node) in enumerate(zip(waypoints[1:-1], covered), 1):
        if node:
            dist_to_center = np.linalg.norm(wp - node.position)
            print(f"   {i}. Customer {node.id}: entered at ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f})")
            print(f"      Distance to center: {dist_to_center:.2f} (radius: {node.radius:.2f})")
    
    print(f"   End: Return to Depot")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    
    # Compare with standard TSP (visiting centers)
    standard_tsp_dist = 0
    positions = [depot.position] + [c.position for c in customers] + [depot.position]
    for i in range(len(positions) - 1):
        standard_tsp_dist += np.linalg.norm(positions[i+1] - positions[i])
    
    print(f"   Standard TSP distance (centers): ~{standard_tsp_dist:.2f}")
    print(f"   CETSP distance (with coverage):  {improved_solution.total_distance:.2f}")
    savings = standard_tsp_dist - improved_solution.total_distance
    print(f"   Savings from coverage radii: {savings:.2f} ({savings/standard_tsp_dist*100:.1f}%)")
    print(f"   Solution complete: {improved_solution.is_complete}")
    
    # =========================================================================
    # VISUALIZATION SECTION
    # =========================================================================
    print("\n" + "=" * 60)
    print("🎨 Creating Visualizations (Matplotlib & Plotly)")
    print("=" * 60)
    
    # Collect all solutions
    solutions_dict = {
        'Greedy': greedy_solution,
        'A*': astar_solution,
        'A* + 2-opt': improved_solution
    }
    
    # Create output directory
    output_dir = Path(__file__).parent
    
    # 1. Matplotlib 3D plots
    print("\n   [Matplotlib] Creating 3D tour plots...")
    fig_mpl_3d = plot_matplotlib_3d(
        problem, solutions_dict, depot, customers,
        save_path=output_dir / 'cetsp_tours_matplotlib.png'
    )
    
    # 2. Plotly 3D plots
    print("\n   [Plotly] Creating interactive 3D tour plots...")
    fig_plotly_3d = plot_plotly_3d(
        problem, solutions_dict, depot, customers,
        save_path=output_dir / 'cetsp_tours_plotly.html'
    )
    
    # 3. Comparison bar charts
    print("\n   Creating comparison bar charts...")
    fig_mpl_comp, fig_plotly_comp = plot_comparison_bar_charts(
        solutions_dict,
        save_matplotlib=output_dir / 'cetsp_comparison_matplotlib.png',
        save_plotly=output_dir / 'cetsp_comparison_plotly.html'
    )
    
    # 4. Individual tour plots with Plotly (enhanced version)
    print("\n   [Plotly] Creating individual enhanced tour plots...")
    for name, solution in solutions_dict.items():
        safe_name = name.lower().replace(' ', '_').replace('+', '_').replace('*', 'star')
        
        fig_single = go.Figure()
        
        # Add customer spheres
        for c in customers:
            u = np.linspace(0, 2 * np.pi, 25)
            v = np.linspace(0, np.pi, 25)
            x = c.x + c.radius * np.outer(np.cos(u), np.sin(v))
            y = c.y + c.radius * np.outer(np.sin(u), np.sin(v))
            z = c.z + c.radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig_single.add_trace(go.Surface(
                x=x, y=y, z=z,
                colorscale=[[0, 'rgba(52, 152, 219, 0.25)'], [1, 'rgba(52, 152, 219, 0.25)']],
                showscale=False,
                name=f'Customer {c.id}',
                hoverinfo='name'
            ))
        
        # Customer centers with labels
        fig_single.add_trace(go.Scatter3d(
            x=[c.x for c in customers],
            y=[c.y for c in customers],
            z=[c.z for c in customers],
            mode='markers+text',
            marker=dict(size=6, color='blue'),
            text=[str(c.id) for c in customers],
            textposition='top center',
            name='Customers'
        ))
        
        # Depot
        fig_single.add_trace(go.Scatter3d(
            x=[depot.x], y=[depot.y], z=[depot.z],
            mode='markers',
            marker=dict(size=14, color='red', symbol='diamond'),
            name='Depot'
        ))
        
        # Tour path
        waypoints = solution.path.waypoints
        fig_single.add_trace(go.Scatter3d(
            x=[wp[0] for wp in waypoints],
            y=[wp[1] for wp in waypoints],
            z=[wp[2] for wp in waypoints],
            mode='lines+markers',
            line=dict(color='#2ecc71', width=6),
            marker=dict(size=5, color='#2ecc71'),
            name=f'Tour'
        ))
        
        # Coverage connections
        covered = solution.path.covered_nodes
        for wp, node in zip(waypoints[1:], covered):
            if node:
                fig_single.add_trace(go.Scatter3d(
                    x=[wp[0], node.x],
                    y=[wp[1], node.y],
                    z=[wp[2], node.z],
                    mode='lines',
                    line=dict(color='orange', width=3, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        fig_single.update_layout(
            title=dict(
                text=f'{name} Solution<br><sup>Distance: {solution.total_distance:.2f} | Time: {solution.computation_time*1000:.2f}ms</sup>',
                font=dict(size=18)
            ),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y', 
                zaxis_title='Z',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=700,
            width=900,
            legend=dict(x=0.01, y=0.99)
        )
        
        fig_single.write_html(output_dir / f'cetsp_{safe_name}.html')
        print(f"      - Saved: cetsp_{safe_name}.html")
    
    # Show summary of files
    print("\n" + "=" * 60)
    print("📁 Generated Files")
    print("=" * 60)
    print("   Matplotlib:")
    print("      - cetsp_tours_matplotlib.png (3D tours comparison)")
    print("      - cetsp_comparison_matplotlib.png (bar charts)")
    print("   Plotly (interactive HTML):")
    print("      - cetsp_tours_plotly.html (3D tours comparison)")
    print("      - cetsp_comparison_plotly.html (bar charts)")
    print("      - cetsp_greedy.html (Greedy solution)")
    print("      - cetsp_astar.html (A* solution)")
    print("      - cetsp_astar_2_opt.html (A* + 2-opt solution)")
    
    # Show interactive plots
    print("\n   Opening interactive Plotly plots in browser...")
    fig_plotly_3d.show()
    fig_plotly_comp.show()
    
    # Show matplotlib plots
    plt.show()
    
    print("\n✅ Done!")


def compare_tsp_vs_cetsp():
    """
    Demonstrate the difference between standard TSP and CETSP.
    """
    print("\n" + "=" * 60)
    print("TSP vs CETSP Comparison")
    print("=" * 60)
    
    # Same problem, different approaches
    depot = CETSPNode(id=0, x=0, y=0, z=0, radius=0)
    customers = [
        CETSPNode(id=1, x=20, y=0, z=0, radius=5),
        CETSPNode(id=2, x=20, y=20, z=0, radius=5),
        CETSPNode(id=3, x=0, y=20, z=0, radius=5),
    ]
    
    # Standard TSP (radius=0)
    print("\n📍 Standard TSP (must visit exact locations):")
    tsp_customers = [
        CETSPNode(id=c.id, x=c.x, y=c.y, z=c.z, radius=0)
        for c in customers
    ]
    tsp_problem = CETSP(depot, tsp_customers)
    tsp_solution = tsp_problem.solve(method="greedy")
    print(f"   Distance: {tsp_solution.total_distance:.2f}")
    
    # CETSP (with coverage radii)
    print("\n🎯 CETSP (only need to get within radius):")
    cetsp_problem = CETSP(depot, customers)
    cetsp_solution = cetsp_problem.solve(method="astar")
    print(f"   Distance: {cetsp_solution.total_distance:.2f}")
    
    savings = tsp_solution.total_distance - cetsp_solution.total_distance
    print(f"\n   💰 CETSP saves: {savings:.2f} units ({savings/tsp_solution.total_distance*100:.1f}%)")


if __name__ == "__main__":
    main()
    
    # Uncomment to see TSP vs CETSP comparison
    # compare_tsp_vs_cetsp()
