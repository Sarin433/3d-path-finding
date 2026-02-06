"""Close-Enough Travelling Salesman Problem definition."""

from dataclasses import dataclass, field

import numpy as np

from .node import CETSPNode


@dataclass
class CETSPPath:
    """
    Represents a path/tour in the CETSP.

    The path consists of waypoints (actual positions visited) that
    provide coverage to the required nodes.
    """

    waypoints: list[np.ndarray] = field(default_factory=list)
    covered_nodes: list[CETSPNode] = field(default_factory=list)

    @property
    def total_distance(self) -> float:
        """Calculate total path distance."""
        if len(self.waypoints) < 2:
            return 0.0

        distance = 0.0
        for i in range(len(self.waypoints) - 1):
            distance += np.linalg.norm(self.waypoints[i + 1] - self.waypoints[i])
        return distance

    @property
    def num_waypoints(self) -> int:
        """Number of waypoints in the path."""
        return len(self.waypoints)

    def add_waypoint(self, point: np.ndarray, covered: CETSPNode | None = None):
        """Add a waypoint to the path."""
        self.waypoints.append(point.copy())
        if covered:
            self.covered_nodes.append(covered)

    def get_waypoint_positions(self) -> np.ndarray:
        """Return all waypoint positions as a 2D array."""
        if not self.waypoints:
            return np.array([])
        return np.array(self.waypoints)


@dataclass
class CETSPSolution:
    """
    Complete solution to a CETSP instance.
    """

    path: CETSPPath
    is_complete: bool = False
    computation_time: float = 0.0

    @property
    def total_distance(self) -> float:
        return self.path.total_distance

    @property
    def num_waypoints(self) -> int:
        return self.path.num_waypoints


class CETSP:
    """
    Close-Enough Travelling Salesman Problem in 3D space.

    In CETSP, the salesman must visit a set of customers, but instead of
    visiting the exact location, they only need to get "close enough"
    (within a specified radius) to each customer.

    This is useful for:
    - Drone delivery with drop zones
    - Wireless coverage optimization
    - Sensor placement and data collection
    - Service vehicles with proximity-based service

    Example:
        >>> depot = CETSPNode(id=0, x=0, y=0, z=0, radius=0)
        >>> customers = [
        ...     CETSPNode(id=1, x=10, y=0, z=0, radius=3),
        ...     CETSPNode(id=2, x=10, y=10, z=0, radius=3),
        ... ]
        >>> problem = CETSP(depot, customers)
        >>> solution = problem.solve()
    """

    def __init__(
        self,
        depot: CETSPNode,
        customers: list[CETSPNode],
        return_to_depot: bool = True,
    ):
        """
        Initialize CETSP problem.

        Args:
            depot: Starting/ending location (typically radius=0).
            customers: List of customer nodes with coverage radii.
            return_to_depot: Whether the path must return to depot.
        """
        self.depot = depot
        self.customers = customers
        self.return_to_depot = return_to_depot
        self.all_nodes = [depot] + customers

        # Precompute distance matrix between node centers
        self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> None:
        """Compute distance matrix between all node centers."""
        n = len(self.all_nodes)
        self.distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = self.all_nodes[i].distance_to(self.all_nodes[j])
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist

    def get_center_distance(self, node1: CETSPNode, node2: CETSPNode) -> float:
        """Get distance between node centers."""
        idx1 = self.all_nodes.index(node1)
        idx2 = self.all_nodes.index(node2)
        return self.distance_matrix[idx1, idx2]

    def get_effective_distance(self, node1: CETSPNode, node2: CETSPNode) -> float:
        """
        Get effective travel distance between two nodes.

        This accounts for the coverage radii - the salesman can exit
        node1's coverage area and enter node2's coverage area at optimal points.
        """
        center_dist = self.get_center_distance(node1, node2)
        effective = center_dist - node1.radius - node2.radius
        return max(0.0, effective)

    def compute_optimal_waypoint(
        self, from_point: np.ndarray, target: CETSPNode, next_target: CETSPNode | None = None
    ) -> np.ndarray:
        """
        Compute optimal waypoint to cover target node.

        If next_target is provided, optimizes the waypoint position
        to minimize total travel to cover both.
        """
        if next_target is None:
            # Simple case: just get closest coverage point
            return target.coverage_entry_point(from_point)

        # Find point that minimizes: dist(from_point, waypoint) + dist(waypoint, next_coverage)
        # The optimal point lies on the line from from_point through target center
        # at the intersection with target's coverage sphere

        entry = target.coverage_entry_point(from_point)

        # Check if we can do better by going through the coverage area
        # towards the next target
        if target.radius > 0:
            # Direction to next target
            next_entry = next_target.coverage_entry_point(target.position)

            # Try exit point towards next target
            exit_direction = next_entry - target.position
            exit_dist = np.linalg.norm(exit_direction)

            if exit_dist > 0:
                exit_direction = exit_direction / exit_dist
                potential_exit = target.position + exit_direction * target.radius

                # Compare paths
                np.linalg.norm(entry - from_point) + np.linalg.norm(next_entry - entry)
                (
                    np.linalg.norm(entry - from_point)
                    + np.linalg.norm(potential_exit - entry)
                    + np.linalg.norm(next_entry - potential_exit)
                )

                # For now, return simple entry point
                # More sophisticated optimization could find the true optimal

        return entry

    def is_valid_solution(self, solution: CETSPSolution) -> bool:
        """Check if a solution covers all customers."""
        covered_ids = {node.id for node in solution.path.covered_nodes}
        required_ids = {c.id for c in self.customers}
        return required_ids.issubset(covered_ids)

    def solve(self, method: str = "greedy", **kwargs) -> CETSPSolution:
        """
        Solve the CETSP using specified method.

        Args:
            method: Solving method. Options:
                - "greedy": Nearest neighbor heuristic
                - "astar": A* search with greedy fallback
                - "genetic": Genetic Algorithm
                - "adaptive_genetic": Adaptive Genetic Algorithm
                - "aco": Ant Colony Optimization
                - "mmas": MAX-MIN Ant System
                - "acs": Ant Colony System
                - "aha": Artificial Hummingbird Algorithm
                - "enhanced_aha": Enhanced Artificial Hummingbird Algorithm
                - "pso": Particle Swarm Optimization
                - "adaptive_pso": Adaptive Particle Swarm Optimization
                - "discrete_pso": Discrete PSO with priority encoding
            **kwargs: Additional arguments passed to the solver.

        Returns:
            CETSPSolution with the computed path.
        """
        from .solvers import (
            AdaptiveGeneticCETSP,
            AdaptivePSOCETSP,
            AntColonyCETSP,
            AntColonySystemCETSP,
            ArtificialHummingbirdCETSP,
            AStarCETSP,
            DiscretePSOCETSP,
            EnhancedAHACETSP,
            GeneticCETSP,
            GreedyCETSP,
            MaxMinAntSystem,
            ParticleSwarmCETSP,
        )

        if method == "greedy":
            solver = GreedyCETSP(self)
        elif method == "astar":
            solver = AStarCETSP(self, **kwargs)
        elif method == "genetic":
            solver = GeneticCETSP(self, **kwargs)
        elif method == "adaptive_genetic":
            solver = AdaptiveGeneticCETSP(self, **kwargs)
        elif method == "aco":
            solver = AntColonyCETSP(self, **kwargs)
        elif method == "mmas":
            solver = MaxMinAntSystem(self, **kwargs)
        elif method == "acs":
            solver = AntColonySystemCETSP(self, **kwargs)
        elif method == "aha":
            solver = ArtificialHummingbirdCETSP(self, **kwargs)
        elif method == "enhanced_aha":
            solver = EnhancedAHACETSP(self, **kwargs)
        elif method == "pso":
            solver = ParticleSwarmCETSP(self, **kwargs)
        elif method == "adaptive_pso":
            solver = AdaptivePSOCETSP(self, **kwargs)
        elif method == "discrete_pso":
            solver = DiscretePSOCETSP(self, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        return solver.solve()

    def visualize(self, solution: CETSPSolution | None = None):
        """
        Create 3D visualization of the problem and solution.

        Returns:
            Plotly Figure object.
        """
        import plotly.graph_objects as go

        fig = go.Figure()

        # Plot depot
        fig.add_trace(
            go.Scatter3d(
                x=[self.depot.x],
                y=[self.depot.y],
                z=[self.depot.z],
                mode="markers",
                marker={"size": 12, "color": "red", "symbol": "diamond"},
                name="Depot",
            )
        )

        # Plot customers with coverage spheres
        for customer in self.customers:
            # Customer center
            fig.add_trace(
                go.Scatter3d(
                    x=[customer.x],
                    y=[customer.y],
                    z=[customer.z],
                    mode="markers",
                    marker={"size": 6, "color": "blue"},
                    name=f"Customer {customer.id}",
                    showlegend=False,
                )
            )

            # Coverage sphere (approximated with mesh)
            if customer.radius > 0:
                # Create sphere mesh
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 10)
                x = customer.x + customer.radius * np.outer(np.cos(u), np.sin(v))
                y = customer.y + customer.radius * np.outer(np.sin(u), np.sin(v))
                z = customer.z + customer.radius * np.outer(np.ones(np.size(u)), np.cos(v))

                fig.add_trace(
                    go.Surface(
                        x=x,
                        y=y,
                        z=z,
                        opacity=0.2,
                        colorscale=[[0, "lightblue"], [1, "lightblue"]],
                        showscale=False,
                        name=f"Coverage {customer.id}",
                        showlegend=False,
                    )
                )

        # Plot solution path
        if solution and solution.path.waypoints:
            waypoints = solution.path.get_waypoint_positions()

            fig.add_trace(
                go.Scatter3d(
                    x=waypoints[:, 0],
                    y=waypoints[:, 1],
                    z=waypoints[:, 2],
                    mode="lines+markers",
                    line={"color": "green", "width": 4},
                    marker={"size": 5, "color": "green"},
                    name=f"Path (dist: {solution.total_distance:.2f})",
                )
            )

        fig.update_layout(
            title="Close-Enough TSP in 3D Space",
            scene={
                "xaxis_title": "X",
                "yaxis_title": "Y",
                "zaxis_title": "Z",
                "aspectmode": "data",
            },
            showlegend=True,
        )

        return fig
