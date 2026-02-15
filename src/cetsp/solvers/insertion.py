"""Insertion-based solver for CETSP."""

from __future__ import annotations

import time

import numpy as np

from ..node import CETSPNode
from ..problem import CETSPPath, CETSPSolution
from .base import CETSPSolver


class InsertionCETSP(CETSPSolver):
    """
    Insertion-based solver for CETSP.

    Starts with depot and iteratively inserts customers at
    positions that minimize total path length increase.
    """

    def solve(self) -> CETSPSolution:
        """Solve using cheapest insertion heuristic."""
        start_time = time.time()

        # Start with just depot
        tour_nodes = [self.problem.depot]
        if self.problem.return_to_depot:
            tour_nodes.append(self.problem.depot)

        unvisited = set(self.problem.customers)

        while unvisited:
            best_customer = None
            best_position = None
            best_cost_increase = float("inf")

            for customer in unvisited:
                # Try inserting at each position
                for pos in range(1, len(tour_nodes)):
                    cost_increase = self._insertion_cost(tour_nodes, pos, customer)

                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_customer = customer
                        best_position = pos

            if best_customer is None:
                break

            # Insert customer
            tour_nodes.insert(best_position, best_customer)
            unvisited.remove(best_customer)

        # Convert tour to path with optimal waypoints
        path = self._tour_to_path(tour_nodes)

        computation_time = time.time() - start_time

        return CETSPSolution(
            path=path, is_complete=len(unvisited) == 0, computation_time=computation_time
        )

    def _insertion_cost(self, tour: list[CETSPNode], position: int, customer: CETSPNode) -> float:
        """Calculate cost of inserting customer at position."""
        prev_node = tour[position - 1]
        next_node = tour[position]

        # Current distance
        current_dist = self.problem.get_effective_distance(prev_node, next_node)

        # New distance via customer
        new_dist = self.problem.get_effective_distance(
            prev_node, customer
        ) + self.problem.get_effective_distance(customer, next_node)

        return new_dist - current_dist

    def _tour_to_path(self, tour: list[CETSPNode]) -> CETSPPath:
        """Convert node tour to path with optimal waypoints."""
        path = CETSPPath()

        if not tour:
            return path

        current_pos = tour[0].position.copy()
        path.add_waypoint(current_pos)

        for i in range(1, len(tour)):
            node = tour[i]

            # Find optimal entry point
            if node.radius > 0:
                entry = node.coverage_entry_point(current_pos)
            else:
                entry = node.position.copy()

            # Only add if significantly different from current
            if np.linalg.norm(entry - current_pos) > 1e-6:
                path.add_waypoint(entry, node if node != self.problem.depot else None)
                current_pos = entry

        return path
