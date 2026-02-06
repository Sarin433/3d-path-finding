"""A* search solver for CETSP."""

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from ..problem import CETSP, CETSPPath, CETSPSolution
from .base import CETSPSolver


@dataclass(order=True)
class CETSPState:
    """State for A* search in CETSP."""

    f_score: float
    g_score: float = field(compare=False)
    position: np.ndarray = field(compare=False)
    visited: frozenset[int] = field(compare=False, default_factory=frozenset)
    path_positions: tuple = field(compare=False, default_factory=tuple)
    path_covered: tuple = field(compare=False, default_factory=tuple)

    def __hash__(self) -> int:
        # Discretize position for hashing
        pos_key = tuple(np.round(self.position, 2))
        return hash((pos_key, self.visited))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CETSPState):
            return False
        return (
            np.allclose(self.position, other.position, atol=0.01) and self.visited == other.visited
        )


class AStarCETSP(CETSPSolver):
    """
    A* search solver for CETSP.

    Uses A* with an admissible heuristic based on minimum
    spanning tree of uncovered customers.
    """

    def __init__(
        self,
        problem: CETSP,
        max_iterations: int = 100000,
        position_tolerance: float = 0.5,
        fallback_to_greedy: bool = True,
    ):
        super().__init__(problem)
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.fallback_to_greedy = fallback_to_greedy

    def _greedy_complete(
        self,
        current_pos: np.ndarray,
        visited: frozenset[int],
        path_positions: list,
        path_covered: list,
    ) -> tuple:
        """Complete the tour greedily from current state."""
        customer_ids = {c.id for c in self.problem.customers}
        unvisited = customer_ids - visited

        pos = current_pos.copy()
        total_cost = 0.0

        while unvisited:
            # Find nearest unvisited customer
            best_customer = None
            best_dist = float("inf")
            best_entry = None

            for customer in self.problem.customers:
                if customer.id not in unvisited:
                    continue
                entry_point = customer.coverage_entry_point(pos)
                dist = np.linalg.norm(entry_point - pos)
                if dist < best_dist:
                    best_dist = dist
                    best_customer = customer
                    best_entry = entry_point

            if best_customer is None:
                break

            # Move to this customer
            path_positions.append(tuple(best_entry))
            path_covered.append(best_customer.id)
            unvisited.remove(best_customer.id)
            total_cost += best_dist
            pos = best_entry

        return path_positions, path_covered, total_cost, pos

    def solve(self) -> CETSPSolution:
        """Solve using A* search."""
        start_time = time.time()

        # Initial state at depot
        initial_pos = self.problem.depot.position.copy()
        h = self._heuristic(initial_pos, frozenset())

        initial_state = CETSPState(
            f_score=h,
            g_score=0.0,
            position=initial_pos,
            visited=frozenset(),
            path_positions=(tuple(initial_pos),),
            path_covered=(),
        )

        open_set = [initial_state]
        closed_positions: dict[tuple, float] = {}

        customer_ids = frozenset(c.id for c in self.problem.customers)
        iterations = 0

        while open_set and iterations < self.max_iterations:
            iterations += 1

            current = heapq.heappop(open_set)

            # Check if goal reached
            if current.visited == customer_ids:
                # Add return to depot if needed
                path = self._reconstruct_path(current)

                if self.problem.return_to_depot:
                    depot_pos = self.problem.depot.position
                    if not np.allclose(path.waypoints[-1], depot_pos):
                        path.add_waypoint(depot_pos.copy())

                computation_time = time.time() - start_time

                return CETSPSolution(path=path, is_complete=True, computation_time=computation_time)

            # Skip if we've seen better
            state_key = (tuple(np.round(current.position, 1)), current.visited)
            if state_key in closed_positions and closed_positions[state_key] <= current.g_score:
                continue
            closed_positions[state_key] = current.g_score

            # Expand to unvisited customers
            for customer in self.problem.customers:
                if customer.id in current.visited:
                    continue

                # Calculate entry point and cost
                entry_point = customer.coverage_entry_point(current.position)
                travel_cost = np.linalg.norm(entry_point - current.position)

                new_g = current.g_score + travel_cost
                new_visited = current.visited | {customer.id}
                new_h = self._heuristic(entry_point, new_visited)

                new_state = CETSPState(
                    f_score=new_g + new_h,
                    g_score=new_g,
                    position=entry_point,
                    visited=new_visited,
                    path_positions=current.path_positions + (tuple(entry_point),),
                    path_covered=current.path_covered + (customer.id,),
                )

                heapq.heappush(open_set, new_state)

        # A* did not complete - use fallback strategy
        if self.fallback_to_greedy and open_set:
            # Find best partial state (most customers visited)
            best = max(open_set, key=lambda s: (len(s.visited), -s.g_score))

            # Complete greedily from best state
            path_positions = list(best.path_positions)
            path_covered = list(best.path_covered)

            path_positions, path_covered, extra_cost, final_pos = self._greedy_complete(
                np.array(best.position), best.visited, path_positions, path_covered
            )

            # Build path
            path = CETSPPath()
            id_to_customer = {c.id: c for c in self.problem.customers}

            for i, pos_tuple in enumerate(path_positions):
                pos = np.array(pos_tuple)
                covered = None
                if i > 0 and i - 1 < len(path_covered):
                    covered = id_to_customer.get(path_covered[i - 1])
                path.add_waypoint(pos, covered)

            if self.problem.return_to_depot:
                path.add_waypoint(self.problem.depot.position.copy())

            computation_time = time.time() - start_time

            return CETSPSolution(
                path=path,
                is_complete=True,  # Now complete with greedy fallback
                computation_time=computation_time,
            )

        # Return best partial solution (no fallback)
        if open_set:
            best = min(open_set, key=lambda s: -len(s.visited))
            path = self._reconstruct_path(best)

            if self.problem.return_to_depot:
                path.add_waypoint(self.problem.depot.position.copy())

            computation_time = time.time() - start_time

            return CETSPSolution(path=path, is_complete=False, computation_time=computation_time)

        raise RuntimeError("No solution found")

    def _heuristic(self, position: np.ndarray, visited: frozenset[int]) -> float:
        """
        Calculate admissible heuristic (fast version).

        Uses minimum distance to nearest unvisited customer + return to depot.
        This is faster than MST-based heuristic while still admissible.
        """
        unvisited = [c for c in self.problem.customers if c.id not in visited]

        if not unvisited:
            if self.problem.return_to_depot:
                return np.linalg.norm(position - self.problem.depot.position)
            return 0.0

        # Distance from current position to nearest unvisited customer
        min_dist = float("inf")
        for c in unvisited:
            # Quick distance to coverage boundary
            dist_to_center = np.linalg.norm(c.position - position)
            dist = max(0, dist_to_center - c.radius)
            min_dist = min(min_dist, dist)

        # Add return to depot (use minimum distance from any unvisited)
        h = min_dist
        if self.problem.return_to_depot:
            min_return = float("inf")
            depot_pos = self.problem.depot.position
            for c in unvisited:
                dist = max(0, np.linalg.norm(c.position - depot_pos) - c.radius)
                min_return = min(min_return, dist)
            h += min_return

        return h

    def _reconstruct_path(self, state: CETSPState) -> CETSPPath:
        """Reconstruct path from state."""
        path = CETSPPath()

        id_to_customer = {c.id: c for c in self.problem.customers}

        for i, pos_tuple in enumerate(state.path_positions):
            pos = np.array(pos_tuple)
            covered = None
            if i > 0 and i - 1 < len(state.path_covered):
                covered = id_to_customer.get(state.path_covered[i - 1])
            path.add_waypoint(pos, covered)

        return path
