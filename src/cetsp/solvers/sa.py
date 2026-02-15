"""Simulated Annealing (SA) solver for CETSP.

Simulated Annealing is a probabilistic optimization algorithm inspired by the
annealing process in metallurgy. It allows occasional moves to worse solutions
to escape local optima, with the probability of accepting worse moves decreasing
as the "temperature" cools down.

Key Concepts:
- Temperature: Controls the probability of accepting worse solutions
- Cooling Schedule: How temperature decreases over time
- Neighborhood: How new solutions are generated from current solution
- Acceptance Probability: P(accept) = exp(-ΔE / T) for worse solutions

Reference:
    Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983).
    Optimization by Simulated Annealing. Science, 220(4598), 671-680.
"""

from __future__ import annotations

import math
import random
import time
from enum import Enum

import numpy as np

from ..problem import CETSP, CETSPPath, CETSPSolution
from .base import CETSPSolver


class CoolingSchedule(Enum):
    """Available cooling schedules for temperature reduction."""

    GEOMETRIC = "geometric"  # T = T * alpha
    LINEAR = "linear"  # T = T - alpha
    EXPONENTIAL = "exponential"  # T = T * exp(-alpha * t)
    LOGARITHMIC = "logarithmic"  # T = T0 / (1 + alpha * log(1 + t))
    ADAPTIVE = "adaptive"  # Adjusts based on acceptance rate


class SimulatedAnnealingCETSP(CETSPSolver):
    """
    Simulated Annealing solver for Close-Enough TSP.

    SA explores the solution space by probabilistically accepting worse
    solutions early in the search (high temperature), then becoming more
    greedy as the temperature cools.

    Parameters:
        initial_temp: Starting temperature (default: auto-calculated)
        final_temp: Ending temperature
        cooling_rate: Rate of temperature decrease (0 < alpha < 1 for geometric)
        cooling_schedule: Type of cooling schedule to use
        n_iterations: Maximum iterations per temperature level
        n_reheats: Number of reheating cycles (0 for standard SA)
        seed: Random seed for reproducibility

    Example:
        >>> solver = SimulatedAnnealingCETSP(problem, initial_temp=1000, cooling_rate=0.995)
        >>> solution = solver.solve()
    """

    def __init__(
        self,
        problem: CETSP,
        initial_temp: float | None = None,
        final_temp: float = 0.01,
        cooling_rate: float = 0.995,
        cooling_schedule: CoolingSchedule = CoolingSchedule.GEOMETRIC,
        n_iterations: int = 100,
        n_reheats: int = 0,
        seed: int | None = None,
    ):
        super().__init__(problem)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.cooling_schedule = cooling_schedule
        self.n_iterations = n_iterations
        self.n_reheats = n_reheats

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.n_customers = len(problem.customers)

        # Statistics
        self.best_tour: list[int] = []
        self.best_distance: float = float("inf")
        self.temperature_history: list[float] = []
        self.distance_history: list[float] = []
        self.acceptance_history: list[float] = []

    def _calculate_tour_distance(self, tour: list[int]) -> float:
        """Calculate actual CETSP tour distance with optimal waypoints."""
        if not tour:
            return float("inf")

        total_dist = 0.0
        current_pos = self.problem.depot.position.copy()

        for customer_idx in tour:
            customer = self.problem.customers[customer_idx]
            entry_point = customer.coverage_entry_point(current_pos)
            total_dist += np.linalg.norm(entry_point - current_pos)
            current_pos = entry_point

        if self.problem.return_to_depot:
            total_dist += np.linalg.norm(self.problem.depot.position - current_pos)

        return total_dist

    def _create_initial_solution(self) -> list[int]:
        """Create initial solution using nearest neighbor heuristic."""
        tour = []
        unvisited = set(range(self.n_customers))
        current_pos = self.problem.depot.position.copy()

        while unvisited:
            best_idx = None
            best_dist = float("inf")

            for idx in unvisited:
                customer = self.problem.customers[idx]
                entry = customer.coverage_entry_point(current_pos)
                dist = np.linalg.norm(entry - current_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            tour.append(best_idx)
            unvisited.remove(best_idx)
            current_pos = self.problem.customers[best_idx].coverage_entry_point(current_pos)

        return tour

    def _create_random_solution(self) -> list[int]:
        """Create a random tour."""
        tour = list(range(self.n_customers))
        random.shuffle(tour)
        return tour

    def _estimate_initial_temperature(self, tour: list[int], n_samples: int = 100) -> float:
        """
        Estimate initial temperature based on average cost differences.

        The idea is to set initial temperature such that ~80% of moves
        are accepted initially.
        """
        deltas = []
        current_cost = self._calculate_tour_distance(tour)

        for _ in range(n_samples):
            neighbor = self._get_neighbor(tour)
            neighbor_cost = self._calculate_tour_distance(neighbor)
            delta = neighbor_cost - current_cost
            if delta > 0:
                deltas.append(delta)

        if not deltas:
            return 1000.0  # Default if no worse moves found

        avg_delta = np.mean(deltas)
        # Set T such that exp(-avg_delta / T) = 0.8
        # => T = -avg_delta / ln(0.8)
        return -avg_delta / math.log(0.8)

    def _get_neighbor(self, tour: list[int]) -> list[int]:
        """
        Generate a neighbor solution using random move.

        Randomly selects one of several neighborhood operators:
        - 2-opt: Reverse a segment
        - Swap: Exchange two cities
        - Insert: Move a city to a different position
        - Or-opt: Move a segment to a different position
        """
        move_type = random.randint(0, 3)
        neighbor = tour.copy()
        n = len(neighbor)

        if n < 4:
            # For small tours, just swap
            if n >= 2:
                i, j = random.sample(range(n), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            return neighbor

        if move_type == 0:
            # 2-opt: Reverse segment
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            neighbor[i : j + 1] = reversed(neighbor[i : j + 1])

        elif move_type == 1:
            # Swap: Exchange two cities
            i, j = random.sample(range(n), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        elif move_type == 2:
            # Insert: Move city to new position
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                city = neighbor.pop(i)
                neighbor.insert(j, city)

        else:
            # Or-opt: Move segment of 1-3 cities
            seg_len = random.randint(1, min(3, n - 1))
            i = random.randint(0, n - seg_len)
            segment = neighbor[i : i + seg_len]
            del neighbor[i : i + seg_len]
            j = random.randint(0, len(neighbor))
            neighbor[j:j] = segment

        return neighbor

    def _acceptance_probability(self, delta: float, temperature: float) -> float:
        """Calculate probability of accepting a worse solution."""
        if delta <= 0:
            return 1.0
        return math.exp(-delta / temperature)

    def _cool_temperature(self, temperature: float, iteration: int, initial_temp: float) -> float:
        """Apply cooling schedule to reduce temperature."""
        if self.cooling_schedule == CoolingSchedule.GEOMETRIC:
            return temperature * self.cooling_rate

        elif self.cooling_schedule == CoolingSchedule.LINEAR:
            return max(self.final_temp, temperature - self.cooling_rate)

        elif self.cooling_schedule == CoolingSchedule.EXPONENTIAL:
            return initial_temp * math.exp(-self.cooling_rate * iteration)

        elif self.cooling_schedule == CoolingSchedule.LOGARITHMIC:
            return initial_temp / (1 + self.cooling_rate * math.log(1 + iteration))

        else:
            return temperature * self.cooling_rate

    def _apply_2opt(self, tour: list[int]) -> list[int]:
        """Apply 2-opt local search to improve tour."""
        improved = True
        best_tour = tour.copy()
        best_dist = self._calculate_tour_distance(best_tour)

        while improved:
            improved = False
            for i in range(len(best_tour) - 1):
                for j in range(i + 2, len(best_tour)):
                    new_tour = best_tour.copy()
                    new_tour[i : j + 1] = reversed(new_tour[i : j + 1])
                    new_dist = self._calculate_tour_distance(new_tour)

                    if new_dist < best_dist - 1e-9:
                        best_tour = new_tour
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break

        return best_tour

    def _build_solution(self, tour: list[int]) -> CETSPSolution:
        """Build CETSPSolution from tour."""
        path = CETSPPath()

        # Start at depot
        current_pos = self.problem.depot.position.copy()
        path.add_waypoint(current_pos, self.problem.depot)

        # Visit customers in tour order
        for customer_idx in tour:
            customer = self.problem.customers[customer_idx]
            entry_point = customer.coverage_entry_point(current_pos)
            path.add_waypoint(entry_point, customer)
            current_pos = entry_point

        # Return to depot if required
        if self.problem.return_to_depot:
            path.add_waypoint(self.problem.depot.position.copy())

        return CETSPSolution(path=path, is_complete=True)

    def solve(self) -> CETSPSolution:
        """
        Execute Simulated Annealing algorithm.

        Returns:
            CETSPSolution: Best solution found
        """
        time.time()

        # Initialize with greedy solution
        current_tour = self._create_initial_solution()
        current_cost = self._calculate_tour_distance(current_tour)

        self.best_tour = current_tour.copy()
        self.best_distance = current_cost

        # Set initial temperature
        if self.initial_temp is None:
            temperature = self._estimate_initial_temperature(current_tour)
        else:
            temperature = self.initial_temp

        initial_temp = temperature

        # Reset history
        self.temperature_history = []
        self.distance_history = []
        self.acceptance_history = []

        # Main SA loop with optional reheating
        for reheat in range(self.n_reheats + 1):
            if reheat > 0:
                # Reheat: restart from best solution with reduced temperature
                current_tour = self.best_tour.copy()
                current_cost = self.best_distance
                temperature = initial_temp * (0.5**reheat)

            iteration = 0

            while temperature > self.final_temp:
                accepted = 0

                for _ in range(self.n_iterations):
                    # Generate neighbor
                    neighbor_tour = self._get_neighbor(current_tour)
                    neighbor_cost = self._calculate_tour_distance(neighbor_tour)

                    # Calculate energy difference
                    delta = neighbor_cost - current_cost

                    # Accept or reject
                    if random.random() < self._acceptance_probability(delta, temperature):
                        current_tour = neighbor_tour
                        current_cost = neighbor_cost
                        accepted += 1

                        # Update best if improved
                        if current_cost < self.best_distance:
                            self.best_tour = current_tour.copy()
                            self.best_distance = current_cost

                # Record statistics
                self.temperature_history.append(temperature)
                self.distance_history.append(self.best_distance)
                self.acceptance_history.append(accepted / self.n_iterations)

                # Cool temperature
                temperature = self._cool_temperature(temperature, iteration, initial_temp)
                iteration += 1

        # Final 2-opt improvement
        self.best_tour = self._apply_2opt(self.best_tour)
        self.best_distance = self._calculate_tour_distance(self.best_tour)

        return self._build_solution(self.best_tour)


class AdaptiveSimulatedAnnealingCETSP(CETSPSolver):
    """
    Adaptive Simulated Annealing with dynamic parameter control.

    Enhancements over standard SA:
    - Adaptive cooling based on acceptance rate
    - Multiple neighborhood structures with adaptive selection
    - Reheating when stuck in local optima
    - Memory of good solutions for intensification

    Parameters:
        initial_temp: Starting temperature (auto-calculated if None)
        final_temp: Ending temperature
        target_acceptance_high: Target high acceptance rate (early search)
        target_acceptance_low: Target low acceptance rate (late search)
        n_iterations: Iterations per temperature level
        max_stagnation: Max iterations without improvement before reheat
        elite_size: Number of elite solutions to maintain
        seed: Random seed for reproducibility

    Example:
        >>> solver = AdaptiveSimulatedAnnealingCETSP(problem)
        >>> solution = solver.solve()
    """

    def __init__(
        self,
        problem: CETSP,
        initial_temp: float | None = None,
        final_temp: float = 0.001,
        target_acceptance_high: float = 0.5,
        target_acceptance_low: float = 0.02,
        n_iterations: int = 100,
        max_stagnation: int = 20,
        elite_size: int = 5,
        seed: int | None = None,
    ):
        super().__init__(problem)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.target_acceptance_high = target_acceptance_high
        self.target_acceptance_low = target_acceptance_low
        self.n_iterations = n_iterations
        self.max_stagnation = max_stagnation
        self.elite_size = elite_size

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.n_customers = len(problem.customers)

        # Neighborhood operators
        self.operators = [
            self._two_opt_move,
            self._swap_move,
            self._insert_move,
            self._or_opt_move,
            self._three_opt_move,
        ]
        self.operator_weights = [1.0] * len(self.operators)
        self.operator_success = [0] * len(self.operators)
        self.operator_calls = [0] * len(self.operators)

        # Elite solutions
        self.elite: list[tuple[list[int], float]] = []

        # Statistics
        self.best_tour: list[int] = []
        self.best_distance: float = float("inf")

    def _calculate_tour_distance(self, tour: list[int]) -> float:
        """Calculate actual CETSP tour distance with optimal waypoints."""
        if not tour:
            return float("inf")

        total_dist = 0.0
        current_pos = self.problem.depot.position.copy()

        for customer_idx in tour:
            customer = self.problem.customers[customer_idx]
            entry_point = customer.coverage_entry_point(current_pos)
            total_dist += np.linalg.norm(entry_point - current_pos)
            current_pos = entry_point

        if self.problem.return_to_depot:
            total_dist += np.linalg.norm(self.problem.depot.position - current_pos)

        return total_dist

    def _create_initial_solution(self) -> list[int]:
        """Create initial solution using nearest neighbor heuristic."""
        tour = []
        unvisited = set(range(self.n_customers))
        current_pos = self.problem.depot.position.copy()

        while unvisited:
            best_idx = None
            best_dist = float("inf")

            for idx in unvisited:
                customer = self.problem.customers[idx]
                entry = customer.coverage_entry_point(current_pos)
                dist = np.linalg.norm(entry - current_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            tour.append(best_idx)
            unvisited.remove(best_idx)
            current_pos = self.problem.customers[best_idx].coverage_entry_point(current_pos)

        return tour

    def _estimate_initial_temperature(self, tour: list[int], n_samples: int = 200) -> float:
        """Estimate initial temperature for ~80% acceptance rate."""
        deltas = []
        current_cost = self._calculate_tour_distance(tour)

        for _ in range(n_samples):
            operator = random.choice(self.operators)
            neighbor = operator(tour)
            neighbor_cost = self._calculate_tour_distance(neighbor)
            delta = neighbor_cost - current_cost
            if delta > 0:
                deltas.append(delta)

        if not deltas:
            return 1000.0

        avg_delta = np.mean(deltas)
        return -avg_delta / math.log(0.8)

    def _two_opt_move(self, tour: list[int]) -> list[int]:
        """2-opt: Reverse a segment."""
        neighbor = tour.copy()
        n = len(neighbor)
        if n < 4:
            return neighbor

        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        neighbor[i : j + 1] = reversed(neighbor[i : j + 1])
        return neighbor

    def _swap_move(self, tour: list[int]) -> list[int]:
        """Swap: Exchange two cities."""
        neighbor = tour.copy()
        n = len(neighbor)
        if n < 2:
            return neighbor

        i, j = random.sample(range(n), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    def _insert_move(self, tour: list[int]) -> list[int]:
        """Insert: Move a city to a different position."""
        neighbor = tour.copy()
        n = len(neighbor)
        if n < 2:
            return neighbor

        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j:
            city = neighbor.pop(i)
            neighbor.insert(j, city)
        return neighbor

    def _or_opt_move(self, tour: list[int]) -> list[int]:
        """Or-opt: Move a segment of 1-3 cities."""
        neighbor = tour.copy()
        n = len(neighbor)
        if n < 4:
            return neighbor

        seg_len = random.randint(1, min(3, n - 1))
        i = random.randint(0, n - seg_len)
        segment = neighbor[i : i + seg_len]
        del neighbor[i : i + seg_len]
        j = random.randint(0, len(neighbor))
        neighbor[j:j] = segment
        return neighbor

    def _three_opt_move(self, tour: list[int]) -> list[int]:
        """3-opt: Break tour into 3 segments and reconnect."""
        neighbor = tour.copy()
        n = len(neighbor)
        if n < 6:
            return self._two_opt_move(tour)

        # Select 3 cut points
        cuts = sorted(random.sample(range(n), 3))
        i, j, k = cuts

        # Different reconnection options
        option = random.randint(0, 3)

        if option == 0:
            # Reverse segment i:j
            neighbor[i:j] = reversed(neighbor[i:j])
        elif option == 1:
            # Reverse segment j:k
            neighbor[j:k] = reversed(neighbor[j:k])
        elif option == 2:
            # Reverse both segments
            neighbor[i:j] = reversed(neighbor[i:j])
            neighbor[j:k] = reversed(neighbor[j:k])
        else:
            # Swap segments
            seg1 = neighbor[i:j]
            seg2 = neighbor[j:k]
            neighbor = neighbor[:i] + seg2 + seg1 + neighbor[k:]

        return neighbor

    def _select_operator(self) -> int:
        """Select neighborhood operator based on adaptive weights."""
        total = sum(self.operator_weights)
        r = random.random() * total
        cumsum = 0
        for i, w in enumerate(self.operator_weights):
            cumsum += w
            if r <= cumsum:
                return i
        return len(self.operators) - 1

    def _update_operator_weights(self) -> None:
        """Update operator weights based on success rates."""
        for i in range(len(self.operators)):
            if self.operator_calls[i] > 0:
                success_rate = self.operator_success[i] / self.operator_calls[i]
                # Blend current weight with success rate
                self.operator_weights[i] = 0.9 * self.operator_weights[i] + 0.1 * (
                    success_rate + 0.1
                )

        # Normalize weights
        total = sum(self.operator_weights)
        self.operator_weights = [w / total * len(self.operators) for w in self.operator_weights]

    def _update_elite(self, tour: list[int], cost: float) -> None:
        """Update elite solution archive."""
        # Check if tour is different enough from existing elite
        for elite_tour, _ in self.elite:
            if tour == elite_tour:
                return

        self.elite.append((tour.copy(), cost))
        self.elite.sort(key=lambda x: x[1])

        if len(self.elite) > self.elite_size:
            self.elite = self.elite[: self.elite_size]

    def _get_elite_solution(self) -> tuple[list[int], float]:
        """Get a random elite solution."""
        if not self.elite:
            return None, float("inf")
        return random.choice(self.elite)

    def _acceptance_probability(self, delta: float, temperature: float) -> float:
        """Calculate probability of accepting a worse solution."""
        if delta <= 0:
            return 1.0
        return math.exp(-delta / temperature)

    def _adapt_temperature(
        self, temperature: float, acceptance_rate: float, progress: float
    ) -> float:
        """
        Adapt temperature based on acceptance rate and progress.

        Args:
            temperature: Current temperature
            acceptance_rate: Recent acceptance rate
            progress: Progress through search (0 to 1)

        Returns:
            Adjusted temperature
        """
        # Target acceptance rate decreases over time
        target = (
            self.target_acceptance_high * (1 - progress) + self.target_acceptance_low * progress
        )

        # Adjust temperature to approach target acceptance rate
        if acceptance_rate > target * 1.2:
            # Too many acceptances - cool faster
            return temperature * 0.95
        elif acceptance_rate < target * 0.8:
            # Too few acceptances - cool slower
            return temperature * 0.99
        else:
            # On target - standard cooling
            return temperature * 0.97

    def _apply_2opt(self, tour: list[int]) -> list[int]:
        """Apply 2-opt local search to improve tour."""
        improved = True
        best_tour = tour.copy()
        best_dist = self._calculate_tour_distance(best_tour)

        while improved:
            improved = False
            for i in range(len(best_tour) - 1):
                for j in range(i + 2, len(best_tour)):
                    new_tour = best_tour.copy()
                    new_tour[i : j + 1] = reversed(new_tour[i : j + 1])
                    new_dist = self._calculate_tour_distance(new_tour)

                    if new_dist < best_dist - 1e-9:
                        best_tour = new_tour
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break

        return best_tour

    def _build_solution(self, tour: list[int]) -> CETSPSolution:
        """Build CETSPSolution from tour."""
        path = CETSPPath()

        # Start at depot
        current_pos = self.problem.depot.position.copy()
        path.add_waypoint(current_pos, self.problem.depot)

        # Visit customers in tour order
        for customer_idx in tour:
            customer = self.problem.customers[customer_idx]
            entry_point = customer.coverage_entry_point(current_pos)
            path.add_waypoint(entry_point, customer)
            current_pos = entry_point

        # Return to depot if required
        if self.problem.return_to_depot:
            path.add_waypoint(self.problem.depot.position.copy())

        return CETSPSolution(path=path, is_complete=True)

    def solve(self) -> CETSPSolution:
        """
        Execute Adaptive Simulated Annealing algorithm.

        Returns:
            CETSPSolution: Best solution found
        """
        time.time()

        # Initialize with greedy solution
        current_tour = self._create_initial_solution()
        current_cost = self._calculate_tour_distance(current_tour)

        self.best_tour = current_tour.copy()
        self.best_distance = current_cost
        self._update_elite(current_tour, current_cost)

        # Set initial temperature
        if self.initial_temp is None:
            temperature = self._estimate_initial_temperature(current_tour)
        else:
            temperature = self.initial_temp

        initial_temp = temperature

        # Reset operator statistics
        self.operator_weights = [1.0] * len(self.operators)
        self.operator_success = [0] * len(self.operators)
        self.operator_calls = [0] * len(self.operators)

        stagnation_count = 0
        iteration = 0
        max_iterations = 1000  # Safety limit

        while temperature > self.final_temp and iteration < max_iterations:
            accepted = 0
            improved_in_temp = False

            for _ in range(self.n_iterations):
                # Select and apply operator
                op_idx = self._select_operator()
                neighbor_tour = self.operators[op_idx](current_tour)
                neighbor_cost = self._calculate_tour_distance(neighbor_tour)
                self.operator_calls[op_idx] += 1

                # Calculate energy difference
                delta = neighbor_cost - current_cost

                # Accept or reject
                if random.random() < self._acceptance_probability(delta, temperature):
                    current_tour = neighbor_tour
                    current_cost = neighbor_cost
                    accepted += 1

                    if delta < 0:
                        self.operator_success[op_idx] += 1

                    # Update best and elite
                    if current_cost < self.best_distance:
                        self.best_tour = current_tour.copy()
                        self.best_distance = current_cost
                        improved_in_temp = True
                        self._update_elite(current_tour, current_cost)

            # Track stagnation
            if improved_in_temp:
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Reheat if stagnated
            if stagnation_count >= self.max_stagnation:
                # Try to escape by jumping to elite solution
                elite_tour, elite_cost = self._get_elite_solution()
                if elite_tour is not None:
                    current_tour = elite_tour.copy()
                    current_cost = elite_cost

                # Reheat
                temperature = min(temperature * 5, initial_temp * 0.5)
                stagnation_count = 0

            # Adaptive temperature update
            acceptance_rate = accepted / self.n_iterations
            progress = math.log(initial_temp / temperature) / math.log(
                initial_temp / self.final_temp
            )
            progress = min(1.0, max(0.0, progress))

            temperature = self._adapt_temperature(temperature, acceptance_rate, progress)

            # Update operator weights periodically
            if iteration % 10 == 0:
                self._update_operator_weights()

            iteration += 1

        # Final intensification with 2-opt
        self.best_tour = self._apply_2opt(self.best_tour)
        self.best_distance = self._calculate_tour_distance(self.best_tour)

        return self._build_solution(self.best_tour)


class ThresholdAcceptingCETSP(CETSPSolver):
    """
    Threshold Accepting algorithm - a deterministic variant of SA.

    Instead of probabilistic acceptance, uses a fixed threshold:
    - Accept if delta < threshold
    - Threshold decreases over time

    This can be faster than SA while achieving similar results.

    Parameters:
        initial_threshold: Starting threshold value
        final_threshold: Ending threshold value
        threshold_reduction: Rate of threshold decrease
        n_iterations: Iterations per threshold level
        seed: Random seed for reproducibility

    Reference:
        Dueck, G., & Scheuer, T. (1990).
        Threshold Accepting: A General Purpose Optimization Algorithm.
        Journal of Computational Physics, 90(1), 161-175.
    """

    def __init__(
        self,
        problem: CETSP,
        initial_threshold: float | None = None,
        final_threshold: float = 0.01,
        threshold_reduction: float = 0.995,
        n_iterations: int = 100,
        seed: int | None = None,
    ):
        super().__init__(problem)
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.threshold_reduction = threshold_reduction
        self.n_iterations = n_iterations

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.n_customers = len(problem.customers)

        self.best_tour: list[int] = []
        self.best_distance: float = float("inf")

    def _calculate_tour_distance(self, tour: list[int]) -> float:
        """Calculate actual CETSP tour distance with optimal waypoints."""
        if not tour:
            return float("inf")

        total_dist = 0.0
        current_pos = self.problem.depot.position.copy()

        for customer_idx in tour:
            customer = self.problem.customers[customer_idx]
            entry_point = customer.coverage_entry_point(current_pos)
            total_dist += np.linalg.norm(entry_point - current_pos)
            current_pos = entry_point

        if self.problem.return_to_depot:
            total_dist += np.linalg.norm(self.problem.depot.position - current_pos)

        return total_dist

    def _create_initial_solution(self) -> list[int]:
        """Create initial solution using nearest neighbor heuristic."""
        tour = []
        unvisited = set(range(self.n_customers))
        current_pos = self.problem.depot.position.copy()

        while unvisited:
            best_idx = None
            best_dist = float("inf")

            for idx in unvisited:
                customer = self.problem.customers[idx]
                entry = customer.coverage_entry_point(current_pos)
                dist = np.linalg.norm(entry - current_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            tour.append(best_idx)
            unvisited.remove(best_idx)
            current_pos = self.problem.customers[best_idx].coverage_entry_point(current_pos)

        return tour

    def _estimate_initial_threshold(self, tour: list[int], n_samples: int = 100) -> float:
        """Estimate initial threshold based on average cost differences."""
        deltas = []
        current_cost = self._calculate_tour_distance(tour)

        for _ in range(n_samples):
            neighbor = self._get_neighbor(tour)
            neighbor_cost = self._calculate_tour_distance(neighbor)
            delta = neighbor_cost - current_cost
            if delta > 0:
                deltas.append(delta)

        if not deltas:
            return 50.0

        # Set threshold to cover most uphill moves initially
        return np.percentile(deltas, 90)

    def _get_neighbor(self, tour: list[int]) -> list[int]:
        """Generate a neighbor solution."""
        move_type = random.randint(0, 3)
        neighbor = tour.copy()
        n = len(neighbor)

        if n < 4:
            if n >= 2:
                i, j = random.sample(range(n), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            return neighbor

        if move_type == 0:
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            neighbor[i : j + 1] = reversed(neighbor[i : j + 1])
        elif move_type == 1:
            i, j = random.sample(range(n), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        elif move_type == 2:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                city = neighbor.pop(i)
                neighbor.insert(j, city)
        else:
            seg_len = random.randint(1, min(3, n - 1))
            i = random.randint(0, n - seg_len)
            segment = neighbor[i : i + seg_len]
            del neighbor[i : i + seg_len]
            j = random.randint(0, len(neighbor))
            neighbor[j:j] = segment

        return neighbor

    def _apply_2opt(self, tour: list[int]) -> list[int]:
        """Apply 2-opt local search to improve tour."""
        improved = True
        best_tour = tour.copy()
        best_dist = self._calculate_tour_distance(best_tour)

        while improved:
            improved = False
            for i in range(len(best_tour) - 1):
                for j in range(i + 2, len(best_tour)):
                    new_tour = best_tour.copy()
                    new_tour[i : j + 1] = reversed(new_tour[i : j + 1])
                    new_dist = self._calculate_tour_distance(new_tour)

                    if new_dist < best_dist - 1e-9:
                        best_tour = new_tour
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break

        return best_tour

    def _build_solution(self, tour: list[int]) -> CETSPSolution:
        """Build CETSPSolution from tour."""
        path = CETSPPath()

        # Start at depot
        current_pos = self.problem.depot.position.copy()
        path.add_waypoint(current_pos, self.problem.depot)

        # Visit customers in tour order
        for customer_idx in tour:
            customer = self.problem.customers[customer_idx]
            entry_point = customer.coverage_entry_point(current_pos)
            path.add_waypoint(entry_point, customer)
            current_pos = entry_point

        # Return to depot if required
        if self.problem.return_to_depot:
            path.add_waypoint(self.problem.depot.position.copy())

        return CETSPSolution(path=path, is_complete=True)

    def solve(self) -> CETSPSolution:
        """
        Execute Threshold Accepting algorithm.

        Returns:
            CETSPSolution: Best solution found
        """
        # Initialize
        current_tour = self._create_initial_solution()
        current_cost = self._calculate_tour_distance(current_tour)

        self.best_tour = current_tour.copy()
        self.best_distance = current_cost

        # Set initial threshold
        if self.initial_threshold is None:
            threshold = self._estimate_initial_threshold(current_tour)
        else:
            threshold = self.initial_threshold

        # Main loop
        while threshold > self.final_threshold:
            for _ in range(self.n_iterations):
                neighbor_tour = self._get_neighbor(current_tour)
                neighbor_cost = self._calculate_tour_distance(neighbor_tour)

                delta = neighbor_cost - current_cost

                # Accept if improvement or within threshold
                if delta < threshold:
                    current_tour = neighbor_tour
                    current_cost = neighbor_cost

                    if current_cost < self.best_distance:
                        self.best_tour = current_tour.copy()
                        self.best_distance = current_cost

            # Reduce threshold
            threshold *= self.threshold_reduction

        # Final 2-opt improvement
        self.best_tour = self._apply_2opt(self.best_tour)
        self.best_distance = self._calculate_tour_distance(self.best_tour)

        return self._build_solution(self.best_tour)
