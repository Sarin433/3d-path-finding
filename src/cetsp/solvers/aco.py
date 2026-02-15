"""Ant Colony Optimization (ACO) solver for CETSP."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np

from ..problem import CETSP, CETSPPath, CETSPSolution
from .base import CETSPSolver


@dataclass
class Ant:
    """Represents an ant in the colony."""

    tour: list[int] = field(default_factory=list)
    distance: float = float("inf")
    visited: set = field(default_factory=set)

    def reset(self) -> None:
        """Reset ant for a new iteration."""
        self.tour = []
        self.distance = float("inf")
        self.visited = set()


class AntColonyCETSP(CETSPSolver):
    """
    Ant Colony Optimization solver for Close-Enough TSP.

    ACO is a metaheuristic inspired by the foraging behavior of ants.
    Ants deposit pheromones on paths, and good paths accumulate more
    pheromones, guiding future ants toward better solutions.

    Parameters:
        n_ants: Number of ants in the colony
        n_iterations: Maximum number of iterations
        alpha: Pheromone importance factor (α)
        beta: Heuristic importance factor (β)
        rho: Pheromone evaporation rate (ρ)
        q: Pheromone deposit factor
        initial_pheromone: Initial pheromone level on all edges
        min_pheromone: Minimum pheromone level (for MMAS variant)
        max_pheromone: Maximum pheromone level (for MMAS variant)
        use_elitist: Use elitist ant system (best ant deposits extra pheromone)
        use_local_search: Apply 2-opt local search to best solution
        seed: Random seed for reproducibility

    Example:
        >>> solver = AntColonyCETSP(problem, n_ants=20, n_iterations=100)
        >>> solution = solver.solve()
    """

    def __init__(
        self,
        problem: CETSP,
        n_ants: int = 20,
        n_iterations: int = 100,
        alpha: float = 1.0,
        beta: float = 3.0,
        rho: float = 0.1,
        q: float = 100.0,
        initial_pheromone: float = 1.0,
        min_pheromone: float | None = None,
        max_pheromone: float | None = None,
        use_elitist: bool = True,
        elitist_weight: float = 2.0,
        use_local_search: bool = True,
        seed: int | None = None,
        early_stop_iterations: int = 30,
    ):
        super().__init__(problem)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.initial_pheromone = initial_pheromone
        self.use_elitist = use_elitist
        self.elitist_weight = elitist_weight
        self.use_local_search = use_local_search
        self.early_stop_iterations = early_stop_iterations

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.n_customers = len(problem.customers)
        self.n_nodes = self.n_customers + 1  # +1 for depot (index 0)

        # Initialize pheromone matrix
        self.pheromone = np.full((self.n_nodes, self.n_nodes), initial_pheromone)

        # Pheromone bounds for MMAS
        self.min_pheromone = min_pheromone if min_pheromone else initial_pheromone * 0.01
        self.max_pheromone = max_pheromone if max_pheromone else initial_pheromone * 10.0

        # Precompute heuristic information (inverse of distance)
        self._precompute_heuristic()

        # Create ants
        self.ants = [Ant() for _ in range(n_ants)]

    def _precompute_heuristic(self) -> None:
        """Precompute heuristic information (η = 1/distance)."""
        self.distances = np.zeros((self.n_nodes, self.n_nodes))
        self.heuristic = np.zeros((self.n_nodes, self.n_nodes))

        all_nodes = [self.problem.depot] + list(self.problem.customers)

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    # Use effective distance (accounting for coverage radius)
                    dist = self.problem.get_effective_distance(all_nodes[i], all_nodes[j])
                    self.distances[i, j] = dist if dist > 0 else 0.001
                    self.heuristic[i, j] = 1.0 / self.distances[i, j]

    def _calculate_tour_distance(self, tour: list[int]) -> float:
        """
        Calculate actual CETSP tour distance with optimal waypoints.

        Args:
            tour: List of customer indices (0-based, not including depot)

        Returns:
            Total tour distance considering coverage radii
        """
        if not tour:
            return 0.0

        total_dist = 0.0
        current_pos = self.problem.depot.position.copy()

        for customer_idx in tour:
            customer = self.problem.customers[customer_idx]
            entry_point = customer.coverage_entry_point(current_pos)
            total_dist += np.linalg.norm(entry_point - current_pos)
            current_pos = entry_point

        # Return to depot
        if self.problem.return_to_depot:
            total_dist += np.linalg.norm(self.problem.depot.position - current_pos)

        return total_dist

    def _select_next_node(self, ant: Ant, current: int) -> int:
        """
        Select next node for ant using probabilistic rule.

        Uses the standard ACO transition probability:
        P(i,j) = [τ(i,j)]^α * [η(i,j)]^β / Σ[τ(i,k)]^α * [η(i,k)]^β

        Args:
            ant: Current ant
            current: Current node index (0=depot, 1..n=customers)

        Returns:
            Index of next node to visit
        """
        # Get unvisited customer indices (1 to n_customers)
        unvisited = [i for i in range(1, self.n_nodes) if i not in ant.visited]

        if not unvisited:
            return 0  # Return to depot

        # Calculate probabilities
        probabilities = np.zeros(len(unvisited))

        for idx, j in enumerate(unvisited):
            tau = self.pheromone[current, j] ** self.alpha
            eta = self.heuristic[current, j] ** self.beta
            probabilities[idx] = tau * eta

        # Normalize
        prob_sum = probabilities.sum()
        if prob_sum > 0:
            probabilities /= prob_sum
        else:
            probabilities = np.ones(len(unvisited)) / len(unvisited)

        # Roulette wheel selection
        r = random.random()
        cumsum = 0.0
        for idx, prob in enumerate(probabilities):
            cumsum += prob
            if r <= cumsum:
                return unvisited[idx]

        return unvisited[-1]

    def _construct_solution(self, ant: Ant) -> None:
        """Construct a complete tour for an ant."""
        ant.reset()

        # Start at depot (node 0)
        current = 0
        ant.visited.add(0)

        # Visit all customers
        while len(ant.visited) < self.n_nodes:
            next_node = self._select_next_node(ant, current)

            if next_node == 0:
                break  # No more customers to visit

            ant.tour.append(next_node - 1)  # Convert to 0-based customer index
            ant.visited.add(next_node)
            current = next_node

        # Calculate tour distance
        ant.distance = self._calculate_tour_distance(ant.tour)

    def _update_pheromone(self, best_ant: Ant) -> None:
        """
        Update pheromone matrix.

        Applies evaporation and deposits pheromone based on ant tours.
        """
        # Evaporation
        self.pheromone *= 1 - self.rho

        # Deposit pheromone for all ants
        for ant in self.ants:
            if ant.distance < float("inf") and ant.tour:
                deposit = self.q / ant.distance

                # Depot to first customer
                first_node = ant.tour[0] + 1
                self.pheromone[0, first_node] += deposit
                self.pheromone[first_node, 0] += deposit

                # Between customers
                for i in range(len(ant.tour) - 1):
                    from_node = ant.tour[i] + 1
                    to_node = ant.tour[i + 1] + 1
                    self.pheromone[from_node, to_node] += deposit
                    self.pheromone[to_node, from_node] += deposit

                # Last customer to depot
                last_node = ant.tour[-1] + 1
                self.pheromone[last_node, 0] += deposit
                self.pheromone[0, last_node] += deposit

        # Elitist ant system: best ant deposits extra pheromone
        if self.use_elitist and best_ant.tour:
            elite_deposit = self.elitist_weight * self.q / best_ant.distance

            first_node = best_ant.tour[0] + 1
            self.pheromone[0, first_node] += elite_deposit
            self.pheromone[first_node, 0] += elite_deposit

            for i in range(len(best_ant.tour) - 1):
                from_node = best_ant.tour[i] + 1
                to_node = best_ant.tour[i + 1] + 1
                self.pheromone[from_node, to_node] += elite_deposit
                self.pheromone[to_node, from_node] += elite_deposit

            last_node = best_ant.tour[-1] + 1
            self.pheromone[last_node, 0] += elite_deposit
            self.pheromone[0, last_node] += elite_deposit

        # Apply pheromone bounds (MMAS)
        self.pheromone = np.clip(self.pheromone, self.min_pheromone, self.max_pheromone)

    def _local_search_2opt(self, tour: list[int]) -> tuple[list[int], float]:
        """Apply 2-opt local search to improve tour."""
        improved = True
        best_tour = tour.copy()
        best_dist = self._calculate_tour_distance(best_tour)

        while improved:
            improved = False

            for i in range(len(best_tour) - 1):
                for j in range(i + 2, len(best_tour)):
                    # Try reversing segment [i+1, j]
                    new_tour = (
                        best_tour[: i + 1] + best_tour[i + 1 : j + 1][::-1] + best_tour[j + 1 :]
                    )
                    new_dist = self._calculate_tour_distance(new_tour)

                    if new_dist < best_dist - 1e-6:
                        best_tour = new_tour
                        best_dist = new_dist
                        improved = True
                        break

                if improved:
                    break

        return best_tour, best_dist

    def _tour_to_solution(self, tour: list[int], computation_time: float) -> CETSPSolution:
        """Convert tour to CETSPSolution."""
        path = CETSPPath()
        current_pos = self.problem.depot.position.copy()
        path.add_waypoint(current_pos)

        for customer_idx in tour:
            customer = self.problem.customers[customer_idx]
            entry_point = customer.coverage_entry_point(current_pos)
            path.add_waypoint(entry_point, customer)
            current_pos = entry_point

        if self.problem.return_to_depot:
            path.add_waypoint(self.problem.depot.position.copy())

        return CETSPSolution(path=path, is_complete=True, computation_time=computation_time)

    def solve(self) -> CETSPSolution:
        """
        Solve CETSP using Ant Colony Optimization.

        Returns:
            CETSPSolution with the best tour found
        """
        start_time = time.time()

        best_tour = []
        best_distance = float("inf")
        iterations_without_improvement = 0

        for _iteration in range(self.n_iterations):
            # Construct solutions for all ants
            for ant in self.ants:
                self._construct_solution(ant)

            # Find iteration best
            iteration_best = min(self.ants, key=lambda a: a.distance)

            # Update global best
            if iteration_best.distance < best_distance:
                best_tour = iteration_best.tour.copy()
                best_distance = iteration_best.distance
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            # Early stopping
            if iterations_without_improvement >= self.early_stop_iterations:
                break

            # Update pheromone
            global_best_ant = Ant(tour=best_tour, distance=best_distance)
            self._update_pheromone(global_best_ant)

        # Apply local search to best solution
        if self.use_local_search and best_tour:
            best_tour, best_distance = self._local_search_2opt(best_tour)

        computation_time = time.time() - start_time

        return self._tour_to_solution(best_tour, computation_time)


class MaxMinAntSystem(AntColonyCETSP):
    """
    MAX-MIN Ant System (MMAS) variant of ACO.

    Key differences from standard ACO:
    - Only the best ant (iteration or global best) deposits pheromone
    - Pheromone values are bounded between min and max limits
    - Pheromone trails are initialized to maximum value

    This variant often produces better results than standard ACO.
    """

    def __init__(
        self,
        problem: CETSP,
        n_ants: int = 20,
        n_iterations: int = 100,
        alpha: float = 1.0,
        beta: float = 3.0,
        rho: float = 0.02,
        use_global_best: bool = True,
        pbest: float = 0.05,
        seed: int | None = None,
    ):
        # Calculate initial bounds based on problem size
        len(problem.customers)

        super().__init__(
            problem=problem,
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            q=1.0,
            initial_pheromone=1.0,
            use_elitist=False,
            use_local_search=True,
            seed=seed,
        )

        self.use_global_best = use_global_best
        self.pbest = pbest

        # Initialize pheromone to max value
        self._update_pheromone_bounds(1000.0)  # Initial estimate
        self.pheromone.fill(self.max_pheromone)

    def _update_pheromone_bounds(self, best_distance: float) -> None:
        """Update pheromone bounds based on best solution."""
        if best_distance <= 0:
            return

        # τ_max = 1 / (ρ * L_best)
        self.max_pheromone = 1.0 / (self.rho * best_distance)

        # τ_min = τ_max * (1 - p_best^(1/n)) / ((n/2 - 1) * p_best^(1/n))
        n = self.n_customers
        p_dec = self.pbest ** (1.0 / n)
        avg = n / 2.0

        if avg > 1 and p_dec > 0:
            self.min_pheromone = (self.max_pheromone * (1 - p_dec)) / ((avg - 1) * p_dec)
        else:
            self.min_pheromone = self.max_pheromone * 0.01

        self.min_pheromone = max(self.min_pheromone, 1e-6)

    def _update_pheromone(self, best_ant: Ant) -> None:
        """Update pheromone using MMAS rules."""
        # Evaporation
        self.pheromone *= 1 - self.rho

        # Only best ant deposits pheromone
        if best_ant.tour and best_ant.distance < float("inf"):
            deposit = 1.0 / best_ant.distance

            # Depot to first customer
            first_node = best_ant.tour[0] + 1
            self.pheromone[0, first_node] += deposit
            self.pheromone[first_node, 0] += deposit

            # Between customers
            for i in range(len(best_ant.tour) - 1):
                from_node = best_ant.tour[i] + 1
                to_node = best_ant.tour[i + 1] + 1
                self.pheromone[from_node, to_node] += deposit
                self.pheromone[to_node, from_node] += deposit

            # Last customer to depot
            last_node = best_ant.tour[-1] + 1
            self.pheromone[last_node, 0] += deposit
            self.pheromone[0, last_node] += deposit

            # Update bounds
            self._update_pheromone_bounds(best_ant.distance)

        # Apply pheromone bounds
        self.pheromone = np.clip(self.pheromone, self.min_pheromone, self.max_pheromone)

    def solve(self) -> CETSPSolution:
        """Solve using MAX-MIN Ant System."""
        start_time = time.time()

        best_tour = []
        best_distance = float("inf")
        iterations_without_improvement = 0

        for _iteration in range(self.n_iterations):
            # Construct solutions
            for ant in self.ants:
                self._construct_solution(ant)

            # Find iteration best
            iteration_best = min(self.ants, key=lambda a: a.distance)

            # Update global best
            if iteration_best.distance < best_distance:
                best_tour = iteration_best.tour.copy()
                best_distance = iteration_best.distance
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            # Early stopping
            if iterations_without_improvement >= self.early_stop_iterations:
                break

            # Choose which best to use for pheromone update
            if self.use_global_best:
                update_ant = Ant(tour=best_tour, distance=best_distance)
            else:
                update_ant = iteration_best

            self._update_pheromone(update_ant)

        # Apply local search
        if self.use_local_search and best_tour:
            best_tour, best_distance = self._local_search_2opt(best_tour)

        computation_time = time.time() - start_time

        return self._tour_to_solution(best_tour, computation_time)


class AntColonySystemCETSP(AntColonyCETSP):
    """
    Ant Colony System (ACS) variant.

    Key differences:
    - Uses pseudo-random proportional rule for selection
    - Local pheromone update after each step
    - Global pheromone update only by best ant
    """

    def __init__(
        self,
        problem: CETSP,
        n_ants: int = 10,
        n_iterations: int = 100,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        q0: float = 0.9,
        xi: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__(
            problem=problem,
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            use_elitist=False,
            use_local_search=True,
            seed=seed,
        )

        self.q0 = q0  # Exploitation vs exploration balance
        self.xi = xi  # Local pheromone decay
        self.tau0 = self.initial_pheromone  # Initial pheromone for local update

    def _select_next_node(self, ant: Ant, current: int) -> int:
        """
        Select next node using pseudo-random proportional rule.

        With probability q0, choose the best option (exploitation).
        Otherwise, use standard probabilistic selection (exploration).
        """
        unvisited = [i for i in range(1, self.n_nodes) if i not in ant.visited]

        if not unvisited:
            return 0

        # Calculate attractiveness for each unvisited node
        attractiveness = np.zeros(len(unvisited))
        for idx, j in enumerate(unvisited):
            tau = self.pheromone[current, j]
            eta = self.heuristic[current, j]
            attractiveness[idx] = tau * (eta**self.beta)

        if random.random() < self.q0:
            # Exploitation: choose best
            best_idx = np.argmax(attractiveness)
            return unvisited[best_idx]
        else:
            # Exploration: probabilistic selection
            prob_sum = attractiveness.sum()
            if prob_sum > 0:
                probabilities = attractiveness / prob_sum
            else:
                probabilities = np.ones(len(unvisited)) / len(unvisited)

            r = random.random()
            cumsum = 0.0
            for idx, prob in enumerate(probabilities):
                cumsum += prob
                if r <= cumsum:
                    return unvisited[idx]

            return unvisited[-1]

    def _local_pheromone_update(self, from_node: int, to_node: int) -> None:
        """Apply local pheromone update after ant moves."""
        # τ(i,j) = (1-ξ) * τ(i,j) + ξ * τ0
        self.pheromone[from_node, to_node] = (1 - self.xi) * self.pheromone[
            from_node, to_node
        ] + self.xi * self.tau0
        self.pheromone[to_node, from_node] = self.pheromone[from_node, to_node]

    def _construct_solution(self, ant: Ant) -> None:
        """Construct solution with local pheromone update."""
        ant.reset()

        current = 0
        ant.visited.add(0)

        while len(ant.visited) < self.n_nodes:
            next_node = self._select_next_node(ant, current)

            if next_node == 0:
                break

            # Local pheromone update
            self._local_pheromone_update(current, next_node)

            ant.tour.append(next_node - 1)
            ant.visited.add(next_node)
            current = next_node

        # Local update for return to depot
        if current != 0:
            self._local_pheromone_update(current, 0)

        ant.distance = self._calculate_tour_distance(ant.tour)

    def _update_pheromone(self, best_ant: Ant) -> None:
        """Global pheromone update by best ant only."""
        # Global update only on edges used by best ant
        if best_ant.tour and best_ant.distance < float("inf"):
            deposit = 1.0 / best_ant.distance

            # Depot to first
            first_node = best_ant.tour[0] + 1
            self.pheromone[0, first_node] = (1 - self.rho) * self.pheromone[
                0, first_node
            ] + self.rho * deposit
            self.pheromone[first_node, 0] = self.pheromone[0, first_node]

            # Between customers
            for i in range(len(best_ant.tour) - 1):
                from_node = best_ant.tour[i] + 1
                to_node = best_ant.tour[i + 1] + 1
                self.pheromone[from_node, to_node] = (1 - self.rho) * self.pheromone[
                    from_node, to_node
                ] + self.rho * deposit
                self.pheromone[to_node, from_node] = self.pheromone[from_node, to_node]

            # Last to depot
            last_node = best_ant.tour[-1] + 1
            self.pheromone[last_node, 0] = (1 - self.rho) * self.pheromone[
                last_node, 0
            ] + self.rho * deposit
            self.pheromone[0, last_node] = self.pheromone[last_node, 0]
