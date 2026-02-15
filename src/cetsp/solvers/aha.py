"""Artificial Hummingbird Algorithm (AHA) solver for CETSP."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np

from ..problem import CETSP, CETSPPath, CETSPSolution
from .base import CETSPSolver


@dataclass
class Hummingbird:
    """Represents a hummingbird (solution) in the algorithm."""

    tour: list[int] = field(default_factory=list)
    fitness: float = float("inf")
    nectar: float = 0.0  # Inverse of fitness (higher is better)

    def copy(self) -> Hummingbird:
        return Hummingbird(tour=self.tour.copy(), fitness=self.fitness, nectar=self.nectar)


class ArtificialHummingbirdCETSP(CETSPSolver):
    """
    Artificial Hummingbird Algorithm solver for Close-Enough TSP.

    AHA is a metaheuristic inspired by the foraging behavior of hummingbirds.
    Hummingbirds use three main foraging strategies:

    1. Guided Foraging: Move toward the best food source (exploitation)
    2. Territorial Foraging: Explore around current position (local search)
    3. Migration Foraging: Random movement when stuck (exploration)

    The algorithm maintains a visit table to track food source visits
    and balances exploration vs exploitation.

    Parameters:
        n_hummingbirds: Number of hummingbirds (population size)
        n_iterations: Maximum number of iterations
        visit_table_rate: Rate of visit table decay
        guided_prob: Probability of guided foraging
        territorial_prob: Probability of territorial foraging
        use_local_search: Apply 2-opt local search to best solution
        seed: Random seed for reproducibility

    Reference:
        Zhao, W., Wang, L., & Mirjalili, S. (2022). Artificial hummingbird
        algorithm: A new bio-inspired optimizer with its engineering applications.
        Computer Methods in Applied Mechanics and Engineering.

    Example:
        >>> solver = ArtificialHummingbirdCETSP(problem, n_hummingbirds=30)
        >>> solution = solver.solve()
    """

    def __init__(
        self,
        problem: CETSP,
        n_hummingbirds: int = 30,
        n_iterations: int = 200,
        visit_table_rate: float = 0.9,
        guided_prob: float = 0.5,
        territorial_prob: float = 0.3,
        use_local_search: bool = True,
        seed: int | None = None,
        early_stop_iterations: int = 50,
    ):
        super().__init__(problem)
        self.n_hummingbirds = n_hummingbirds
        self.n_iterations = n_iterations
        self.visit_table_rate = visit_table_rate
        self.guided_prob = guided_prob
        self.territorial_prob = territorial_prob
        self.use_local_search = use_local_search
        self.early_stop_iterations = early_stop_iterations

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.n_customers = len(problem.customers)

        # Initialize population
        self.hummingbirds: list[Hummingbird] = []

        # Visit table: tracks how often each hummingbird visits each food source
        self.visit_table = np.ones((n_hummingbirds, n_hummingbirds))

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

    def _create_random_hummingbird(self) -> Hummingbird:
        """Create a hummingbird with random tour."""
        tour = list(range(self.n_customers))
        random.shuffle(tour)

        hb = Hummingbird(tour=tour)
        hb.fitness = self._calculate_tour_distance(tour)
        hb.nectar = 1.0 / hb.fitness if hb.fitness > 0 else 0
        return hb

    def _create_greedy_hummingbird(self) -> Hummingbird:
        """Create a hummingbird using greedy construction."""
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

        hb = Hummingbird(tour=tour)
        hb.fitness = self._calculate_tour_distance(tour)
        hb.nectar = 1.0 / hb.fitness if hb.fitness > 0 else 0
        return hb

    def _initialize_population(self) -> None:
        """Initialize population of hummingbirds."""
        self.hummingbirds = []

        # Add greedy solution
        self.hummingbirds.append(self._create_greedy_hummingbird())

        # Add random solutions
        while len(self.hummingbirds) < self.n_hummingbirds:
            self.hummingbirds.append(self._create_random_hummingbird())

        # Initialize visit table
        self.visit_table = np.ones((self.n_hummingbirds, self.n_hummingbirds))

    def _get_best_hummingbird(self) -> Hummingbird:
        """Get the hummingbird with best fitness (lowest distance)."""
        return min(self.hummingbirds, key=lambda h: h.fitness)

    def _guided_foraging(self, hb_idx: int, target_idx: int) -> list[int]:
        """
        Guided foraging: Move toward target food source.

        Uses crossover-like operation to combine current tour with target tour.
        """
        current = self.hummingbirds[hb_idx]
        target = self.hummingbirds[target_idx]

        # Order crossover (OX)
        size = len(current.tour)
        start, end = sorted(random.sample(range(size), 2))

        new_tour = [None] * size
        new_tour[start : end + 1] = current.tour[start : end + 1]

        # Fill remaining from target in order
        target_idx_pos = (end + 1) % size
        new_idx = (end + 1) % size

        while None in new_tour:
            if target.tour[target_idx_pos] not in new_tour:
                new_tour[new_idx] = target.tour[target_idx_pos]
                new_idx = (new_idx + 1) % size
            target_idx_pos = (target_idx_pos + 1) % size

        return new_tour

    def _territorial_foraging(self, hb_idx: int) -> list[int]:
        """
        Territorial foraging: Explore around current position.

        Applies local perturbations (swap, insert, reverse) to current tour.
        """
        current = self.hummingbirds[hb_idx]
        tour = current.tour.copy()

        # Choose random perturbation
        perturbation = random.choice(["swap", "insert", "reverse", "or_opt"])

        if perturbation == "swap":
            # Swap two random positions
            if len(tour) >= 2:
                i, j = random.sample(range(len(tour)), 2)
                tour[i], tour[j] = tour[j], tour[i]

        elif perturbation == "insert":
            # Remove and reinsert at different position
            if len(tour) >= 2:
                i = random.randrange(len(tour))
                j = random.randrange(len(tour))
                if i != j:
                    customer = tour.pop(i)
                    tour.insert(j, customer)

        elif perturbation == "reverse":
            # Reverse a random segment (2-opt move)
            if len(tour) >= 2:
                i, j = sorted(random.sample(range(len(tour)), 2))
                tour[i : j + 1] = reversed(tour[i : j + 1])

        elif perturbation == "or_opt" and len(tour) >= 3:
            # Move a segment of 1-3 customers to another position
            seg_len = random.randint(1, min(3, len(tour) - 1))
            start = random.randrange(len(tour) - seg_len + 1)
            segment = tour[start : start + seg_len]
            del tour[start : start + seg_len]
            insert_pos = random.randrange(len(tour) + 1)
            tour[insert_pos:insert_pos] = segment

        return tour

    def _migration_foraging(self, hb_idx: int) -> list[int]:
        """
        Migration foraging: Random exploration.

        Creates a mostly new random tour with some elements from current.
        """
        current = self.hummingbirds[hb_idx]

        # Keep small portion from current tour
        keep_ratio = random.uniform(0.1, 0.3)
        keep_count = max(1, int(len(current.tour) * keep_ratio))

        # Select random positions to keep
        keep_positions = random.sample(range(len(current.tour)), keep_count)
        kept_elements = {current.tour[pos] for pos in keep_positions}

        # Build new tour
        new_tour = [None] * len(current.tour)

        # Place kept elements
        for pos in keep_positions:
            new_tour[pos] = current.tour[pos]

        # Fill remaining randomly
        remaining = [i for i in range(self.n_customers) if i not in kept_elements]
        random.shuffle(remaining)

        rem_idx = 0
        for i in range(len(new_tour)):
            if new_tour[i] is None:
                new_tour[i] = remaining[rem_idx]
                rem_idx += 1

        return new_tour

    def _select_food_source(self, hb_idx: int) -> int:
        """Select target food source based on visit table."""
        # Calculate selection probabilities based on visit table
        visits = self.visit_table[hb_idx].copy()

        # Don't select self
        visits[hb_idx] = 0

        # Add nectar-based attraction
        nectar_weights = np.array([h.nectar for h in self.hummingbirds])
        nectar_weights[hb_idx] = 0

        # Combine visit table (lower is better) and nectar (higher is better)
        if visits.sum() > 0:
            visit_probs = 1.0 / (visits + 1e-6)
            visit_probs[hb_idx] = 0

            # Weighted combination
            combined = visit_probs * (nectar_weights + 1e-6)
            probs = combined / combined.sum()
        else:
            probs = (
                nectar_weights / nectar_weights.sum()
                if nectar_weights.sum() > 0
                else np.ones(self.n_hummingbirds) / self.n_hummingbirds
            )
            probs[hb_idx] = 0
            probs = probs / probs.sum()

        # Roulette wheel selection
        return np.random.choice(self.n_hummingbirds, p=probs)

    def _update_visit_table(self, hb_idx: int, target_idx: int) -> None:
        """Update visit table after foraging."""
        # Increase visit count
        self.visit_table[hb_idx, target_idx] += 1

        # Decay all visit counts
        self.visit_table *= self.visit_table_rate
        self.visit_table = np.maximum(self.visit_table, 0.1)

    def _local_search_2opt(self, tour: list[int]) -> tuple[list[int], float]:
        """Apply 2-opt local search."""
        improved = True
        best_tour = tour.copy()
        best_dist = self._calculate_tour_distance(best_tour)

        while improved:
            improved = False

            for i in range(len(best_tour) - 1):
                for j in range(i + 2, len(best_tour)):
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
        Solve CETSP using Artificial Hummingbird Algorithm.

        Returns:
            CETSPSolution with the best tour found
        """
        start_time = time.time()

        # Initialize population
        self._initialize_population()

        best_hummingbird = self._get_best_hummingbird().copy()
        iterations_without_improvement = 0

        for iteration in range(self.n_iterations):
            # For each hummingbird
            for hb_idx in range(self.n_hummingbirds):
                # Determine foraging strategy
                r = random.random()

                if r < self.guided_prob:
                    # Guided foraging toward best or selected food source
                    if random.random() < 0.5:
                        # Move toward global best
                        target_idx = self.hummingbirds.index(
                            min(self.hummingbirds, key=lambda h: h.fitness)
                        )
                    else:
                        # Move toward selected food source
                        target_idx = self._select_food_source(hb_idx)

                    new_tour = self._guided_foraging(hb_idx, target_idx)
                    self._update_visit_table(hb_idx, target_idx)

                elif r < self.guided_prob + self.territorial_prob:
                    # Territorial foraging (local search)
                    new_tour = self._territorial_foraging(hb_idx)

                else:
                    # Migration foraging (exploration)
                    new_tour = self._migration_foraging(hb_idx)

                # Evaluate new tour
                new_fitness = self._calculate_tour_distance(new_tour)

                # Accept if better (greedy selection)
                if new_fitness < self.hummingbirds[hb_idx].fitness:
                    self.hummingbirds[hb_idx].tour = new_tour
                    self.hummingbirds[hb_idx].fitness = new_fitness
                    self.hummingbirds[hb_idx].nectar = 1.0 / new_fitness if new_fitness > 0 else 0

            # Update global best
            current_best = self._get_best_hummingbird()
            if current_best.fitness < best_hummingbird.fitness:
                best_hummingbird = current_best.copy()
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            # Early stopping
            if iterations_without_improvement >= self.early_stop_iterations:
                break

            # Periodic restart of worst solutions
            if iteration > 0 and iteration % 30 == 0:
                # Replace worst 20% with new random solutions
                sorted_hbs = sorted(
                    enumerate(self.hummingbirds), key=lambda x: x[1].fitness, reverse=True
                )
                replace_count = max(1, self.n_hummingbirds // 5)

                for i in range(replace_count):
                    idx = sorted_hbs[i][0]
                    if self.hummingbirds[idx].fitness > best_hummingbird.fitness:
                        self.hummingbirds[idx] = self._create_random_hummingbird()

        # Apply local search to best solution
        if self.use_local_search:
            best_tour, best_dist = self._local_search_2opt(best_hummingbird.tour)
            best_hummingbird.tour = best_tour
            best_hummingbird.fitness = best_dist

        computation_time = time.time() - start_time

        return self._tour_to_solution(best_hummingbird.tour, computation_time)


class EnhancedAHACETSP(ArtificialHummingbirdCETSP):
    """
    Enhanced Artificial Hummingbird Algorithm with additional features.

    Improvements:
    - Adaptive foraging probabilities based on search progress
    - Memory of best positions for each hummingbird
    - Opposition-based learning for diversity
    """

    def __init__(
        self,
        problem: CETSP,
        n_hummingbirds: int = 30,
        n_iterations: int = 200,
        seed: int | None = None,
    ):
        super().__init__(
            problem=problem,
            n_hummingbirds=n_hummingbirds,
            n_iterations=n_iterations,
            visit_table_rate=0.9,
            guided_prob=0.5,
            territorial_prob=0.3,
            use_local_search=True,
            seed=seed,
        )

        # Memory for each hummingbird's personal best
        self.personal_best: list[Hummingbird] = []

        # Adaptive parameters
        self.initial_guided_prob = 0.5
        self.initial_territorial_prob = 0.3

    def _initialize_population(self) -> None:
        """Initialize population with diversity mechanisms."""
        super()._initialize_population()

        # Initialize personal best memory
        self.personal_best = [hb.copy() for hb in self.hummingbirds]

    def _opposition_based_tour(self, tour: list[int]) -> list[int]:
        """Create opposition-based tour for diversity."""
        # Reverse the tour (simple opposition)
        return tour[::-1]

    def _adapt_parameters(
        self, iteration: int, total_iterations: int, improvement_ratio: float
    ) -> None:
        """Adapt foraging probabilities based on search progress."""
        progress = iteration / total_iterations

        # Early: more exploration, Late: more exploitation
        if improvement_ratio < 0.1:  # Stagnation
            # Increase exploration
            self.guided_prob = max(0.2, self.initial_guided_prob * 0.7)
            self.territorial_prob = max(0.2, self.initial_territorial_prob * 0.8)
        else:
            # Normal adaptive change
            self.guided_prob = self.initial_guided_prob * (1 + 0.3 * progress)
            self.territorial_prob = self.initial_territorial_prob * (1 - 0.2 * progress)

        # Ensure probabilities don't exceed 1
        total = self.guided_prob + self.territorial_prob
        if total > 0.95:
            scale = 0.95 / total
            self.guided_prob *= scale
            self.territorial_prob *= scale

    def solve(self) -> CETSPSolution:
        """Solve with enhanced AHA."""
        start_time = time.time()

        self._initialize_population()

        best_hummingbird = self._get_best_hummingbird().copy()
        iterations_without_improvement = 0
        recent_improvements = []

        for iteration in range(self.n_iterations):
            improvements_this_iter = 0

            for hb_idx in range(self.n_hummingbirds):
                r = random.random()

                if r < self.guided_prob:
                    # Guided foraging - toward personal best or global best
                    if random.random() < 0.6:
                        # Use global best
                        target_idx = self.hummingbirds.index(
                            min(self.hummingbirds, key=lambda h: h.fitness)
                        )
                    else:
                        # Use personal best of another hummingbird
                        target_idx = self._select_food_source(hb_idx)

                    new_tour = self._guided_foraging(hb_idx, target_idx)
                    self._update_visit_table(hb_idx, target_idx)

                elif r < self.guided_prob + self.territorial_prob:
                    new_tour = self._territorial_foraging(hb_idx)

                else:
                    # Migration with opposition-based learning
                    if random.random() < 0.3:
                        new_tour = self._opposition_based_tour(self.hummingbirds[hb_idx].tour)
                    else:
                        new_tour = self._migration_foraging(hb_idx)

                new_fitness = self._calculate_tour_distance(new_tour)

                if new_fitness < self.hummingbirds[hb_idx].fitness:
                    self.hummingbirds[hb_idx].tour = new_tour
                    self.hummingbirds[hb_idx].fitness = new_fitness
                    self.hummingbirds[hb_idx].nectar = 1.0 / new_fitness
                    improvements_this_iter += 1

                    # Update personal best
                    if new_fitness < self.personal_best[hb_idx].fitness:
                        self.personal_best[hb_idx] = self.hummingbirds[hb_idx].copy()

            # Track improvements
            recent_improvements.append(improvements_this_iter)
            if len(recent_improvements) > 10:
                recent_improvements.pop(0)

            improvement_ratio = sum(recent_improvements) / (
                len(recent_improvements) * self.n_hummingbirds
            )

            # Adapt parameters
            self._adapt_parameters(iteration, self.n_iterations, improvement_ratio)

            # Update global best
            current_best = self._get_best_hummingbird()
            if current_best.fitness < best_hummingbird.fitness:
                best_hummingbird = current_best.copy()
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            if iterations_without_improvement >= self.early_stop_iterations:
                break

            # Periodic restart with opposition
            if iteration > 0 and iteration % 25 == 0:
                sorted_hbs = sorted(
                    enumerate(self.hummingbirds), key=lambda x: x[1].fitness, reverse=True
                )
                replace_count = max(1, self.n_hummingbirds // 4)

                for i in range(replace_count):
                    idx = sorted_hbs[i][0]
                    if random.random() < 0.5:
                        self.hummingbirds[idx] = self._create_random_hummingbird()
                    else:
                        # Opposition of personal best
                        opp_tour = self._opposition_based_tour(self.personal_best[idx].tour)
                        self.hummingbirds[idx].tour = opp_tour
                        self.hummingbirds[idx].fitness = self._calculate_tour_distance(opp_tour)
                        self.hummingbirds[idx].nectar = 1.0 / self.hummingbirds[idx].fitness

        # Local search
        if self.use_local_search:
            best_tour, best_dist = self._local_search_2opt(best_hummingbird.tour)
            best_hummingbird.tour = best_tour
            best_hummingbird.fitness = best_dist

        computation_time = time.time() - start_time

        return self._tour_to_solution(best_hummingbird.tour, computation_time)
