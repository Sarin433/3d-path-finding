"""Grey Wolf Optimization (GWO) solver for CETSP.

Grey Wolf Optimization is a nature-inspired metaheuristic algorithm based on
the social hierarchy and hunting behavior of grey wolves. The algorithm mimics
the leadership hierarchy and hunting mechanism of grey wolves in nature.

Social Hierarchy:
- Alpha (α): The leader, best solution found
- Beta (β): Second best, assists alpha in decision-making
- Delta (δ): Third best, submits to alpha and beta
- Omega (ω): Rest of the wolves, follow the leaders

Hunting Phases:
1. Encircling prey: Wolves surround the prey
2. Hunting: Guided by alpha, beta, delta positions
3. Attacking: Converge towards the prey (solution)

Reference:
    Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014).
    Grey Wolf Optimizer. Advances in Engineering Software, 69, 46-61.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from ..problem import CETSP, CETSPPath, CETSPSolution
from .base import CETSPSolver


@dataclass
class Wolf:
    """Represents a wolf (solution) in GWO."""

    position: list[int] = field(default_factory=list)  # Tour representation
    fitness: float = float("inf")

    def copy(self) -> Wolf:
        return Wolf(position=self.position.copy(), fitness=self.fitness)


class GreyWolfCETSP(CETSPSolver):
    """
    Grey Wolf Optimization solver for Close-Enough TSP.

    GWO is adapted for the discrete TSP domain where:
    - Position: A permutation representing the tour
    - Movement: Position updates using crossover-like operations

    The algorithm maintains three leader wolves (alpha, beta, delta) that
    guide the search, while omega wolves explore around these leaders.

    Parameters:
        n_wolves: Number of wolves (population size)
        n_iterations: Maximum number of iterations
        use_local_search: Apply 2-opt local search to best solution
        seed: Random seed for reproducibility
        early_stop_iterations: Stop if no improvement for this many iterations

    Example:
        >>> solver = GreyWolfCETSP(problem, n_wolves=30)
        >>> solution = solver.solve()
    """

    def __init__(
        self,
        problem: CETSP,
        n_wolves: int = 30,
        n_iterations: int = 200,
        use_local_search: bool = True,
        seed: int | None = None,
        early_stop_iterations: int = 50,
    ):
        super().__init__(problem)
        self.n_wolves = n_wolves
        self.n_iterations = n_iterations
        self.use_local_search = use_local_search
        self.early_stop_iterations = early_stop_iterations

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.n_customers = len(problem.customers)

        # Wolf pack
        self.wolves: list[Wolf] = []

        # Leader wolves
        self.alpha: Wolf | None = None  # Best
        self.beta: Wolf | None = None  # Second best
        self.delta: Wolf | None = None  # Third best

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

    def _create_random_wolf(self) -> Wolf:
        """Create a wolf with random tour."""
        tour = list(range(self.n_customers))
        random.shuffle(tour)

        return Wolf(position=tour, fitness=self._calculate_tour_distance(tour))

    def _create_greedy_wolf(self) -> Wolf:
        """Create a wolf using greedy nearest neighbor construction."""
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

        return Wolf(position=tour, fitness=self._calculate_tour_distance(tour))

    def _initialize_pack(self) -> None:
        """Initialize the wolf pack."""
        self.wolves = []

        # Add greedy solution
        self.wolves.append(self._create_greedy_wolf())

        # Add random solutions
        while len(self.wolves) < self.n_wolves:
            self.wolves.append(self._create_random_wolf())

        # Initialize leaders
        self._update_leaders()

    def _update_leaders(self) -> None:
        """Update alpha, beta, delta wolves based on fitness."""
        # Sort wolves by fitness (ascending - lower is better)
        sorted_wolves = sorted(self.wolves, key=lambda w: w.fitness)

        self.alpha = sorted_wolves[0].copy()
        self.beta = sorted_wolves[1].copy() if len(sorted_wolves) > 1 else sorted_wolves[0].copy()
        self.delta = sorted_wolves[2].copy() if len(sorted_wolves) > 2 else sorted_wolves[0].copy()

    def _position_based_crossover(
        self, parent1: list[int], parent2: list[int], weight: float
    ) -> list[int]:
        """
        Position-based crossover for discrete optimization.

        Creates a child by taking positions from parent1 with probability
        proportional to weight, and filling remaining from parent2.

        Args:
            parent1: First parent tour
            parent2: Second parent tour
            weight: Probability weight for parent1 (0 to 1)

        Returns:
            Child tour
        """
        n = len(parent1)
        child = [-1] * n
        used = set()

        # Copy some positions from parent1 based on weight
        for i in range(n):
            if random.random() < weight:
                city = parent1[i]
                if city not in used:
                    child[i] = city
                    used.add(city)

        # Fill remaining positions from parent2 in order
        p2_idx = 0
        for i in range(n):
            if child[i] == -1:
                while parent2[p2_idx] in used:
                    p2_idx += 1
                child[i] = parent2[p2_idx]
                used.add(parent2[p2_idx])
                p2_idx += 1

        return child

    def _order_crossover(self, parent1: list[int], parent2: list[int]) -> list[int]:
        """
        Order crossover (OX) operator.

        Takes a random segment from parent1 and fills the rest from parent2
        while maintaining relative order.
        """
        n = len(parent1)

        # Select random segment
        start = random.randint(0, n - 2)
        end = random.randint(start + 1, n - 1)

        # Copy segment from parent1
        child = [-1] * n
        segment = set()
        for i in range(start, end + 1):
            child[i] = parent1[i]
            segment.add(parent1[i])

        # Fill remaining from parent2 in order
        p2_cities = [c for c in parent2 if c not in segment]
        j = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = p2_cities[j]
                j += 1

        return child

    def _update_wolf_position(self, wolf: Wolf, a: float) -> None:
        """
        Update wolf position guided by alpha, beta, delta.

        In the original GWO, positions are updated using:
        X(t+1) = (X1 + X2 + X3) / 3

        where X1, X2, X3 are positions calculated from alpha, beta, delta.

        For discrete TSP, we use crossover operations to combine tours.

        Args:
            wolf: The wolf to update
            a: Linearly decreasing parameter from 2 to 0
        """
        # Calculate adaptive weights based on 'a' parameter
        # As 'a' decreases, wolves converge more towards alpha
        r1, r2, r3 = random.random(), random.random(), random.random()

        # Coefficient vectors (simplified for discrete case)
        A1 = 2 * a * r1 - a  # Range: [-a, a]
        A2 = 2 * a * r2 - a
        A3 = 2 * a * r3 - a

        # Convert A values to weights (higher A = more exploration)
        w_alpha = 1.0 / (1.0 + abs(A1))
        w_beta = 1.0 / (1.0 + abs(A2))
        w_delta = 1.0 / (1.0 + abs(A3))

        # Normalize weights
        total = w_alpha + w_beta + w_delta
        w_alpha /= total
        w_beta /= total
        w_delta /= total

        # Create three candidate positions guided by leaders
        if random.random() < 0.5:
            # Position-based approach
            X1 = self._position_based_crossover(self.alpha.position, wolf.position, w_alpha)
            X2 = self._position_based_crossover(self.beta.position, wolf.position, w_beta)
            X3 = self._position_based_crossover(self.delta.position, wolf.position, w_delta)
        else:
            # Order crossover approach
            X1 = self._order_crossover(self.alpha.position, wolf.position)
            X2 = self._order_crossover(self.beta.position, wolf.position)
            X3 = self._order_crossover(self.delta.position, wolf.position)

        # Calculate fitness of candidates
        f1 = self._calculate_tour_distance(X1)
        f2 = self._calculate_tour_distance(X2)
        f3 = self._calculate_tour_distance(X3)

        # Select best candidate
        candidates = [(X1, f1), (X2, f2), (X3, f3)]
        best_candidate = min(candidates, key=lambda x: x[1])

        # Apply mutation with probability based on 'a'
        # Higher 'a' = more exploration = higher mutation rate
        mutation_rate = a / 4.0  # Range: [0, 0.5]

        if random.random() < mutation_rate:
            new_position = self._mutate(best_candidate[0])
            new_fitness = self._calculate_tour_distance(new_position)
        else:
            new_position = best_candidate[0]
            new_fitness = best_candidate[1]

        # Update if improved
        if new_fitness < wolf.fitness:
            wolf.position = new_position
            wolf.fitness = new_fitness

    def _mutate(self, tour: list[int]) -> list[int]:
        """Apply mutation operator (2-opt move or swap)."""
        new_tour = tour.copy()
        n = len(new_tour)

        if random.random() < 0.5:
            # 2-opt move
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            new_tour[i : j + 1] = reversed(new_tour[i : j + 1])
        else:
            # Swap mutation
            i, j = random.sample(range(n), 2)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

        return new_tour

    def _local_search_2opt(self, tour: list[int]) -> tuple[list[int], float]:
        """Apply 2-opt local search to improve tour."""
        improved = True
        current_tour = tour.copy()
        current_fitness = self._calculate_tour_distance(current_tour)

        while improved:
            improved = False
            for i in range(len(current_tour) - 1):
                for j in range(i + 2, len(current_tour)):
                    # Try 2-opt swap
                    new_tour = current_tour.copy()
                    new_tour[i : j + 1] = reversed(new_tour[i : j + 1])
                    new_fitness = self._calculate_tour_distance(new_tour)

                    if new_fitness < current_fitness - 1e-10:
                        current_tour = new_tour
                        current_fitness = new_fitness
                        improved = True
                        break
                if improved:
                    break

        return current_tour, current_fitness

    def _build_solution(self, tour: list[int]) -> CETSPSolution:
        """Build CETSP solution from tour."""
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
        Solve CETSP using Grey Wolf Optimization.

        Returns:
            CETSPSolution with the best tour found
        """
        start_time = time.time()

        # Initialize pack
        self._initialize_pack()

        best_fitness_history = []
        no_improvement_count = 0

        for iteration in range(self.n_iterations):
            # Linearly decrease 'a' from 2 to 0
            a = 2 - iteration * (2 / self.n_iterations)

            # Update each wolf's position
            for wolf in self.wolves:
                # Skip leaders (they guide, not follow)
                if wolf.fitness > self.delta.fitness:
                    self._update_wolf_position(wolf, a)

            # Update leaders
            self._update_leaders()

            # Track progress
            best_fitness_history.append(self.alpha.fitness)

            # Early stopping check
            if len(best_fitness_history) > 1:
                if abs(best_fitness_history[-1] - best_fitness_history[-2]) < 1e-10:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0

            if no_improvement_count >= self.early_stop_iterations:
                break

        # Get best tour
        best_tour = self.alpha.position

        # Apply local search if enabled
        if self.use_local_search:
            best_tour, _ = self._local_search_2opt(best_tour)

        # Build and return solution
        solution = self._build_solution(best_tour)
        solution.computation_time = time.time() - start_time
        solution.method = "gwo"

        return solution


class EnhancedGreyWolfCETSP(CETSPSolver):
    """
    Enhanced Grey Wolf Optimization with improved exploration and exploitation.

    Improvements over standard GWO:
    1. Opposition-based learning for initialization
    2. Adaptive parameter control
    3. Dimension learning hunting (DLH) strategy
    4. Lévy flight for better exploration
    5. Memory mechanism for elite solutions

    Parameters:
        n_wolves: Number of wolves (population size)
        n_iterations: Maximum number of iterations
        elite_size: Number of elite solutions to preserve
        levy_factor: Scale factor for Lévy flight
        use_local_search: Apply 2-opt local search
        seed: Random seed

    Reference:
        Based on enhanced GWO variants from literature.
    """

    def __init__(
        self,
        problem: CETSP,
        n_wolves: int = 40,
        n_iterations: int = 300,
        elite_size: int = 5,
        levy_factor: float = 1.5,
        use_local_search: bool = True,
        seed: int | None = None,
        early_stop_iterations: int = 60,
    ):
        super().__init__(problem)
        self.n_wolves = n_wolves
        self.n_iterations = n_iterations
        self.elite_size = elite_size
        self.levy_factor = levy_factor
        self.use_local_search = use_local_search
        self.early_stop_iterations = early_stop_iterations

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.n_customers = len(problem.customers)

        # Wolf pack
        self.wolves: list[Wolf] = []

        # Leader wolves
        self.alpha: Wolf | None = None
        self.beta: Wolf | None = None
        self.delta: Wolf | None = None

        # Elite archive
        self.elite_archive: list[Wolf] = []

    def _calculate_tour_distance(self, tour: list[int]) -> float:
        """Calculate CETSP tour distance."""
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

    def _create_wolf(self, tour: list[int]) -> Wolf:
        """Create a wolf with given tour."""
        return Wolf(position=tour, fitness=self._calculate_tour_distance(tour))

    def _create_random_wolf(self) -> Wolf:
        """Create a wolf with random tour."""
        tour = list(range(self.n_customers))
        random.shuffle(tour)
        return self._create_wolf(tour)

    def _create_greedy_wolf(self) -> Wolf:
        """Create a wolf using greedy construction."""
        tour = []
        unvisited = set(range(self.n_customers))
        current_pos = self.problem.depot.position.copy()

        while unvisited:
            best_idx = min(
                unvisited,
                key=lambda idx: np.linalg.norm(
                    self.problem.customers[idx].coverage_entry_point(current_pos) - current_pos
                ),
            )
            tour.append(best_idx)
            unvisited.remove(best_idx)
            current_pos = self.problem.customers[best_idx].coverage_entry_point(current_pos)

        return self._create_wolf(tour)

    def _opposite_tour(self, tour: list[int]) -> list[int]:
        """Create opposite tour for opposition-based learning."""
        # Reverse the tour
        return tour[::-1]

    def _initialize_pack(self) -> None:
        """Initialize wolf pack with opposition-based learning."""
        self.wolves = []

        # Add greedy solution and its opposite
        greedy = self._create_greedy_wolf()
        self.wolves.append(greedy)
        opposite_greedy = self._create_wolf(self._opposite_tour(greedy.position))
        self.wolves.append(opposite_greedy)

        # Generate random solutions with opposition
        while len(self.wolves) < self.n_wolves:
            wolf = self._create_random_wolf()
            opposite = self._create_wolf(self._opposite_tour(wolf.position))

            # Keep the better one
            if wolf.fitness <= opposite.fitness:
                self.wolves.append(wolf)
            else:
                self.wolves.append(opposite)

        # Trim to exact size
        self.wolves = self.wolves[: self.n_wolves]

        # Initialize leaders and archive
        self._update_leaders()
        self._update_elite_archive()

    def _update_leaders(self) -> None:
        """Update alpha, beta, delta wolves."""
        sorted_wolves = sorted(self.wolves, key=lambda w: w.fitness)

        self.alpha = sorted_wolves[0].copy()
        self.beta = sorted_wolves[1].copy() if len(sorted_wolves) > 1 else sorted_wolves[0].copy()
        self.delta = sorted_wolves[2].copy() if len(sorted_wolves) > 2 else sorted_wolves[0].copy()

    def _update_elite_archive(self) -> None:
        """Update elite archive with best solutions."""
        # Add current leaders to archive candidates
        candidates = self.elite_archive + [self.alpha.copy(), self.beta.copy(), self.delta.copy()]

        # Remove duplicates based on fitness
        unique = {}
        for wolf in candidates:
            key = round(wolf.fitness, 6)
            if key not in unique or wolf.fitness < unique[key].fitness:
                unique[key] = wolf

        # Keep top elite_size
        self.elite_archive = sorted(unique.values(), key=lambda w: w.fitness)[: self.elite_size]

    def _levy_flight(self, beta: float = 1.5) -> float:
        """Generate Lévy flight step."""
        sigma_u = (
            math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, 1)

        step = u / (abs(v) ** (1 / beta))
        return step

    def _dimension_learning(self, wolf: Wolf) -> list[int]:
        """
        Dimension Learning Hunting (DLH) strategy.

        Each dimension (position) learns from a randomly selected neighbor
        that is better in that dimension's context.
        """
        new_tour = wolf.position.copy()
        n = len(new_tour)

        # Select random reference wolves for learning
        reference_wolves = [self.alpha, self.beta, self.delta]
        if self.elite_archive:
            reference_wolves.extend(self.elite_archive)

        # For each position, decide whether to learn from a reference
        for i in range(n):
            if random.random() < 0.3:  # Learning probability
                ref = random.choice(reference_wolves)
                # Find where the city at position i in reference appears
                city = ref.position[i]
                # Find current position of this city
                current_pos = new_tour.index(city)
                # Swap
                new_tour[i], new_tour[current_pos] = new_tour[current_pos], new_tour[i]

        return new_tour

    def _adaptive_update(self, wolf: Wolf, a: float, iteration: int) -> None:
        """
        Update wolf with adaptive strategies.

        Uses different strategies based on iteration progress and randomness.
        """
        progress = iteration / self.n_iterations

        # Strategy selection based on progress
        if progress < 0.3:
            # Early: More exploration
            strategy = random.choices(["dlh", "levy", "crossover"], weights=[0.3, 0.4, 0.3])[0]
        elif progress < 0.7:
            # Middle: Balanced
            strategy = random.choices(
                ["dlh", "levy", "crossover", "local"], weights=[0.25, 0.25, 0.25, 0.25]
            )[0]
        else:
            # Late: More exploitation
            strategy = random.choices(["dlh", "crossover", "local"], weights=[0.2, 0.3, 0.5])[0]

        if strategy == "dlh":
            # Dimension learning hunting
            new_tour = self._dimension_learning(wolf)

        elif strategy == "levy":
            # Lévy flight inspired mutation
            new_tour = wolf.position.copy()
            levy_steps = int(abs(self._levy_flight(self.levy_factor)) * len(new_tour) / 4)
            levy_steps = max(1, min(levy_steps, len(new_tour) // 2))

            for _ in range(levy_steps):
                i, j = random.sample(range(len(new_tour)), 2)
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

        elif strategy == "crossover":
            # Leader-guided crossover
            leader = random.choice([self.alpha, self.beta, self.delta])
            new_tour = self._order_crossover(leader.position, wolf.position)

        else:  # local
            # Local 2-opt move
            new_tour = wolf.position.copy()
            i = random.randint(0, len(new_tour) - 2)
            j = random.randint(i + 1, len(new_tour) - 1)
            new_tour[i : j + 1] = reversed(new_tour[i : j + 1])

        new_fitness = self._calculate_tour_distance(new_tour)

        # Accept if improved or with small probability (simulated annealing)
        if new_fitness < wolf.fitness or random.random() < 0.05 * (1 - progress):
            wolf.position = new_tour
            wolf.fitness = new_fitness

    def _order_crossover(self, parent1: list[int], parent2: list[int]) -> list[int]:
        """Order crossover operator."""
        n = len(parent1)
        start = random.randint(0, n - 2)
        end = random.randint(start + 1, n - 1)

        child = [-1] * n
        segment = set()
        for i in range(start, end + 1):
            child[i] = parent1[i]
            segment.add(parent1[i])

        p2_cities = [c for c in parent2 if c not in segment]
        j = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = p2_cities[j]
                j += 1

        return child

    def _local_search_2opt(self, tour: list[int]) -> tuple[list[int], float]:
        """Apply 2-opt local search."""
        improved = True
        current_tour = tour.copy()
        current_fitness = self._calculate_tour_distance(current_tour)

        while improved:
            improved = False
            for i in range(len(current_tour) - 1):
                for j in range(i + 2, len(current_tour)):
                    new_tour = current_tour.copy()
                    new_tour[i : j + 1] = reversed(new_tour[i : j + 1])
                    new_fitness = self._calculate_tour_distance(new_tour)

                    if new_fitness < current_fitness - 1e-10:
                        current_tour = new_tour
                        current_fitness = new_fitness
                        improved = True
                        break
                if improved:
                    break

        return current_tour, current_fitness

    def _build_solution(self, tour: list[int]) -> CETSPSolution:
        """Build CETSP solution from tour."""
        path = CETSPPath()

        current_pos = self.problem.depot.position.copy()
        path.add_waypoint(current_pos, self.problem.depot)

        for customer_idx in tour:
            customer = self.problem.customers[customer_idx]
            entry_point = customer.coverage_entry_point(current_pos)
            path.add_waypoint(entry_point, customer)
            current_pos = entry_point

        if self.problem.return_to_depot:
            path.add_waypoint(self.problem.depot.position.copy())

        return CETSPSolution(path=path, is_complete=True)

    def solve(self) -> CETSPSolution:
        """
        Solve CETSP using Enhanced Grey Wolf Optimization.

        Returns:
            CETSPSolution with the best tour found
        """
        start_time = time.time()

        # Initialize pack with opposition-based learning
        self._initialize_pack()

        best_fitness_history = []
        no_improvement_count = 0
        global_best_fitness = self.alpha.fitness

        for iteration in range(self.n_iterations):
            # Adaptive 'a' parameter with non-linear decrease
            a = 2 * (1 - (iteration / self.n_iterations) ** 2)

            # Update wolves
            for wolf in self.wolves:
                if wolf.fitness > self.delta.fitness:
                    self._adaptive_update(wolf, a, iteration)

            # Update leaders
            self._update_leaders()

            # Update elite archive
            self._update_elite_archive()

            # Track progress
            best_fitness_history.append(self.alpha.fitness)

            # Early stopping
            if self.alpha.fitness < global_best_fitness - 1e-10:
                global_best_fitness = self.alpha.fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.early_stop_iterations:
                break

            # Reinitialize worst wolves periodically
            if iteration > 0 and iteration % 50 == 0:
                sorted_wolves = sorted(self.wolves, key=lambda w: w.fitness)
                n_replace = max(1, self.n_wolves // 5)
                for i in range(n_replace):
                    sorted_wolves[-(i + 1)] = self._create_random_wolf()
                self.wolves = sorted_wolves

        # Get best tour from elite archive
        if self.elite_archive:
            best_wolf = min(self.elite_archive, key=lambda w: w.fitness)
        else:
            best_wolf = self.alpha

        best_tour = best_wolf.position

        # Apply local search
        if self.use_local_search:
            best_tour, _ = self._local_search_2opt(best_tour)

        # Build solution
        solution = self._build_solution(best_tour)
        solution.computation_time = time.time() - start_time
        solution.method = "egwo"

        return solution
