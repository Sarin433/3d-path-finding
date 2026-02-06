"""Particle Swarm Optimization (PSO) solver for CETSP."""

import random
import time
from dataclasses import dataclass, field

import numpy as np

from ..problem import CETSP, CETSPPath, CETSPSolution
from .base import CETSPSolver


@dataclass
class Particle:
    """Represents a particle (solution) in PSO."""

    position: list[int] = field(default_factory=list)  # Tour representation
    velocity: list[float] = field(default_factory=list)  # Velocity vector
    fitness: float = float("inf")

    # Personal best
    pbest_position: list[int] = field(default_factory=list)
    pbest_fitness: float = float("inf")

    def copy(self) -> Particle:
        return Particle(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            fitness=self.fitness,
            pbest_position=self.pbest_position.copy(),
            pbest_fitness=self.pbest_fitness,
        )


class ParticleSwarmCETSP(CETSPSolver):
    """
    Particle Swarm Optimization solver for Close-Enough TSP.

    PSO is adapted for the discrete TSP domain using a position-velocity
    model where:
    - Position: A permutation representing the tour
    - Velocity: A swap sequence that transforms one tour into another

    The algorithm uses:
    - Cognitive component (c1): Learning from personal best
    - Social component (c2): Learning from global best
    - Inertia weight (w): Balance between exploration and exploitation

    Parameters:
        n_particles: Number of particles (swarm size)
        n_iterations: Maximum number of iterations
        w: Inertia weight (decreases over time)
        c1: Cognitive coefficient (personal best attraction)
        c2: Social coefficient (global best attraction)
        w_min: Minimum inertia weight
        w_max: Maximum inertia weight
        use_local_search: Apply 2-opt local search to best solution
        seed: Random seed for reproducibility

    Reference:
        Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
        Proceedings of ICNN'95.

        Adapted for TSP using swap-based velocity representation.

    Example:
        >>> solver = ParticleSwarmCETSP(problem, n_particles=30)
        >>> solution = solver.solve()
    """

    def __init__(
        self,
        problem: CETSP,
        n_particles: int = 30,
        n_iterations: int = 200,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        use_local_search: bool = True,
        seed: int | None = None,
        early_stop_iterations: int = 50,
    ):
        super().__init__(problem)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.use_local_search = use_local_search
        self.early_stop_iterations = early_stop_iterations

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.n_customers = len(problem.customers)

        # Swarm
        self.particles: list[Particle] = []

        # Global best
        self.gbest_position: list[int] = []
        self.gbest_fitness: float = float("inf")

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

    def _create_random_particle(self) -> Particle:
        """Create a particle with random tour."""
        tour = list(range(self.n_customers))
        random.shuffle(tour)

        particle = Particle(
            position=tour,
            velocity=[0.0] * self.n_customers,
            fitness=self._calculate_tour_distance(tour),
        )

        # Initialize personal best
        particle.pbest_position = tour.copy()
        particle.pbest_fitness = particle.fitness

        return particle

    def _create_greedy_particle(self) -> Particle:
        """Create a particle using greedy construction."""
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

        particle = Particle(
            position=tour,
            velocity=[0.0] * self.n_customers,
            fitness=self._calculate_tour_distance(tour),
        )
        particle.pbest_position = tour.copy()
        particle.pbest_fitness = particle.fitness

        return particle

    def _initialize_swarm(self) -> None:
        """Initialize swarm of particles."""
        self.particles = []

        # Add greedy solution
        self.particles.append(self._create_greedy_particle())

        # Add random solutions
        while len(self.particles) < self.n_particles:
            self.particles.append(self._create_random_particle())

        # Initialize global best
        self._update_global_best()

    def _update_global_best(self) -> None:
        """Update global best from all particles."""
        for particle in self.particles:
            if particle.fitness < self.gbest_fitness:
                self.gbest_fitness = particle.fitness
                self.gbest_position = particle.position.copy()

    def _get_swap_sequence(self, tour1: list[int], tour2: list[int]) -> list[tuple[int, int]]:
        """
        Get swap sequence to transform tour1 into tour2.

        Returns list of (i, j) swaps needed.
        """
        swaps = []
        current = tour1.copy()

        for i in range(len(tour2)):
            if current[i] != tour2[i]:
                # Find where tour2[i] is in current
                j = current.index(tour2[i])
                # Swap
                current[i], current[j] = current[j], current[i]
                swaps.append((i, j))

        return swaps

    def _apply_swap_sequence(
        self, tour: list[int], swaps: list[tuple[int, int]], probability: float
    ) -> list[int]:
        """Apply swap sequence with given probability for each swap."""
        result = tour.copy()

        for i, j in swaps:
            if random.random() < probability:
                result[i], result[j] = result[j], result[i]

        return result

    def _update_velocity_and_position(self, particle: Particle, w: float) -> None:
        """
        Update particle velocity and position using PSO equations.

        For discrete TSP, we use swap-based operations:
        - Velocity is represented as a probability of accepting swaps
        - Position update combines personal best and global best influence
        """
        # Compute swap sequences
        pbest_swaps = self._get_swap_sequence(particle.position, particle.pbest_position)
        gbest_swaps = self._get_swap_sequence(particle.position, self.gbest_position)

        # Apply inertia (keep some of current position)
        new_position = particle.position.copy()

        # Apply cognitive component (move toward personal best)
        r1 = random.random()
        cognitive_prob = self.c1 * r1 / 4.0  # Normalize to reasonable probability
        new_position = self._apply_swap_sequence(new_position, pbest_swaps, cognitive_prob)

        # Apply social component (move toward global best)
        r2 = random.random()
        social_prob = self.c2 * r2 / 4.0
        new_position = self._apply_swap_sequence(new_position, gbest_swaps, social_prob)

        # Apply random perturbation based on inertia (exploration)
        if random.random() < w:
            perturbation = random.choice(["swap", "insert", "reverse"])

            if perturbation == "swap" and len(new_position) >= 2:
                i, j = random.sample(range(len(new_position)), 2)
                new_position[i], new_position[j] = new_position[j], new_position[i]

            elif perturbation == "insert" and len(new_position) >= 2:
                i = random.randrange(len(new_position))
                j = random.randrange(len(new_position))
                if i != j:
                    customer = new_position.pop(i)
                    new_position.insert(j, customer)

            elif perturbation == "reverse" and len(new_position) >= 2:
                i, j = sorted(random.sample(range(len(new_position)), 2))
                new_position[i : j + 1] = reversed(new_position[i : j + 1])

        particle.position = new_position
        particle.fitness = self._calculate_tour_distance(new_position)

        # Update personal best
        if particle.fitness < particle.pbest_fitness:
            particle.pbest_fitness = particle.fitness
            particle.pbest_position = particle.position.copy()

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
        Solve CETSP using Particle Swarm Optimization.

        Returns:
            CETSPSolution with the best tour found
        """
        start_time = time.time()

        # Initialize swarm
        self._initialize_swarm()

        iterations_without_improvement = 0

        for iteration in range(self.n_iterations):
            # Linearly decreasing inertia weight
            w = self.w_max - (self.w_max - self.w_min) * iteration / self.n_iterations

            old_gbest = self.gbest_fitness

            # Update each particle
            for particle in self.particles:
                self._update_velocity_and_position(particle, w)

            # Update global best
            self._update_global_best()

            # Check for improvement
            if self.gbest_fitness < old_gbest - 1e-6:
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            # Early stopping
            if iterations_without_improvement >= self.early_stop_iterations:
                break

            # Periodic restart of worst particles
            if iteration > 0 and iteration % 30 == 0:
                sorted_particles = sorted(self.particles, key=lambda p: p.fitness, reverse=True)
                replace_count = max(1, self.n_particles // 5)

                for i in range(replace_count):
                    if sorted_particles[i].fitness > self.gbest_fitness * 1.2:
                        idx = self.particles.index(sorted_particles[i])
                        self.particles[idx] = self._create_random_particle()

        # Apply local search to best solution
        if self.use_local_search:
            best_tour, best_dist = self._local_search_2opt(self.gbest_position)
            self.gbest_position = best_tour
            self.gbest_fitness = best_dist

        computation_time = time.time() - start_time

        return self._tour_to_solution(self.gbest_position, computation_time)


class AdaptivePSOCETSP(ParticleSwarmCETSP):
    """
    Adaptive Particle Swarm Optimization with enhanced features.

    Improvements:
    - Adaptive inertia weight based on fitness improvement
    - Constriction factor for better convergence
    - Multiple neighborhood structures
    - Elite preservation
    """

    def __init__(
        self,
        problem: CETSP,
        n_particles: int = 30,
        n_iterations: int = 200,
        seed: int | None = None,
    ):
        super().__init__(
            problem=problem,
            n_particles=n_particles,
            n_iterations=n_iterations,
            w_max=0.9,
            w_min=0.4,
            c1=2.05,
            c2=2.05,
            use_local_search=True,
            seed=seed,
        )

        # Constriction factor (Clerc's constriction)
        phi = self.c1 + self.c2
        self.chi = 2.0 / abs(2 - phi - np.sqrt(phi * phi - 4 * phi))

        # Elite particles
        self.elite_size = max(2, n_particles // 10)
        self.elites: list[Particle] = []

    def _update_elites(self) -> None:
        """Update elite particles."""
        all_particles = self.particles + self.elites
        sorted_particles = sorted(all_particles, key=lambda p: p.fitness)
        self.elites = [p.copy() for p in sorted_particles[: self.elite_size]]

    def _or_opt_move(self, tour: list[int]) -> list[int]:
        """Apply Or-opt move (relocate segment)."""
        if len(tour) < 3:
            return tour.copy()

        result = tour.copy()
        seg_len = random.randint(1, min(3, len(tour) - 1))
        start = random.randrange(len(tour) - seg_len + 1)
        segment = result[start : start + seg_len]
        del result[start : start + seg_len]
        insert_pos = random.randrange(len(result) + 1)
        result[insert_pos:insert_pos] = segment

        return result

    def _update_velocity_and_position(self, particle: Particle, w: float) -> None:
        """Enhanced velocity and position update with constriction."""
        # Get swap sequences
        pbest_swaps = self._get_swap_sequence(particle.position, particle.pbest_position)
        gbest_swaps = self._get_swap_sequence(particle.position, self.gbest_position)

        new_position = particle.position.copy()

        # Apply with constriction factor
        r1, r2 = random.random(), random.random()

        # Cognitive component
        cognitive_prob = self.chi * self.c1 * r1 / 4.0
        new_position = self._apply_swap_sequence(new_position, pbest_swaps, cognitive_prob)

        # Social component
        social_prob = self.chi * self.c2 * r2 / 4.0
        new_position = self._apply_swap_sequence(new_position, gbest_swaps, social_prob)

        # Elite learning (learn from random elite)
        if self.elites and random.random() < 0.3:
            elite = random.choice(self.elites)
            elite_swaps = self._get_swap_sequence(new_position, elite.position)
            elite_prob = 0.2
            new_position = self._apply_swap_sequence(new_position, elite_swaps, elite_prob)

        # Adaptive perturbation
        if random.random() < w:
            perturbation = random.choice(["swap", "insert", "reverse", "or_opt"])

            if perturbation == "swap" and len(new_position) >= 2:
                i, j = random.sample(range(len(new_position)), 2)
                new_position[i], new_position[j] = new_position[j], new_position[i]

            elif perturbation == "insert" and len(new_position) >= 2:
                i = random.randrange(len(new_position))
                j = random.randrange(len(new_position))
                if i != j:
                    customer = new_position.pop(i)
                    new_position.insert(j, customer)

            elif perturbation == "reverse" and len(new_position) >= 2:
                i, j = sorted(random.sample(range(len(new_position)), 2))
                new_position[i : j + 1] = reversed(new_position[i : j + 1])

            elif perturbation == "or_opt":
                new_position = self._or_opt_move(new_position)

        particle.position = new_position
        particle.fitness = self._calculate_tour_distance(new_position)

        # Update personal best
        if particle.fitness < particle.pbest_fitness:
            particle.pbest_fitness = particle.fitness
            particle.pbest_position = particle.position.copy()

    def solve(self) -> CETSPSolution:
        """Solve with adaptive PSO."""
        start_time = time.time()

        self._initialize_swarm()
        self._update_elites()

        iterations_without_improvement = 0
        recent_improvements = []

        for iteration in range(self.n_iterations):
            # Adaptive inertia weight
            if len(recent_improvements) >= 5:
                improvement_rate = sum(recent_improvements[-5:]) / 5
                if improvement_rate < 0.1:
                    # Increase exploration
                    w = min(self.w_max, self.w_min + 0.3)
                else:
                    # Normal decrease
                    w = self.w_max - (self.w_max - self.w_min) * iteration / self.n_iterations
            else:
                w = self.w_max - (self.w_max - self.w_min) * iteration / self.n_iterations

            old_gbest = self.gbest_fitness

            # Update particles
            for particle in self.particles:
                self._update_velocity_and_position(particle, w)

            # Update global best and elites
            self._update_global_best()
            self._update_elites()

            # Track improvement
            improved = self.gbest_fitness < old_gbest - 1e-6
            recent_improvements.append(1 if improved else 0)
            if len(recent_improvements) > 10:
                recent_improvements.pop(0)

            if improved:
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            if iterations_without_improvement >= self.early_stop_iterations:
                break

            # Restart worst particles with elite guidance
            if iteration > 0 and iteration % 25 == 0:
                sorted_particles = sorted(self.particles, key=lambda p: p.fitness, reverse=True)
                replace_count = max(1, self.n_particles // 4)

                for i in range(replace_count):
                    idx = self.particles.index(sorted_particles[i])
                    if random.random() < 0.5:
                        self.particles[idx] = self._create_random_particle()
                    else:
                        # Initialize near an elite
                        elite = random.choice(self.elites)
                        new_particle = Particle(
                            position=elite.position.copy(),
                            velocity=[0.0] * self.n_customers,
                            fitness=elite.fitness,
                        )
                        # Apply small perturbation
                        new_particle.position = self._or_opt_move(new_particle.position)
                        new_particle.fitness = self._calculate_tour_distance(new_particle.position)
                        new_particle.pbest_position = new_particle.position.copy()
                        new_particle.pbest_fitness = new_particle.fitness
                        self.particles[idx] = new_particle

        # Apply local search
        if self.use_local_search:
            best_tour, best_dist = self._local_search_2opt(self.gbest_position)
            self.gbest_position = best_tour
            self.gbest_fitness = best_dist

        computation_time = time.time() - start_time

        return self._tour_to_solution(self.gbest_position, computation_time)


class DiscretePSOCETSP(CETSPSolver):
    """
    Discrete PSO using position-based representation.

    This variant uses a priority-based encoding where particles
    represent priorities for each customer, and decoding produces
    a tour by sorting customers by priority.
    """

    def __init__(
        self,
        problem: CETSP,
        n_particles: int = 30,
        n_iterations: int = 200,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        use_local_search: bool = True,
        seed: int | None = None,
        early_stop_iterations: int = 50,
    ):
        super().__init__(problem)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.use_local_search = use_local_search
        self.early_stop_iterations = early_stop_iterations

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.n_customers = len(problem.customers)

        # Particles with continuous position vectors
        self.positions: np.ndarray = None
        self.velocities: np.ndarray = None
        self.fitnesses: np.ndarray = None

        # Personal bests
        self.pbest_positions: np.ndarray = None
        self.pbest_fitnesses: np.ndarray = None

        # Global best
        self.gbest_position: np.ndarray = None
        self.gbest_fitness: float = float("inf")

    def _decode_position(self, position: np.ndarray) -> list[int]:
        """Decode continuous position to discrete tour."""
        # Sort indices by position values
        return list(np.argsort(position))

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

    def _evaluate_position(self, position: np.ndarray) -> float:
        """Evaluate fitness of a position."""
        tour = self._decode_position(position)
        return self._calculate_tour_distance(tour)

    def _initialize_swarm(self) -> None:
        """Initialize swarm with random positions."""
        # Random positions in [0, n_customers]
        self.positions = np.random.uniform(
            0, self.n_customers, (self.n_particles, self.n_customers)
        )
        self.velocities = np.random.uniform(-1, 1, (self.n_particles, self.n_customers))

        # Add greedy solution
        greedy_tour = self._create_greedy_tour()
        # Convert tour to position (priorities)
        self.positions[0] = np.argsort(greedy_tour).astype(float)

        # Evaluate all
        self.fitnesses = np.array([self._evaluate_position(pos) for pos in self.positions])

        # Initialize personal bests
        self.pbest_positions = self.positions.copy()
        self.pbest_fitnesses = self.fitnesses.copy()

        # Initialize global best
        best_idx = np.argmin(self.fitnesses)
        self.gbest_position = self.positions[best_idx].copy()
        self.gbest_fitness = self.fitnesses[best_idx]

    def _create_greedy_tour(self) -> list[int]:
        """Create greedy tour."""
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

        return tour

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
        """Solve using discrete PSO."""
        start_time = time.time()

        self._initialize_swarm()

        iterations_without_improvement = 0

        for iteration in range(self.n_iterations):
            # Linearly decreasing inertia
            w = self.w_max - (self.w_max - self.w_min) * iteration / self.n_iterations

            old_gbest = self.gbest_fitness

            for i in range(self.n_particles):
                r1, r2 = np.random.random(self.n_customers), np.random.random(self.n_customers)

                # Update velocity
                self.velocities[i] = (
                    w * self.velocities[i]
                    + self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                    + self.c2 * r2 * (self.gbest_position - self.positions[i])
                )

                # Clamp velocity
                v_max = self.n_customers / 2
                self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)

                # Update position
                self.positions[i] = self.positions[i] + self.velocities[i]

                # Clamp position
                self.positions[i] = np.clip(self.positions[i], 0, self.n_customers)

                # Evaluate
                self.fitnesses[i] = self._evaluate_position(self.positions[i])

                # Update personal best
                if self.fitnesses[i] < self.pbest_fitnesses[i]:
                    self.pbest_fitnesses[i] = self.fitnesses[i]
                    self.pbest_positions[i] = self.positions[i].copy()

                # Update global best
                if self.fitnesses[i] < self.gbest_fitness:
                    self.gbest_fitness = self.fitnesses[i]
                    self.gbest_position = self.positions[i].copy()

            # Check improvement
            if self.gbest_fitness < old_gbest - 1e-6:
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            if iterations_without_improvement >= self.early_stop_iterations:
                break

        # Get best tour
        best_tour = self._decode_position(self.gbest_position)

        # Apply local search
        if self.use_local_search:
            best_tour, _ = self._local_search_2opt(best_tour)

        computation_time = time.time() - start_time

        return self._tour_to_solution(best_tour, computation_time)
