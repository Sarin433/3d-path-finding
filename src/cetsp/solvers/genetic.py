"""Genetic Algorithm solver for CETSP."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from ..problem import CETSP, CETSPPath, CETSPSolution
from .base import CETSPSolver


@dataclass
class Individual:
    """Represents an individual (chromosome) in the genetic algorithm."""

    tour: list[int]  # List of customer indices (not including depot)
    fitness: float = float("inf")

    def __lt__(self, other: Individual) -> bool:
        return self.fitness < other.fitness

    def copy(self) -> Individual:
        return Individual(tour=self.tour.copy(), fitness=self.fitness)


class GeneticCETSP(CETSPSolver):
    """
    Genetic Algorithm solver for Close-Enough TSP.

    Uses evolutionary optimization to find near-optimal tours:
    - Population-based search
    - Selection, crossover, and mutation operators
    - Elitism to preserve best solutions

    Parameters:
        population_size: Number of individuals in the population
        generations: Maximum number of generations
        crossover_rate: Probability of crossover (0.0-1.0)
        mutation_rate: Probability of mutation (0.0-1.0)
        elitism_count: Number of best individuals to preserve
        tournament_size: Size of tournament for selection
        seed: Random seed for reproducibility

    Example:
        >>> solver = GeneticCETSP(problem, population_size=100, generations=500)
        >>> solution = solver.solve()
    """

    def __init__(
        self,
        problem: CETSP,
        population_size: int = 100,
        generations: int = 500,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        elitism_count: int = 5,
        tournament_size: int = 5,
        seed: int | None = None,
        early_stop_generations: int = 100,
    ):
        super().__init__(problem)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.early_stop_generations = early_stop_generations

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.n_customers = len(problem.customers)
        self._precompute_distances()

    def _precompute_distances(self) -> None:
        """Precompute effective distances between all nodes."""
        n = self.n_customers + 1  # +1 for depot
        self.eff_distances = np.zeros((n, n))

        all_nodes = [self.problem.depot] + list(self.problem.customers)

        for i in range(n):
            for j in range(n):
                if i != j:
                    self.eff_distances[i, j] = self.problem.get_effective_distance(
                        all_nodes[i], all_nodes[j]
                    )

    def _calculate_tour_distance(self, tour: list[int]) -> float:
        """
        Calculate the actual CETSP tour distance with optimal waypoints.

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

    def _create_random_individual(self) -> Individual:
        """Create a random individual (random tour permutation)."""
        tour = list(range(self.n_customers))
        random.shuffle(tour)
        ind = Individual(tour=tour)
        ind.fitness = self._calculate_tour_distance(tour)
        return ind

    def _create_greedy_individual(self) -> Individual:
        """Create an individual using greedy nearest neighbor."""
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

        ind = Individual(tour=tour)
        ind.fitness = self._calculate_tour_distance(tour)
        return ind

    def _create_nearest_insertion_individual(self) -> Individual:
        """Create an individual using nearest insertion heuristic."""
        if self.n_customers < 2:
            return self._create_greedy_individual()

        # Start with depot and farthest customer
        depot_pos = self.problem.depot.position
        farthest_idx = max(
            range(self.n_customers),
            key=lambda i: np.linalg.norm(self.problem.customers[i].position - depot_pos),
        )

        tour = [farthest_idx]
        in_tour = {farthest_idx}

        while len(tour) < self.n_customers:
            # Find customer closest to any in tour
            best_idx = None
            best_dist = float("inf")

            for idx in range(self.n_customers):
                if idx in in_tour:
                    continue
                cust = self.problem.customers[idx]
                for tour_idx in tour:
                    tour_cust = self.problem.customers[tour_idx]
                    dist = self.problem.get_effective_distance(cust, tour_cust)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx

            # Find best position to insert
            best_pos = 0
            best_increase = float("inf")

            for pos in range(len(tour) + 1):
                # Calculate increase in distance
                prev_idx = tour[pos - 1] if pos > 0 else None
                next_idx = tour[pos] if pos < len(tour) else None

                increase = 0
                new_cust = self.problem.customers[best_idx]

                if prev_idx is not None:
                    prev_cust = self.problem.customers[prev_idx]
                    increase += self.problem.get_effective_distance(prev_cust, new_cust)
                else:
                    increase += self.problem.get_effective_distance(self.problem.depot, new_cust)

                if next_idx is not None:
                    next_cust = self.problem.customers[next_idx]
                    increase += self.problem.get_effective_distance(new_cust, next_cust)
                    if prev_idx is not None:
                        increase -= self.problem.get_effective_distance(
                            self.problem.customers[prev_idx], next_cust
                        )

                if increase < best_increase:
                    best_increase = increase
                    best_pos = pos

            tour.insert(best_pos, best_idx)
            in_tour.add(best_idx)

        ind = Individual(tour=tour)
        ind.fitness = self._calculate_tour_distance(tour)
        return ind

    def _initialize_population(self) -> list[Individual]:
        """Initialize population with diverse solutions."""
        population = []

        # Add greedy solution
        population.append(self._create_greedy_individual())

        # Add nearest insertion solution
        population.append(self._create_nearest_insertion_individual())

        # Fill rest with random individuals
        while len(population) < self.population_size:
            population.append(self._create_random_individual())

        return population

    def _tournament_select(self, population: list[Individual]) -> Individual:
        """Select an individual using tournament selection."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return min(tournament).copy()

    def _order_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual]:
        """
        Order Crossover (OX) operator.

        Preserves relative order of customers from both parents.
        """
        size = len(parent1.tour)

        # Select two crossover points
        start, end = sorted(random.sample(range(size), 2))

        # Create offspring
        def create_child(p1: list[int], p2: list[int]) -> list[int]:
            child = [None] * size

            # Copy segment from parent1
            child[start : end + 1] = p1[start : end + 1]

            # Fill remaining from parent2 in order
            p2_idx = (end + 1) % size
            child_idx = (end + 1) % size

            while None in child:
                if p2[p2_idx] not in child:
                    child[child_idx] = p2[p2_idx]
                    child_idx = (child_idx + 1) % size
                p2_idx = (p2_idx + 1) % size

            return child

        child1_tour = create_child(parent1.tour, parent2.tour)
        child2_tour = create_child(parent2.tour, parent1.tour)

        child1 = Individual(tour=child1_tour)
        child2 = Individual(tour=child2_tour)

        child1.fitness = self._calculate_tour_distance(child1_tour)
        child2.fitness = self._calculate_tour_distance(child2_tour)

        return child1, child2

    def _pmx_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual]:
        """
        Partially Mapped Crossover (PMX) operator.
        """
        size = len(parent1.tour)
        start, end = sorted(random.sample(range(size), 2))

        def create_child(p1: list[int], p2: list[int]) -> list[int]:
            child = [None] * size

            # Copy segment from p1
            child[start : end + 1] = p1[start : end + 1]

            # Map remaining positions
            for i in range(start, end + 1):
                if p2[i] not in child:
                    p2[i]
                    pos = i
                    while start <= pos <= end:
                        p2[p1.index(p2[pos]) if p2[pos] in p1 else pos]
                        pos = p2.index(p1[pos]) if p1[pos] in p2 else pos
                        if pos < start or pos > end:
                            break
                    if child[pos] is None:
                        child[pos] = p2[i]

            # Fill remaining with p2 values
            for i in range(size):
                if child[i] is None:
                    child[i] = p2[i]

            return child

        child1_tour = create_child(parent1.tour, parent2.tour)
        child2_tour = create_child(parent2.tour, parent1.tour)

        child1 = Individual(tour=child1_tour)
        child2 = Individual(tour=child2_tour)

        child1.fitness = self._calculate_tour_distance(child1_tour)
        child2.fitness = self._calculate_tour_distance(child2_tour)

        return child1, child2

    def _swap_mutation(self, individual: Individual) -> Individual:
        """Swap mutation: swap two random positions."""
        tour = individual.tour.copy()
        if len(tour) >= 2:
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]

        mutated = Individual(tour=tour)
        mutated.fitness = self._calculate_tour_distance(tour)
        return mutated

    def _inversion_mutation(self, individual: Individual) -> Individual:
        """Inversion mutation: reverse a random segment."""
        tour = individual.tour.copy()
        if len(tour) >= 2:
            i, j = sorted(random.sample(range(len(tour)), 2))
            tour[i : j + 1] = reversed(tour[i : j + 1])

        mutated = Individual(tour=tour)
        mutated.fitness = self._calculate_tour_distance(tour)
        return mutated

    def _insertion_mutation(self, individual: Individual) -> Individual:
        """Insertion mutation: move a customer to a new position."""
        tour = individual.tour.copy()
        if len(tour) >= 2:
            i = random.randrange(len(tour))
            j = random.randrange(len(tour))
            if i != j:
                customer = tour.pop(i)
                tour.insert(j, customer)

        mutated = Individual(tour=tour)
        mutated.fitness = self._calculate_tour_distance(tour)
        return mutated

    def _mutate(self, individual: Individual) -> Individual:
        """Apply mutation operator (randomly chosen)."""
        mutation_type = random.random()

        if mutation_type < 0.4:
            return self._swap_mutation(individual)
        elif mutation_type < 0.7:
            return self._inversion_mutation(individual)
        else:
            return self._insertion_mutation(individual)

    def _local_search_2opt(self, individual: Individual) -> Individual:
        """Apply 2-opt local search to improve individual."""
        tour = individual.tour.copy()
        improved = True

        while improved:
            improved = False
            best_dist = self._calculate_tour_distance(tour)

            for i in range(len(tour) - 1):
                for j in range(i + 2, len(tour)):
                    # Try reversing segment [i+1, j]
                    new_tour = tour[: i + 1] + tour[i + 1 : j + 1][::-1] + tour[j + 1 :]
                    new_dist = self._calculate_tour_distance(new_tour)

                    if new_dist < best_dist - 1e-6:
                        tour = new_tour
                        best_dist = new_dist
                        improved = True
                        break

                if improved:
                    break

        result = Individual(tour=tour)
        result.fitness = best_dist
        return result

    def _tour_to_solution(self, tour: list[int]) -> CETSPSolution:
        """Convert tour (list of customer indices) to CETSPSolution."""
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

        return CETSPSolution(path=path, is_complete=True, computation_time=0.0)

    def solve(self) -> CETSPSolution:
        """
        Solve CETSP using Genetic Algorithm.

        Returns:
            CETSPSolution with the best tour found
        """
        start_time = time.time()

        # Initialize population
        population = self._initialize_population()

        best_individual = min(population)
        best_fitness_history = [best_individual.fitness]
        generations_without_improvement = 0

        for _generation in range(self.generations):
            # Sort population by fitness
            population.sort()

            # Update best
            if population[0].fitness < best_individual.fitness:
                best_individual = population[0].copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            best_fitness_history.append(best_individual.fitness)

            # Early stopping
            if generations_without_improvement >= self.early_stop_generations:
                break

            # Create new population
            new_population = []

            # Elitism: keep best individuals
            for i in range(self.elitism_count):
                new_population.append(population[i].copy())

            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)

                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population

        # Apply local search to best individual
        best_individual = self._local_search_2opt(best_individual)

        computation_time = time.time() - start_time

        # Convert to solution
        solution = self._tour_to_solution(best_individual.tour)
        solution.computation_time = computation_time

        return solution


class AdaptiveGeneticCETSP(GeneticCETSP):
    """
    Adaptive Genetic Algorithm with self-adjusting parameters.

    Mutation and crossover rates are adjusted based on population diversity.
    """

    def __init__(
        self,
        problem: CETSP,
        population_size: int = 100,
        generations: int = 500,
        seed: int | None = None,
    ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            generations=generations,
            crossover_rate=0.9,
            mutation_rate=0.1,
            elitism_count=5,
            tournament_size=5,
            seed=seed,
        )

        self.min_mutation_rate = 0.05
        self.max_mutation_rate = 0.4
        self.min_crossover_rate = 0.6
        self.max_crossover_rate = 0.95

    def _calculate_diversity(self, population: list[Individual]) -> float:
        """Calculate population diversity based on fitness variance."""
        fitnesses = [ind.fitness for ind in population]
        mean_fitness = np.mean(fitnesses)
        if mean_fitness == 0:
            return 0.0
        std_fitness = np.std(fitnesses)
        return std_fitness / mean_fitness

    def _adapt_parameters(self, diversity: float) -> None:
        """Adapt mutation and crossover rates based on diversity."""
        # Low diversity -> increase mutation, decrease crossover
        # High diversity -> decrease mutation, increase crossover

        if diversity < 0.05:  # Low diversity
            self.mutation_rate = min(self.max_mutation_rate, self.mutation_rate * 1.5)
            self.crossover_rate = max(self.min_crossover_rate, self.crossover_rate * 0.9)
        elif diversity > 0.2:  # High diversity
            self.mutation_rate = max(self.min_mutation_rate, self.mutation_rate * 0.8)
            self.crossover_rate = min(self.max_crossover_rate, self.crossover_rate * 1.1)

    def solve(self) -> CETSPSolution:
        """Solve with adaptive parameter control."""
        start_time = time.time()

        population = self._initialize_population()
        best_individual = min(population)
        generations_without_improvement = 0

        for _generation in range(self.generations):
            population.sort()

            # Adapt parameters
            diversity = self._calculate_diversity(population)
            self._adapt_parameters(diversity)

            # Update best
            if population[0].fitness < best_individual.fitness:
                best_individual = population[0].copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= self.early_stop_generations:
                break

            # Evolution
            new_population = [population[i].copy() for i in range(self.elitism_count)]

            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)

                if random.random() < self.crossover_rate:
                    child1, child2 = self._order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population

        best_individual = self._local_search_2opt(best_individual)

        computation_time = time.time() - start_time
        solution = self._tour_to_solution(best_individual.tour)
        solution.computation_time = computation_time

        return solution
