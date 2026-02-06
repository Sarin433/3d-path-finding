"""Test cases for Close-Enough Travelling Salesman Problem."""

import pytest
import numpy as np

from src.cetsp import CETSPNode, CETSP, GreedyCETSP, AStarCETSP


class TestCETSPNode:
    """Test cases for CETSPNode class."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = CETSPNode(id=1, x=10.0, y=20.0, z=5.0, radius=3.0)
        assert node.id == 1
        assert node.x == 10.0
        assert node.y == 20.0
        assert node.z == 5.0
        assert node.radius == 3.0
    
    def test_node_position(self):
        """Test position property."""
        node = CETSPNode(id=1, x=1.0, y=2.0, z=3.0)
        pos = node.position
        assert np.allclose(pos, [1.0, 2.0, 3.0])
    
    def test_distance_to(self):
        """Test distance calculation between nodes."""
        node1 = CETSPNode(id=1, x=0, y=0, z=0)
        node2 = CETSPNode(id=2, x=3, y=4, z=0)
        assert node1.distance_to(node2) == 5.0
    
    def test_distance_to_point(self):
        """Test distance to arbitrary point."""
        node = CETSPNode(id=1, x=0, y=0, z=0)
        point = np.array([3.0, 4.0, 0.0])
        assert node.distance_to_point(point) == 5.0
    
    def test_is_covered_from_inside(self):
        """Test coverage check when point is inside radius."""
        node = CETSPNode(id=1, x=10, y=10, z=10, radius=5)
        point = np.array([12.0, 10.0, 10.0])  # 2 units away
        assert node.is_covered_from(point)
    
    def test_is_covered_from_outside(self):
        """Test coverage check when point is outside radius."""
        node = CETSPNode(id=1, x=10, y=10, z=10, radius=5)
        point = np.array([20.0, 10.0, 10.0])  # 10 units away
        assert not node.is_covered_from(point)
    
    def test_is_covered_from_boundary(self):
        """Test coverage check when point is exactly on boundary."""
        node = CETSPNode(id=1, x=0, y=0, z=0, radius=5)
        point = np.array([5.0, 0.0, 0.0])
        assert node.is_covered_from(point)
    
    def test_coverage_entry_point_from_outside(self):
        """Test finding entry point when outside coverage."""
        node = CETSPNode(id=1, x=10, y=0, z=0, radius=3)
        from_point = np.array([0.0, 0.0, 0.0])
        
        entry = node.coverage_entry_point(from_point)
        
        # Entry should be on the boundary, closest to from_point
        dist_to_center = np.linalg.norm(entry - node.position)
        assert np.isclose(dist_to_center, node.radius)
        
        # Entry should be between from_point and node center
        assert entry[0] == 7.0  # 10 - 3
    
    def test_coverage_entry_point_from_inside(self):
        """Test entry point when already inside coverage."""
        node = CETSPNode(id=1, x=10, y=0, z=0, radius=5)
        from_point = np.array([8.0, 0.0, 0.0])  # Inside coverage
        
        entry = node.coverage_entry_point(from_point)
        
        # Should return the same point
        assert np.allclose(entry, from_point)
    
    def test_closest_coverage_point(self):
        """Test finding closest point within coverage."""
        node = CETSPNode(id=1, x=10, y=0, z=0, radius=3)
        from_point = np.array([0.0, 0.0, 0.0])
        
        closest = node.closest_coverage_point(from_point)
        
        # Should be on the sphere surface facing from_point
        dist = np.linalg.norm(closest - node.position)
        assert np.isclose(dist, node.radius)


class TestCETSP:
    """Test cases for CETSP problem class."""
    
    @pytest.fixture
    def simple_problem(self):
        """Create simple CETSP instance."""
        depot = CETSPNode(id=0, x=0, y=0, z=0, radius=0)
        customers = [
            CETSPNode(id=1, x=10, y=0, z=0, radius=2),
            CETSPNode(id=2, x=10, y=10, z=0, radius=2),
            CETSPNode(id=3, x=0, y=10, z=0, radius=2),
        ]
        return CETSP(depot, customers)
    
    @pytest.fixture
    def problem_3d(self):
        """Create 3D CETSP instance."""
        depot = CETSPNode(id=0, x=0, y=0, z=0, radius=0)
        customers = [
            CETSPNode(id=1, x=10, y=0, z=5, radius=3),
            CETSPNode(id=2, x=10, y=10, z=10, radius=3),
            CETSPNode(id=3, x=0, y=10, z=15, radius=3),
        ]
        return CETSP(depot, customers)
    
    def test_initialization(self, simple_problem):
        """Test CETSP initialization."""
        assert simple_problem.depot.id == 0
        assert len(simple_problem.customers) == 3
        assert simple_problem.return_to_depot
    
    def test_distance_matrix(self, simple_problem):
        """Test distance matrix computation."""
        # Depot to customer 1: 10 units
        assert simple_problem.distance_matrix[0, 1] == 10.0
        # Customer 1 to customer 2: 10 units
        assert simple_problem.distance_matrix[1, 2] == 10.0
    
    def test_effective_distance(self, simple_problem):
        """Test effective distance considering radii."""
        c1 = simple_problem.customers[0]  # radius=2
        c2 = simple_problem.customers[1]  # radius=2
        
        center_dist = simple_problem.get_center_distance(c1, c2)
        effective_dist = simple_problem.get_effective_distance(c1, c2)
        
        # Effective should be center - both radii
        assert effective_dist == center_dist - 4.0
    
    def test_effective_distance_overlapping(self):
        """Test effective distance when coverages overlap."""
        depot = CETSPNode(id=0, x=0, y=0, z=0)
        customers = [
            CETSPNode(id=1, x=5, y=0, z=0, radius=3),
            CETSPNode(id=2, x=8, y=0, z=0, radius=3),  # Overlaps with c1
        ]
        problem = CETSP(depot, customers)
        
        eff_dist = problem.get_effective_distance(customers[0], customers[1])
        assert eff_dist == 0.0  # Coverages overlap
    
    def test_solve_greedy(self, simple_problem):
        """Test greedy solver."""
        solution = simple_problem.solve(method="greedy")
        
        assert solution.is_complete
        assert solution.total_distance > 0
        assert solution.path.num_waypoints >= 4  # depot + 3 customers + return
    
    def test_solve_astar(self, simple_problem):
        """Test A* solver."""
        solution = simple_problem.solve(method="astar")
        
        assert solution.is_complete
        assert solution.total_distance > 0
    
    def test_solve_3d(self, problem_3d):
        """Test solving 3D problem."""
        solution = problem_3d.solve(method="greedy")
        
        assert solution.is_complete
        # All waypoints should be 3D
        for wp in solution.path.waypoints:
            assert len(wp) == 3
    
    def test_valid_solution(self, simple_problem):
        """Test solution validation."""
        solution = simple_problem.solve(method="greedy")
        assert simple_problem.is_valid_solution(solution)
    
    def test_path_starts_at_depot(self, simple_problem):
        """Test that path starts at depot."""
        solution = simple_problem.solve(method="greedy")
        
        depot_pos = simple_problem.depot.position
        first_waypoint = solution.path.waypoints[0]
        
        assert np.allclose(first_waypoint, depot_pos)
    
    def test_path_ends_at_depot(self, simple_problem):
        """Test that path returns to depot."""
        solution = simple_problem.solve(method="greedy")
        
        depot_pos = simple_problem.depot.position
        last_waypoint = solution.path.waypoints[-1]
        
        assert np.allclose(last_waypoint, depot_pos)
    
    def test_no_return_to_depot(self):
        """Test problem without returning to depot."""
        depot = CETSPNode(id=0, x=0, y=0, z=0)
        customers = [
            CETSPNode(id=1, x=10, y=0, z=0, radius=2),
            CETSPNode(id=2, x=20, y=0, z=0, radius=2),
        ]
        problem = CETSP(depot, customers, return_to_depot=False)
        
        solution = problem.solve(method="greedy")
        
        # Last waypoint should NOT be at depot
        depot_pos = depot.position
        last_waypoint = solution.path.waypoints[-1]
        
        assert not np.allclose(last_waypoint, depot_pos)


class TestGreedyCETSP:
    """Test cases for Greedy CETSP solver."""
    
    def test_greedy_visits_nearest(self):
        """Test that greedy picks nearest customer first."""
        depot = CETSPNode(id=0, x=0, y=0, z=0)
        customers = [
            CETSPNode(id=1, x=100, y=0, z=0, radius=2),  # Far
            CETSPNode(id=2, x=5, y=0, z=0, radius=2),    # Near
        ]
        problem = CETSP(depot, customers)
        solver = GreedyCETSP(problem)
        
        solution = solver.solve()
        
        # Second waypoint should be near customer 2
        second_wp = solution.path.waypoints[1]
        assert second_wp[0] < 10  # Should be close to customer 2
    
    def test_greedy_computation_time(self):
        """Test that computation time is recorded."""
        depot = CETSPNode(id=0, x=0, y=0, z=0)
        customers = [
            CETSPNode(id=i, x=i*10, y=0, z=0, radius=2)
            for i in range(1, 6)
        ]
        problem = CETSP(depot, customers)
        
        solution = problem.solve(method="greedy")
        
        assert solution.computation_time >= 0


class TestAStarCETSP:
    """Test cases for A* CETSP solver."""
    
    def test_astar_finds_optimal_simple(self):
        """Test A* finds optimal for simple case."""
        depot = CETSPNode(id=0, x=0, y=0, z=0)
        customers = [
            CETSPNode(id=1, x=10, y=0, z=0, radius=0),  # No radius
            CETSPNode(id=2, x=20, y=0, z=0, radius=0),
        ]
        problem = CETSP(depot, customers)
        
        solution = problem.solve(method="astar")
        
        # Optimal: 0 -> 10 -> 20 -> 0 = 40
        assert np.isclose(solution.total_distance, 40.0)
    
    def test_astar_uses_coverage(self):
        """Test A* takes advantage of coverage radii."""
        depot = CETSPNode(id=0, x=0, y=0, z=0)
        customers = [
            CETSPNode(id=1, x=10, y=0, z=0, radius=3),
        ]
        problem = CETSP(depot, customers)
        
        solution = problem.solve(method="astar")
        
        # Should only travel to edge of coverage: (10-3)*2 = 14
        assert solution.total_distance < 20  # Less than visiting center
    
    def test_astar_vs_greedy(self):
        """Test A* produces equal or better solution than greedy."""
        depot = CETSPNode(id=0, x=50, y=50, z=0)
        np.random.seed(42)
        customers = [
            CETSPNode(
                id=i,
                x=np.random.uniform(0, 100),
                y=np.random.uniform(0, 100),
                z=0,
                radius=np.random.uniform(2, 5)
            )
            for i in range(1, 6)
        ]
        problem = CETSP(depot, customers)
        
        greedy_sol = problem.solve(method="greedy")
        astar_sol = problem.solve(method="astar")
        
        # A* should be equal or better
        assert astar_sol.total_distance <= greedy_sol.total_distance * 1.01


class TestCETSPPath:
    """Test cases for CETSP path class."""
    
    def test_empty_path(self):
        """Test empty path properties."""
        from src.cetsp.problem import CETSPPath
        
        path = CETSPPath()
        assert path.total_distance == 0.0
        assert path.num_waypoints == 0
    
    def test_path_distance(self):
        """Test path distance calculation."""
        from src.cetsp.problem import CETSPPath
        
        path = CETSPPath()
        path.add_waypoint(np.array([0.0, 0.0, 0.0]))
        path.add_waypoint(np.array([3.0, 4.0, 0.0]))
        path.add_waypoint(np.array([3.0, 4.0, 5.0]))
        
        assert path.total_distance == 10.0  # 5 + 5


class TestCETSPVisualization:
    """Test cases for CETSP visualization."""
    
    def test_visualize_problem(self):
        """Test visualization without solution."""
        depot = CETSPNode(id=0, x=0, y=0, z=0)
        customers = [
            CETSPNode(id=1, x=10, y=0, z=0, radius=3),
        ]
        problem = CETSP(depot, customers)
        
        fig = problem.visualize()
        
        assert fig is not None
        assert len(fig.data) >= 2  # Depot + customer
    
    def test_visualize_with_solution(self):
        """Test visualization with solution."""
        depot = CETSPNode(id=0, x=0, y=0, z=0)
        customers = [
            CETSPNode(id=1, x=10, y=0, z=0, radius=3),
            CETSPNode(id=2, x=10, y=10, z=0, radius=3),
        ]
        problem = CETSP(depot, customers)
        solution = problem.solve()
        
        fig = problem.visualize(solution)
        
        assert fig is not None
        # Should have depot, customers, coverage spheres, and path
        assert len(fig.data) >= 4


class TestCETSPEdgeCases:
    """Edge case tests for CETSP."""
    
    def test_single_customer(self):
        """Test with single customer."""
        depot = CETSPNode(id=0, x=0, y=0, z=0)
        customers = [CETSPNode(id=1, x=10, y=0, z=0, radius=2)]
        problem = CETSP(depot, customers)
        
        solution = problem.solve()
        
        assert solution.is_complete
        assert len(solution.path.covered_nodes) >= 1
    
    def test_customer_at_depot(self):
        """Test customer at depot location."""
        depot = CETSPNode(id=0, x=0, y=0, z=0)
        customers = [
            CETSPNode(id=1, x=0, y=0, z=0, radius=5),  # At depot
            CETSPNode(id=2, x=10, y=0, z=0, radius=2),
        ]
        problem = CETSP(depot, customers)
        
        solution = problem.solve()
        
        assert solution.is_complete
    
    def test_large_coverage_radius(self):
        """Test with large coverage radius covering multiple points."""
        depot = CETSPNode(id=0, x=0, y=0, z=0)
        customers = [
            CETSPNode(id=1, x=5, y=0, z=0, radius=10),  # Covers depot area
        ]
        problem = CETSP(depot, customers)
        
        solution = problem.solve()
        
        # Should have very short path
        assert solution.total_distance < 10
    
    def test_zero_radius(self):
        """Test with zero radius (standard TSP)."""
        depot = CETSPNode(id=0, x=0, y=0, z=0, radius=0)
        customers = [
            CETSPNode(id=1, x=10, y=0, z=0, radius=0),
            CETSPNode(id=2, x=10, y=10, z=0, radius=0),
        ]
        problem = CETSP(depot, customers)
        
        solution = problem.solve()
        
        # Should visit exact locations
        waypoints = solution.path.waypoints
        visited_positions = [tuple(np.round(w, 2)) for w in waypoints]
        
        assert (10.0, 0.0, 0.0) in visited_positions
        assert (10.0, 10.0, 0.0) in visited_positions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
