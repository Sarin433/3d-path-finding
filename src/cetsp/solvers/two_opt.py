"""2-opt improvement for CETSP solutions."""

import numpy as np

from ..problem import CETSP, CETSPPath, CETSPSolution


class TwoOptCETSP:
    """2-opt improvement for CETSP solutions."""

    def __init__(self, problem: CETSP):
        self.problem = problem

    def improve(self, solution: CETSPSolution) -> CETSPSolution:
        """Apply 2-opt improvement to solution."""
        if solution.path.num_waypoints < 4:
            return solution

        waypoints = [w.copy() for w in solution.path.waypoints]
        covered = list(solution.path.covered_nodes)

        improved = True
        while improved:
            improved = False

            for i in range(1, len(waypoints) - 2):
                for j in range(i + 1, len(waypoints) - 1):
                    # Calculate current distance
                    d1 = np.linalg.norm(waypoints[i] - waypoints[i - 1])
                    d2 = np.linalg.norm(waypoints[j + 1] - waypoints[j])

                    # Calculate new distance if we reverse segment
                    d3 = np.linalg.norm(waypoints[j] - waypoints[i - 1])
                    d4 = np.linalg.norm(waypoints[j + 1] - waypoints[i])

                    if d1 + d2 > d3 + d4:
                        # Reverse segment
                        waypoints[i : j + 1] = waypoints[i : j + 1][::-1]
                        improved = True

        new_path = CETSPPath()
        for i, wp in enumerate(waypoints):
            c = covered[i - 1] if i > 0 and i - 1 < len(covered) else None
            new_path.add_waypoint(wp, c)

        return CETSPSolution(
            path=new_path,
            is_complete=solution.is_complete,
            computation_time=solution.computation_time,
        )
