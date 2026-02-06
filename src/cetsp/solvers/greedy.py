"""Greedy nearest-neighbor solver for CETSP."""

import time
import numpy as np

from ..problem import CETSP, CETSPPath, CETSPSolution
from .base import CETSPSolver


class GreedyCETSP(CETSPSolver):
    """
    Greedy nearest-neighbor solver for CETSP.
    
    At each step, selects the customer whose coverage area is
    closest to the current position.
    """
    
    def solve(self) -> CETSPSolution:
        """Solve using greedy nearest neighbor approach."""
        start_time = time.time()
        
        path = CETSPPath()
        current_pos = self.problem.depot.position.copy()
        path.add_waypoint(current_pos)
        
        unvisited = set(self.problem.customers)
        
        while unvisited:
            # Find nearest unvisited customer
            nearest = None
            nearest_dist = float("inf")
            nearest_entry = None
            
            for customer in unvisited:
                entry_point = customer.coverage_entry_point(current_pos)
                dist = np.linalg.norm(entry_point - current_pos)
                
                if dist < nearest_dist:
                    nearest = customer
                    nearest_dist = dist
                    nearest_entry = entry_point
            
            if nearest is None:
                break
            
            # Move to coverage entry point
            path.add_waypoint(nearest_entry, nearest)
            current_pos = nearest_entry
            unvisited.remove(nearest)
        
        # Return to depot if required
        if self.problem.return_to_depot:
            path.add_waypoint(self.problem.depot.position.copy())
        
        computation_time = time.time() - start_time
        
        return CETSPSolution(
            path=path,
            is_complete=len(unvisited) == 0,
            computation_time=computation_time
        )
