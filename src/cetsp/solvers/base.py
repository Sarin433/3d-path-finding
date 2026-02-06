"""Base solver class for Close-Enough TSP."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..problem import CETSP, CETSPSolution


class CETSPSolver(ABC):
    """Abstract base class for CETSP solvers."""

    def __init__(self, problem: CETSP):
        self.problem = problem

    @abstractmethod
    def solve(self) -> CETSPSolution:
        """Solve the CETSP and return solution."""
        pass
