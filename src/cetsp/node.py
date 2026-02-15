"""Node class for Close-Enough TSP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CETSPNode:
    """
    Represents a node in the Close-Enough TSP.

    Unlike standard TSP, the salesman doesn't need to visit the exact
    location but just needs to get within the coverage radius.

    Attributes:
        id: Unique identifier for the node.
        x: X coordinate in 3D space.
        y: Y coordinate in 3D space.
        z: Z coordinate in 3D space.
        radius: Coverage radius - salesman needs to get within this distance.
        priority: Optional priority weight for this node.
    """

    id: int
    x: float
    y: float
    z: float
    radius: float = 0.0  # Coverage radius (0 means exact visit required)
    priority: float = 1.0  # Priority weight

    @property
    def position(self) -> np.ndarray:
        """Return position as numpy array."""
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other: CETSPNode) -> float:
        """Calculate Euclidean distance to another node center."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate distance from node center to a point."""
        return np.linalg.norm(self.position - point)

    def is_covered_from(self, point: np.ndarray) -> bool:
        """Check if this node is covered (close enough) from a given point."""
        return self.distance_to_point(point) <= self.radius

    def closest_coverage_point(self, from_point: np.ndarray) -> np.ndarray:
        """
        Find the closest point that covers this node from a given point.

        If from_point is already within coverage, returns from_point.
        Otherwise, returns the point on the coverage sphere closest to from_point.
        """
        dist = self.distance_to_point(from_point)

        if dist <= self.radius:
            return from_point

        if dist == 0:
            # from_point is at node center, return any point on sphere
            return self.position + np.array([self.radius, 0, 0])

        # Direction from node to from_point
        direction = (from_point - self.position) / dist

        # Point on sphere closest to from_point
        return self.position + direction * self.radius

    def coverage_entry_point(self, from_point: np.ndarray) -> np.ndarray:
        """
        Find the optimal entry point into the coverage area from a given point.

        This is the point on the coverage boundary that minimizes travel distance.
        """
        dist = self.distance_to_point(from_point)

        if dist <= self.radius:
            # Already inside coverage
            return from_point

        # Direction from from_point to node center
        direction = (self.position - from_point) / dist

        # Entry point on the coverage boundary
        return self.position - direction * self.radius

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CETSPNode):
            return False
        return self.id == other.id
