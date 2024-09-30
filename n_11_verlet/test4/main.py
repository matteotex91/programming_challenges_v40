import numpy as np
import matplotlib.pyplot as plt


class Point:
    def __init__(
        self,
        pivot: np.ndarray,
        normal: np.ndarray,
        thickness: float,
    ):
        self.pivot = pivot
        self.normal = normal
        self.thickness = thickness
        self.force = np.array([0, 0, 0])
        self.force_bend = np.array([0, 0, 0])
        self.neighbors = list()
        self.rest_distances = list()
        self.rest_distances_bend = list()
        self.stiffnesses = list()

    def compute_forces(self):
        self.force = np.array([0, 0, 0])
        self.force_bend = np.array([0, 0, 0])
        for n, d, d2, k in zip(
            self.neighbors,
            self.rest_distances,
            self.rest_distances_bend,
            self.stiffnesses,
        ):
            # compute force
            displacement_vec = n.pivot - self.pivot
            displacement_norm = np.linalg.norm(displacement_vec)
            self.force = self.force + (
                k * (displacement_norm - d) * displacement_vec / displacement_norm
            )
            # compute bending force, in the parallel surface
            displacement_vec = (n.pivot + n.thickness * n.normal) - (
                self.pivot + self.thickness * self.normal
            )
            displacement_norm = np.linalg.norm(displacement_vec)
            self.force_bend = self.force_bend + (
                k * (displacement_norm - d2) * displacement_vec / displacement_norm
            )

    def apply_forces(self, rate):
        self.pivot = self.pivot + rate * self.force
        self.normal = self.normal + rate * self.force_bend
        self.normal = self.normal / np.linalg.norm(self.normal)

    def add_neighbor(self, n, d, d2, k):
        self.neighbors.append(n)
        self.rest_distances.append(d)
        self.rest_distances_bend.append(d2)
        self.stiffnesses.append(k)


class Patch:
    def __init__(self): ...


class SquarePatch(Patch):
    def __init__(self, shape: np.ndarray, thickness: float = 0.5, stiffness: float = 1):
        super().__init__()
        self.shape = shape
        self.points = np.empty(shape, dtype=np.object_)
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.points[i, j] = Point(
                    np.array([i, j, 0]), np.array([0, 0, 1]), thickness
                )
        spacing = 1
        spacing2 = 2
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i > 0:
                    self.points[i, j].add_neighbor(
                        self.points[i - 1, j], spacing, spacing2, stiffness
                    )
                if j > 0:
                    self.points[i, j].add_neighbor(
                        self.points[i, j - 1], spacing, spacing2, stiffness
                    )
                if i < shape[0] - 1:
                    self.points[i, j].add_neighbor(
                        self.points[i + 1, j], spacing, spacing2, stiffness
                    )
                if j < shape[1] - 1:
                    self.points[i, j].add_neighbor(
                        self.points[i, j + 1], spacing, spacing2, stiffness
                    )

    def get_z_positions(self):
        return np.array([[point.pivot[2] for point in row] for row in self.points])

    def relaxation_step(self, rate):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.points[i, j].compute_forces()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.points[i, j].apply_forces(rate)


if __name__ == "__main__":
    p = SquarePatch((5, 5))
    p.points[2, 2].pivot = np.array([2, 2, 1])
    print(p.get_z_positions())
    for i in range(1000):
        if i % 50 == 0:
            plt.pcolormesh(p.get_z_positions())
            plt.show()
        p.relaxation_step(0.5)
    print(p.get_z_positions())
