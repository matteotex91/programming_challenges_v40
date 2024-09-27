import numpy as np
import matplotlib as plt


class Point:
    def __init__(
        self,
        pivot: np.ndarray,
        normal: np.ndarray,
        thickness: float,
        rest_displacement: float,
        stiffness: float,
    ):
        self.pivot = pivot
        self.normal = normal
        self.thickness = thickness
        self.stiffness = stiffness
        self.rest_displacement = rest_displacement
        self.force = np.array([0, 0, 0])
        self.force_2 = np.array([0, 0, 0])
        self.momentum = np.array([0, 0, 0])
        self.neighbors = list()

    def compute_forces(self):
        self.force = np.array([0, 0, 0])
        self.force_2 = np.array([0, 0, 0])
        for n in self.neighbors:
            # compute force
            displacement_vec = n.pivot - self.pivot
            displacement_norm = np.linalg.norm(displacement_vec)
            self.force += (
                self.stiffness
                * (displacement_norm - self.rest_displacement)
                * displacement_vec
                / displacement_norm
            )
            # compute momentum in this way :
            # compute the force on the virtual point, far from the real surface,
            # then compute the momentum wrt the normal, times the thickness
            displacement_vec = (n.pivot + n.thickness * n.normal) - (
                self.pivot + self.thickness * self.normal
            )
            displacement_norm = np.linalg.norm(displacement_vec)
            self.force_2 += (
                self.stiffness
                * (displacement_norm - self.rest_displacement)
                * displacement_vec
                / displacement_norm
            )
        self.momentum = np.cross(self.thickness * self.normal, self.force_2)


class Patch:
    def __init__(self): ...


class SquarePatch(Patch):
    def __init__(
        self,
        shape: np.ndarray,
    ):
        super().__init__()
        self.shape = shape
        self.points = np.empty(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.points[i, j] = Point([i, j, 0], [0, 0, 1], 0.1)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i > 0:
                    self.points[i, j].neighbors.append(self.points[i - 1, j])
                elif j > 0:
                    self.points[i, j].neighbors.append(self.points[i, j - 1])
                elif i < shape[0] - 1:
                    self.points[i, j].neighbors.append(self.points[i + 1, j])
                elif j < shape[1] - 1:
                    self.points[i, j].neighbors.append(self.points[i, j + 1])


if __name__ == "__main__":
    ...
