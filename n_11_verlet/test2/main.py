import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Point:
    def __init__(
        self,
        pivot: np.ndarray,
        parallel: np.ndarray,
        rest_distance_norm: float,
        norm_stiffness: float,
    ):
        self.pivot = pivot
        self.parallel = parallel
        self.rest_distance_norm = rest_distance_norm
        self.norm_stiffness = norm_stiffness
        self.force = np.array([0, 0, 0])
        self.force_par = np.array([0, 0, 0])
        self.neighbors = list()
        self.rest_distances = list()
        self.rest_distances_par = list()
        self.rest_distances_cross_12 = list()
        self.rest_distances_cross_21 = list()
        self.stiffnesses = list()

    def add_neighbor(
        self,
        point,
        rest_distance,
        rest_distance_par,
        rest_dist_cross_12,
        rest_dist_cross_21,
        stiffness,
    ):
        self.neighbors.append(point)
        self.rest_distances.append(rest_distance)
        self.rest_distances_par.append(rest_distance_par)
        self.rest_distances_cross_12.append(rest_dist_cross_12)
        self.rest_distances_cross_21.append(rest_dist_cross_21)
        self.stiffnesses.append(stiffness)

    def compute_forces(self):
        self.force = np.array([0, 0, 0])
        self.force_par = np.array([0, 0, 0])
        for n, d0, d0_par, d0_cross_12, d0_cross_21, k in zip(
            self.neighbors,
            self.rest_distances,
            self.rest_distances_par,
            self.rest_distances_cross_12,
            self.rest_distances_cross_21,
            self.stiffnesses,
        ):
            # compute force
            displacement_vec = n.pivot - self.pivot
            displacement_norm = np.linalg.norm(displacement_vec)
            self.force = (
                self.force
                + k * (displacement_norm - d0) * displacement_vec / displacement_norm
            )
            displacement_vec = n.parallel - self.pivot
            displacement_norm = np.linalg.norm(displacement_vec)
            self.force = (
                self.force
                + k
                * (displacement_norm - d0_cross_12)
                * displacement_vec
                / displacement_norm
            )
            # compute force on the parallel point
            displacement_vec = n.parallel - self.parallel
            displacement_norm = np.linalg.norm(displacement_vec)
            self.force_par = (
                self.force
                + k
                * (displacement_norm - d0_par)
                * displacement_vec
                / displacement_norm
            )
            displacement_vec = n.pivot - self.parallel
            displacement_norm = np.linalg.norm(displacement_vec)
            self.force_par = (
                self.force
                + k
                * (displacement_norm - d0_cross_21)
                * displacement_vec
                / displacement_norm
            )
        displacement_vec = self.parallel - self.pivot
        displacement_norm = np.linalg.norm(displacement_vec)
        force_norm = (
            self.norm_stiffness
            * (displacement_norm - self.rest_distance_norm)
            * displacement_vec
            / displacement_norm
        )
        self.force = self.force + force_norm
        self.force_par = self.force_par - force_norm

    def apply_forces(self, rate: float):
        self.pivot = self.pivot + self.force * rate
        self.parallel = self.parallel + self.force_par * rate


class Patch:
    def __init__(self): ...


class SquarePatch(Patch):
    def __init__(
        self,
        shape: np.ndarray,
    ):
        super().__init__()
        self.shape = shape
        self.points = np.empty(shape, dtype=np.object_)
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.points[i, j] = Point(
                    np.array([i, j, 0]), np.array([i, j, 0.1]), 0.1, 1
                )
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i > 0:
                    self.points[i, j].add_neighbor(
                        self.points[i - 1, j],
                        1,
                        1,
                        np.sqrt(1.01),
                        np.sqrt(1.01),
                        1,
                    )
                if i < self.shape[0] - 1:
                    self.points[i, j].add_neighbor(
                        self.points[i + 1, j],
                        1,
                        1,
                        np.sqrt(1.01),
                        np.sqrt(1.01),
                        1,
                    )
                if j > 0:
                    self.points[i, j].add_neighbor(
                        self.points[i, j - 1],
                        1,
                        1,
                        np.sqrt(1.01),
                        np.sqrt(1.01),
                        1,
                    )
                if j < self.shape[1] - 1:
                    self.points[i, j].add_neighbor(
                        self.points[i, j + 1],
                        1,
                        1,
                        np.sqrt(1.01),
                        np.sqrt(1.01),
                        1,
                    )
                if i > 0 and j > 0:
                    self.points[i, j].add_neighbor(
                        self.points[i - 1, j - 1],
                        np.sqrt(2),
                        np.sqrt(2),
                        np.sqrt(2.01),
                        np.sqrt(2.01),
                        1,
                    )
                if i < self.shape[0] - 1 and j > 0:
                    self.points[i, j].add_neighbor(
                        self.points[i + 1, j - 1],
                        np.sqrt(2),
                        np.sqrt(2),
                        np.sqrt(2.01),
                        np.sqrt(2.01),
                        1,
                    )
                if i > 0 and j < shape[0] - 1:
                    self.points[i, j].add_neighbor(
                        self.points[i - 1, j + 1],
                        np.sqrt(2),
                        np.sqrt(2),
                        np.sqrt(2.01),
                        np.sqrt(2.01),
                        1,
                    )
                if i < self.shape[0] - 1 and j < shape[0] - 1:
                    self.points[i, j].add_neighbor(
                        self.points[i + 1, j + 1],
                        np.sqrt(2),
                        np.sqrt(2),
                        np.sqrt(2.01),
                        np.sqrt(2.01),
                        1,
                    )

    def relaxation_step(self, rate: float = 0.01):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.points[i, j].compute_forces()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.points[i, j].apply_forces(rate)

    def get_z_positions(self):
        return np.array([[p.pivot[2] for p in row] for row in self.points])


if __name__ == "__main__":

    p = SquarePatch((5, 5))
    p.points[2, 2].pivot[2] = 1
    fig, ax = plt.subplots()
    pc = plt.pcolormesh(p.get_z_positions())

    def update_animation(itrn):
        p.relaxation_step(rate=0.1)
        pc.set_array(p.get_z_positions())
        pc.
        ax.set_title(itrn)
        print(itrn)
        return pc

    anim = animation.FuncAnimation(
        fig, update_animation, frames=1000, interval=1, blit=False, repeat=False
    )
    plt.show()
