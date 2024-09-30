import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Point:
    def __init__(
        self,
        pivot: np.ndarray,
    ):
        self.pivot = pivot
        self.force = np.array([0, 0, 0])
        self.neighbors = list()
        self.rest_distances = list()
        self.stiffnesses = list()

    def add_neighbor(
        self,
        point,
        rest_distance,
        stiffness,
    ):
        self.neighbors.append(point)
        self.rest_distances.append(rest_distance)
        self.stiffnesses.append(stiffness)

    def compute_forces(self):
        self.force = np.array([0, 0, 0])
        self.force_par = np.array([0, 0, 0])
        for n, d0, k in zip(
            self.neighbors,
            self.rest_distances,
            self.stiffnesses,
        ):
            # compute force
            displacement_vec = n.pivot - self.pivot
            displacement_norm = np.linalg.norm(displacement_vec)
            self.force = (
                self.force
                + k * (displacement_norm - d0) * displacement_vec / displacement_norm
            )

    def apply_forces(self, rate: float):
        self.pivot = self.pivot + self.force * rate


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
        rest_dist = 1
        diag_rest_dist = np.sqrt(2) * rest_dist
        stiffness = 1
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.points[i, j] = Point(
                    np.array([i * rest_dist, j * rest_dist, 0]),
                )
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i > 0:
                    self.points[i, j].add_neighbor(
                        self.points[i - 1, j],
                        rest_dist,
                        stiffness,
                    )
                if i < self.shape[0] - 1:
                    self.points[i, j].add_neighbor(
                        self.points[i + 1, j],
                        rest_dist,
                        stiffness,
                    )
                if j > 0:
                    self.points[i, j].add_neighbor(
                        self.points[i, j - 1],
                        rest_dist,
                        stiffness,
                    )
                if j < self.shape[1] - 1:
                    self.points[i, j].add_neighbor(
                        self.points[i, j + 1],
                        rest_dist,
                        stiffness,
                    )
                if i > 0 and j > 0:
                    self.points[i, j].add_neighbor(
                        self.points[i - 1, j - 1],
                        diag_rest_dist,
                        stiffness,
                    )
                if i < self.shape[0] - 1 and j > 0:
                    self.points[i, j].add_neighbor(
                        self.points[i + 1, j - 1],
                        diag_rest_dist,
                        stiffness,
                    )
                if i > 0 and j < shape[0] - 1:
                    self.points[i, j].add_neighbor(
                        self.points[i - 1, j + 1],
                        diag_rest_dist,
                        stiffness,
                    )
                if i < self.shape[0] - 1 and j < shape[0] - 1:
                    self.points[i, j].add_neighbor(
                        self.points[i + 1, j + 1],
                        diag_rest_dist,
                        stiffness,
                    )

    def relaxation_step(self, rate: float = 0.01):
        self.update_forces()
        self.update_positions(rate)

    def update_forces(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.points[i, j].compute_forces()

    def update_positions(self, rate: float) -> None:
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.points[i, j].apply_forces(rate)

    def get_z_positions(self):
        return np.array([[p.pivot[2] for p in row] for row in self.points])

    def get_z_forces(self):
        return np.array([[p.force[2] for p in row] for row in self.points])

    def get_neighbor_count(self):
        return np.array([[len(p.neighbors) for p in row] for row in self.points])


if __name__ == "__main__":

    p = SquarePatch((10, 10))
    p.points[2, 2].pivot[2] = 1
    fig, ax = plt.subplots()
    pc = ax.pcolormesh(p.get_z_positions())

    def update_animation(itrn):
        p.relaxation_step(rate=0.1)
        pc.set_array(p.get_z_positions())
        # pc.autoscale()
        ax.set_title(itrn)
        return pc

    anim = animation.FuncAnimation(
        fig, update_animation, frames=1000, interval=1, blit=False, repeat=False
    )
    plt.show()
