import sys
import numpy as np
from time import time
from matplotlib import colormaps


from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (
    QPixmap,
    QMouseEvent,
    QImage,
    QWheelEvent,
)


class GameWindow(QMainWindow):
    def __init__(
        self,
        map_size: int = 500,
        real_range: np.ndarray = np.array([-1, 1]),
        comp_range: np.ndarray = np.array([-1, 1]),
    ):
        QMainWindow.__init__(self)
        self.map_size = map_size
        self.real_range = real_range
        self.comp_range = comp_range
        self.map = None
        self.qt_image = None
        self.setFixedSize(map_size, map_size)
        self.label = QLabel()
        canvas = QPixmap(map_size, map_size)

        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)
        self.render_fractal()
        self.redraw_mandelbrot_graphics()

    def render_fractal(self):
        real_arr = np.linspace(*self.real_range, self.map_size).T
        comp_arr = np.linspace(*self.comp_range, self.map_size)
        real = np.resize(real_arr, (self.map_size, self.map_size))
        comp = np.resize(comp_arr, (self.map_size, self.map_size)).T
        mat = real + 1j * comp

        include_map = np.ones_like(mat)
        abs_mat = np.abs(mat)
        iterMat = np.zeros_like(mat)
        iteration_count_map = np.zeros_like(mat)

        range_limit = 4
        iteration_limit = 100
        max_iteration = 0

        # for i in range(iteration_limit):
        while max_iteration < iteration_limit:
            indexes = np.where(include_map == 1)
            iterMat[indexes] = np.power(iterMat[indexes], 2) + mat[indexes]
            iteration_count_map[indexes] += 1
            max_iteration = np.max(iteration_count_map)
            abs_mat = np.abs(iterMat)
            include_map[
                np.where(
                    np.logical_or(
                        abs_mat > range_limit, iteration_count_map > iteration_limit
                    )
                )
            ] = 0
        abs_mat[np.isnan(abs_mat)] = 0
        iteration_count_map = np.abs(iteration_count_map)
        # iteration_count_map[np.where(include_map == 0)] = 0

        hue = iteration_count_map / np.max(iteration_count_map)
        cmap = colormaps["hsv"]
        self.rgb_image = np.uint8(256 * cmap(hue)[..., :3])

    def mousePressEvent(self, event: QMouseEvent):
        mouse_x = event.pos().x()
        mouse_y = event.pos().y()

        mouse_pos = np.array(
            [
                self.real_range[0]
                + (mouse_x) * (self.real_range[1] - self.real_range[0]) / self.map_size,
                self.comp_range[0]
                + (mouse_y) * (self.comp_range[1] - self.comp_range[0]) / self.map_size,
            ]
        )
        magn_factor = 2.5
        if event.button() == Qt.RightButton:
            self.real_range = (
                mouse_pos[0] + (self.real_range - mouse_pos[0]) * magn_factor
            )
            self.comp_range = (
                mouse_pos[1] + (self.comp_range - mouse_pos[1]) * magn_factor
            )
        elif event.button() == Qt.LeftButton:
            self.real_range = (
                mouse_pos[0] + (self.real_range - mouse_pos[0]) / magn_factor
            )
            self.comp_range = (
                mouse_pos[1] + (self.comp_range - mouse_pos[1]) / magn_factor
            )
        self.render_fractal()
        self.redraw_mandelbrot_graphics()

    def mouseReleaseEvent(self, event: QMouseEvent):
        # if (
        #    self.pressPos is not None
        #    and event.pos() in self.label.rect()
        #    and event.button() == Qt.LeftButton
        # ):
        #    y0 = min(self.pressPos.x(), event.pos().x())
        #    y1 = max(self.pressPos.x(), event.pos().x())
        #    x0 = min(self.pressPos.y(), event.pos().y())
        #    x1 = max(self.pressPos.y(), event.pos().y())
        #
        #    xr0 = x0 * (self.real_range[1] - self.real_range[0]) / self.map_size
        #    xr1 = x1 * (self.real_range[1] - self.real_range[0]) / self.map_size
        #    yr0 = y0 * (self.comp_range[1] - self.comp_range[0]) / self.map_size
        #    yr1 = y1 * (self.comp_range[1] - self.comp_range[0]) / self.map_size
        #
        #    self.real_range = np.array([xr0, xr1])
        #    self.comp_range = np.array([yr0, yr1])

        # self.real_range = np.array([-2, 2])
        # self.comp_range = np.array([-2, 2])
        #
        # self.pressPos = None
        # self.render_fractal()
        # self.redraw_mandelbrot_graphics()
        ...

    def redraw_mandelbrot_graphics(self):
        self.label.setPixmap(
            QPixmap(
                QImage(
                    self.rgb_image,
                    self.map_size,
                    self.map_size,
                    QImage.Format.Format_RGB888,
                )
            )
        )
        self.update()


def test_plot():
    N = 501
    real_range = np.linspace(-1, 1, N).T
    comp_range = np.linspace(-1, 1, N)
    real = np.resize(real_range, (N, N))
    comp = np.resize(comp_range, (N, N)).T
    mat = real + 1j * comp

    include_map = np.ones_like(mat)
    abs_mat = np.abs(mat)
    iterMat = np.zeros_like(mat)

    t0 = time()
    for i in range(100):
        indexes = np.where(include_map == 1)
        iterMat[indexes] = np.power(iterMat[indexes], 2) + mat[indexes]
        abs_mat = np.abs(iterMat)
        include_map[np.where(abs_mat > np.average(abs_mat) + np.std(abs_mat))] = 0
    print(time() - t0)

    import matplotlib.pyplot as plt

    plt.pcolormesh(np.abs(iterMat))
    plt.show()


if __name__ == "__main__":
    #  test_plot()
    app = QApplication(sys.argv)
    win = GameWindow()
    win.show()
    app.exec()
