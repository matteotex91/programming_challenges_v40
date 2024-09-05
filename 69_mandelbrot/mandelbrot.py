import sys
import numpy as np
from random import randint
from scipy.ndimage import convolve, label
from time import time
from matplotlib import colormaps


from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import (
    QPixmap,
    QPainter,
    QPen,
    QMouseEvent,
    QColor,
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

        for i in range(20):
            indexes = np.where(include_map == 1)
            iterMat[indexes] = np.power(iterMat[indexes], 2) + mat[indexes]
            abs_mat = np.abs(iterMat)
            include_map[np.where(abs_mat > np.average(abs_mat) + np.std(abs_mat))] = 0
        abs_mat[np.isnan(abs_mat)] = 0
        hue = abs_mat / np.max(abs_mat)
        cmap = colormaps["hsv"]
        self.rgb_image = np.uint8(256 * cmap(hue)[..., :3])

    def wheelEvent(self, event: QWheelEvent):
        # mouse_x = event.pos().y()
        # mouse_y = event.pos().x()
        # mouse_pos = np.array(
        #    [
        #        mouse_x * (self.real_range[1] - self.real_range[0]) / self.map_size,
        #        mouse_y * (self.comp_range[1] - self.comp_range[0]) / self.map_size,
        #    ]
        # )
        # if event.angleDelta().y() > 0:
        #    self.real_range = mouse_pos[0] + 1.1 * (self.real_range - mouse_pos[0])
        #    self.comp_range = mouse_pos[1] + 1.1 * (self.comp_range - mouse_pos[1])
        # else:
        #    self.real_range = mouse_pos[0] + 0.9 * (self.real_range - mouse_pos[0])
        #    self.comp_range = mouse_pos[1] + 0.9 * (self.comp_range - mouse_pos[1])
        # self.render_fractal()
        # self.redraw_mandelbrot_graphics()
        ...

    def mousePressEvent(self, event: QMouseEvent):
        mouse_x = event.pos().y()
        mouse_y = event.pos().x()
        mouse_pos = np.array(
            [
                self.real_range[0]
                + mouse_x * (self.real_range[1] - self.real_range[0]) / self.map_size,
                self.comp_range[0]
                + mouse_y * (self.comp_range[1] - self.comp_range[0]) / self.map_size,
            ]
        )
        if event.button() == Qt.LeftButton:
            self.real_range = mouse_pos[0] + 1.1 * (self.real_range - mouse_pos[0])
            self.comp_range = mouse_pos[1] + 1.1 * (self.comp_range - mouse_pos[1])
        else:
            self.real_range = mouse_pos[0] + 0.9 * (self.real_range - mouse_pos[0])
            self.comp_range = mouse_pos[1] + 0.9 * (self.comp_range - mouse_pos[1])
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
        # canvas = self.label.pixmap()
        # painter = QPainter(canvas)
        # self.label.setPixmap(
        #    QPixmap.fromImage(self.qt_image, QImage.Format.Format_RGB888)
        # )
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
        # painter.end()
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
