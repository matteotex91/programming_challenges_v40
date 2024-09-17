import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QKeyEvent, QPixmap, QPainter, QPen, QMouseEvent


class GameWindow(QMainWindow):
    def __init__(
        self,
        map_shape: np.ndarray = np.array([28, 28]),
        pixel_shape: np.ndarray = np.array([25, 25]),
        pixel_offset: int = 3,
    ):
        QMainWindow.__init__(self)
        self.map_shape = map_shape
        self.map = np.zeros(map_shape)
        self.pixel_shape = pixel_shape
        self.pixel_offset = pixel_offset
        self.label = QLabel()
        self.setFixedSize(*(map_shape * pixel_shape))
        canvas = QPixmap(*(map_shape * pixel_shape))
        canvas.fill(Qt.lightGray)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)
        self.redraw_game_graphics()
        self.clicked = False

    def redraw_game_graphics(self):
        canvas = self.label.pixmap()
        canvas.fill(Qt.lightGray)
        painter = QPainter(canvas)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(Qt.darkGray)
        painter.setPen(pen)
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                painter.drawRect(
                    i * self.pixel_shape[0],
                    j * self.pixel_shape[1],
                    self.pixel_shape[0],
                    self.pixel_shape[1],
                )
                if self.map[i, j] == 255:
                    painter.fillRect(
                        i * self.pixel_shape[0] + self.pixel_offset,
                        j * self.pixel_shape[1] + self.pixel_offset,
                        self.pixel_shape[0] - 2 * self.pixel_offset,
                        self.pixel_shape[1] - 2 * self.pixel_offset,
                        Qt.darkGray,
                    )
        painter.end()
        self.update()

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        self.clicked = True

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        self.clicked = False

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        if self.clicked:
            x = int(a0.x() / self.pixel_shape[0])
            y = int(a0.y() / self.pixel_shape[1])
            self.map[x, y] = 255
            self.redraw_game_graphics()

    def keyReleaseEvent(self, a0: QKeyEvent | None) -> None:
        if a0.key() == Qt.Key_Space:
            self.map = np.zeros(self.map_shape)
            self.redraw_game_graphics()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GameWindow()
    win.show()
    app.exec_()
