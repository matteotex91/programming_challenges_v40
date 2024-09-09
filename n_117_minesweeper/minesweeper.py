import sys
import numpy as np
from random import randint
from scipy.ndimage import convolve, label

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QPainter, QPen, QMouseEvent


class GameWindow(QMainWindow):
    def __init__(
        self,
        map_shape: np.ndarray = np.array([10, 10]),
        mines_count: int = 3,
        pixel_shape: np.ndarray = np.array([25, 25]),
        pixel_offset: int = 3,
    ):
        QMainWindow.__init__(self)
        self.map_shape = map_shape
        self.mines_count = mines_count
        self.pixel_shape = pixel_shape
        self.pixel_offset = pixel_offset
        self.label = QLabel()
        canvas = QPixmap(*(map_shape * pixel_shape))
        canvas.fill(Qt.lightGray)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)
        self.alive = True
        self.win = False
        self.map = None
        self.opened = None  # 0 closed, 1 opened, 2 marked
        self.neighbor_map = None
        self.start_game()
        self.redraw_game_graphics()

    def start_game(self):
        self.opened = np.zeros(self.map_shape)
        self.map = np.zeros_like(self.opened)
        self.neighbor_map = np.zeros_like(self.opened)
        mines_ind = 0
        while mines_ind < self.mines_count:
            candidate_pos = np.array(
                [randint(0, self.map_shape[0] - 1), randint(0, self.map_shape[1] - 1)]
            )
            if self.map[*candidate_pos] == 0:
                self.map[*candidate_pos] = 1
                mines_ind += 1
        self.neighbor_map = convolve(
            self.map, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), mode="constant"
        )

    def mark_position(self, pos):
        self.opened[*pos] = (
            2
            if self.opened[*pos] == 0
            else (0 if self.opened[*pos] == 2 else self.opened[*pos])
        )

    def touch_position(self, pos):
        if self.map[*pos] != 0:
            self.alive = False
        self.opened[*pos] = 1
        has_not_neighbor_mine = np.zeros_like(self.map)
        has_not_neighbor_mine[np.where(self.neighbor_map == 0)] = 1
        labels, _ = label(has_not_neighbor_mine)
        if labels[*pos] != 0:
            self.opened[np.where(labels == labels[*pos])] = 1
            neighbors_to_open = np.zeros_like(self.map)
            neighbors_to_open[
                np.where(np.logical_and(labels == labels[*pos], self.map == 0))
            ] = 1
            neighbors_to_open = convolve(
                neighbors_to_open,
                np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
                mode="constant",
            )
            self.opened[
                np.where(np.logical_and(neighbors_to_open != 0, self.neighbor_map != 0))
            ] = 1
        print("stop here")
        # if map==0 and neighbors==0, calculate all the recursively contiguous cells satisfying the same conditions. Use label
        # then, open these cells and open also the ones with neighbors!=0 next to them, but of course not the map!=0

    def check_win(self):
        if (
            np.sum(np.logical_and(self.opened == 2, self.map != 0)) == self.mines_count
            or np.sum(self.opened == 1)
            == self.map_shape[0] * self.map_shape[1] - self.mines_count
        ):
            self.win = True
            self.alive = False

    def mousePressEvent(self, event: QMouseEvent):
        self.pressPos = event.pos()

    def mouseReleaseEvent(self, event: QMouseEvent):
        # ensure that the left button was pressed *and* released within the
        # geometry of the widget; if so, emit the signal;
        if self.pressPos is not None and event.pos() in self.label.rect():
            if self.alive:
                press_index = np.array(
                    [
                        int(self.pressPos.x() / self.pixel_shape[0]),
                        int(self.pressPos.y() / self.pixel_shape[1]),
                    ]
                )
                if event.button() == Qt.LeftButton:
                    self.touch_position(press_index)
                elif event.button() == Qt.RightButton:
                    self.mark_position(press_index)
            else:
                self.start_game()
                self.alive = True
        self.check_win()
        self.pressPos = None
        self.redraw_game_graphics()

    def redraw_game_graphics(self):
        canvas = self.label.pixmap()
        if self.alive:
            canvas.fill(Qt.lightGray)
        else:
            if self.win:
                canvas.fill(Qt.darkGreen)
            else:
                canvas.fill(Qt.darkRed)
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
                match self.opened[i, j]:
                    case 0:  # closed
                        painter.fillRect(
                            i * self.pixel_shape[0] + self.pixel_offset,
                            j * self.pixel_shape[1] + self.pixel_offset,
                            self.pixel_shape[0] - 2 * self.pixel_offset,
                            self.pixel_shape[1] - 2 * self.pixel_offset,
                            Qt.darkGray,
                        )
                    case 1:  # opened
                        if self.neighbor_map[i, j] != 0 and self.map[i, j] == 0:
                            painter.drawText(
                                QRect(
                                    i * self.pixel_shape[0] + self.pixel_offset,
                                    j * self.pixel_shape[1] + self.pixel_offset,
                                    self.pixel_shape[0] - 2 * self.pixel_offset,
                                    self.pixel_shape[1] - 2 * self.pixel_offset,
                                ),
                                0,
                                str(int(self.neighbor_map[i, j])),
                            )
                        elif self.map[i, j] != 0:
                            painter.drawText(
                                QRect(
                                    i * self.pixel_shape[0] + self.pixel_offset,
                                    j * self.pixel_shape[1] + self.pixel_offset,
                                    self.pixel_shape[0] - 2 * self.pixel_offset,
                                    self.pixel_shape[1] - 2 * self.pixel_offset,
                                ),
                                0,
                                "X",
                            )
                    case 2:
                        painter.fillRect(
                            i * self.pixel_shape[0] + self.pixel_offset,
                            j * self.pixel_shape[1] + self.pixel_offset,
                            self.pixel_shape[0] - 2 * self.pixel_offset,
                            self.pixel_shape[1] - 2 * self.pixel_offset,
                            Qt.green,
                        )

        painter.end()
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GameWindow()
    win.show()
    app.exec_()
