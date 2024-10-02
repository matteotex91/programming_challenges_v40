import sys

from n_72_sudoku_solver.main import Sudoku
import numpy as np


from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QPainter, QPen, QKeyEvent


class GameWindow(QMainWindow):
    def __init__(
        self,
        pixel_shape: np.ndarray = np.array([30, 30]),
        pixel_offset: int = 3,
    ):
        QMainWindow.__init__(self)
        self.map_shape = np.array([9, 9])
        self.cursor_position = np.array([0, 0])
        self.pixel_shape = pixel_shape
        self.setFixedSize(self.pixel_shape[0] * 9, self.pixel_shape[1] * 9)
        self.pixel_offset = pixel_offset
        self.label = QLabel()
        canvas = QPixmap(*(self.map_shape * pixel_shape))
        canvas.fill(Qt.lightGray)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)
        self.start_game()
        self.last_key = None

    def start_game(self):
        self.sudoku = Sudoku(random_fill=True, random_erase=True, guessing_depth=0)
        self.complete = np.sum(self.sudoku.data == 0) == 0
        self.redraw_game_graphics()

    def attempt(self, number):
        if number in self.sudoku.available_number_list(self.cursor_position):
            self.sudoku.data[*self.cursor_position] = number
        self.complete = np.sum(self.sudoku.data == 0) == 0
        self.redraw_game_graphics()

    def move_cursor(self, direction):
        self.cursor_position[0] = (
            self.cursor_position[0] + direction[0]
        ) % self.map_shape[0]
        self.cursor_position[1] = (
            self.cursor_position[1] + direction[1]
        ) % self.map_shape[1]
        self.redraw_game_graphics()

    def keyReleaseEvent(self, event: QKeyEvent):
        match event.key():
            case Qt.Key_Up:
                self.move_cursor([0, -1])
            case Qt.Key_Down:
                self.move_cursor([0, 1])
            case Qt.Key_Right:
                self.move_cursor([1, 0])
            case Qt.Key_Left:
                self.move_cursor([-1, 0])
            case Qt.Key_0:
                self.sudoku.data[*self.cursor_position] = 0
                self.redraw_game_graphics()
            case Qt.Key_1:
                self.attempt(1)
            case Qt.Key_2:
                self.attempt(2)
            case Qt.Key_3:
                self.attempt(3)
            case Qt.Key_4:
                self.attempt(4)
            case Qt.Key_5:
                self.attempt(5)
            case Qt.Key_6:
                self.attempt(6)
            case Qt.Key_7:
                self.attempt(7)
            case Qt.Key_8:
                self.attempt(8)
            case Qt.Key_9:
                self.attempt(9)
            case Qt.Key_N:
                if self.last_key == Qt.Key_N:
                    self.start_game()
        self.last_key = event.key()

    def redraw_game_graphics(self):
        canvas = self.label.pixmap()
        if self.complete:
            canvas.fill(Qt.darkGreen)
        else:
            canvas.fill(Qt.lightGray)
        painter = QPainter(canvas)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(Qt.darkGray)
        painter.setPen(pen)
        cursor_pen = QPen()
        cursor_pen.setWidth(4)
        cursor_pen.setColor(Qt.darkGreen)
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                painter.drawRect(
                    i * self.pixel_shape[0],
                    j * self.pixel_shape[1],
                    self.pixel_shape[0],
                    self.pixel_shape[1],
                )
                if self.sudoku.data[i, j] != 0:
                    painter.drawText(
                        QRect(
                            i * self.pixel_shape[0] + self.pixel_offset,
                            j * self.pixel_shape[1] + self.pixel_offset,
                            self.pixel_shape[0] - 2 * self.pixel_offset,
                            self.pixel_shape[1] - 2 * self.pixel_offset,
                        ),
                        0,
                        str(int(self.sudoku.data[i, j])),
                    )
        painter.setPen(cursor_pen)
        painter.drawRect(
            self.cursor_position[0] * self.pixel_shape[0],
            self.cursor_position[1] * self.pixel_shape[1],
            self.pixel_shape[0],
            self.pixel_shape[1],
        )
        cursor_pen.setColor(Qt.black)
        cursor_pen.setWidth(2)
        painter.setPen(cursor_pen)
        painter.drawLine(
            0, 3 * self.pixel_shape[1], 9 * self.pixel_shape[0], 3 * self.pixel_shape[1]
        )
        painter.drawLine(
            0, 6 * self.pixel_shape[1], 9 * self.pixel_shape[0], 6 * self.pixel_shape[1]
        )
        painter.drawLine(
            3 * self.pixel_shape[0],
            0,
            3 * self.pixel_shape[1],
            9 * self.pixel_shape[0],
        )
        painter.drawLine(
            6 * self.pixel_shape[0],
            0,
            6 * self.pixel_shape[1],
            9 * self.pixel_shape[0],
        )
        painter.end()
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GameWindow()
    win.show()
    app.exec_()
