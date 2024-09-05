import sys
import numpy as np
from random import randint

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
)
from PyQt5.QtCore import QObject, QThread, QMutex, pyqtSignal, Qt
from PyQt5.QtGui import QKeyEvent, QPixmap, QPainter, QPen


class GameEngine(QObject):
    gameDataSignal = pyqtSignal(object)

    """
    1 -> I
    2 -> O
    3 -> T
    4 -> S
    5 -> S'
    6 -> L
    7 -> L'
    """
    PIECES = [
        [],
        [
            np.array([[1, 1, 1, 1]]),
            np.array([[1], [1], [1], [1]]),
        ],
        [np.array([[2, 2], [2, 2]])],
        [
            np.array([[0, 3, 0], [3, 3, 3]]),
            np.array([[0, 3], [3, 3], [0, 3]]),
            np.array([[3, 3, 3], [0, 3, 0]]),
            np.array([[3, 0], [3, 3], [3, 0]]),
        ],
        [
            np.array([[0, 4, 4], [4, 4, 0]]),
            np.array([[4, 0], [4, 4], [0, 4]]),
        ],
        [
            np.array([[5, 5, 0], [0, 5, 5]]),
            np.array([[0, 5], [5, 5], [5, 0]]),
        ],
        [
            np.array([[0, 0, 6], [6, 6, 6]]),
            np.array([[6, 6], [0, 6], [0, 6]]),
            np.array([[6, 6, 6], [6, 0, 0]]),
            np.array([[6, 0], [6, 0], [6, 6]]),
        ],
        [
            np.array([[7, 0, 0], [7, 7, 7]]),
            np.array([[0, 7], [0, 7], [7, 7]]),
            np.array([[7, 7, 7], [0, 0, 7]]),
            np.array([[7, 7], [7, 0], [7, 0]]),
        ],
    ]
    PIECE_COLORS = np.array(
        [None, Qt.red, Qt.green, Qt.blue, Qt.cyan, Qt.magenta, Qt.yellow, Qt.black]
    )

    def __init__(self, map_shape: np.ndarray, tick_time_millis: int = 0):
        QObject.__init__(self)
        self.map_shape = map_shape
        self.tick_time_millis = tick_time_millis
        self.map = np.zeros(map_shape)
        self.current_piece = 0
        self.current_piece_orientation = 0
        self.current_piece_anchor = None
        self.game_started = False
        self.running = True
        self.runningLock = QMutex()
        self.already_updated_flag = False

    def stillRunning(self):
        self.runningLock.lock()
        value = self.running
        self.runningLock.unlock()
        return value

    def start_game(self):
        self.runningLock.lock()
        self.game_started = True
        self.runningLock.unlock()

    def is_game_started(self):
        self.runningLock.lock()
        value = self.game_started
        self.runningLock.unlock()
        return value

    def stop_game(self):
        self.runningLock.lock()
        self.game_started = False
        self.runningLock.unlock()

    def rotate_piece(self):
        self.runningLock.lock()
        self.current_piece_orientation = (self.current_piece_orientation + 1) % len(
            self.PIECES[self.current_piece]
        )
        self.runningLock.unlock()

        valid_pos = (
            self.x_boundary_valid()
            and self.y_boundary_valid()
            and self.is_piece_anchor_valid()
        )
        if not valid_pos:
            self.runningLock.lock()
            self.current_piece_orientation = (self.current_piece_orientation - 1) % len(
                self.PIECES[self.current_piece]
            )
            self.runningLock.unlock()

        self.gameDataSignal.emit(
            (
                self.map,
                self.current_piece,
                self.current_piece_anchor,
                self.current_piece_orientation,
            )
        )

    def move_piece_right(self):
        self.runningLock.lock()
        self.current_piece_anchor[0] += 1
        self.runningLock.unlock()
        valid_pos = (
            self.x_boundary_valid()
            and self.y_boundary_valid()
            and self.is_piece_anchor_valid()
        )
        if not valid_pos:
            self.runningLock.lock()
            self.current_piece_anchor[0] -= 1
            self.runningLock.unlock()

        self.gameDataSignal.emit(
            (
                self.map,
                self.current_piece,
                self.current_piece_anchor,
                self.current_piece_orientation,
            )
        )

    def move_piece_left(self):
        self.runningLock.lock()
        self.current_piece_anchor[0] -= 1
        self.runningLock.unlock()
        valid_pos = (
            self.x_boundary_valid()
            and self.y_boundary_valid()
            and self.is_piece_anchor_valid()
        )
        if not valid_pos:
            self.runningLock.lock()
            self.current_piece_anchor[0] += 1
            self.runningLock.unlock()

        self.gameDataSignal.emit(
            (
                self.map,
                self.current_piece,
                self.current_piece_anchor,
                self.current_piece_orientation,
            )
        )

    def drop_piece(self):
        valid_pos = True
        while valid_pos:
            self.runningLock.lock()
            self.current_piece_anchor[1] += 1
            self.runningLock.unlock()
            valid_pos = (
                self.x_boundary_valid()
                and self.y_boundary_valid()
                and self.is_piece_anchor_valid()
            )

        self.print_piece()
        self.new_piece()

    def stop(self):
        self.runningLock.lock()
        self.running = False
        self.runningLock.unlock()

    def is_piece_anchor_valid(self):
        self.runningLock.lock()
        current_piece_map = self.PIECES[self.current_piece][
            self.current_piece_orientation
        ]
        valid = True
        for i in range(current_piece_map.shape[0]):
            for j in range(current_piece_map.shape[1]):
                if (
                    current_piece_map[i, j] != 0
                    and self.map[
                        self.current_piece_anchor[0] + i,
                        self.current_piece_anchor[1] + j,
                    ]
                    != 0
                ):
                    valid = False
                    break
        self.runningLock.unlock()
        return valid

    def x_boundary_valid(self):
        self.runningLock.lock()
        valid = (
            self.current_piece_anchor[0] >= 0
            and self.current_piece_anchor[0]
            + self.PIECES[self.current_piece][self.current_piece_orientation].shape[0]
            - 1
            < self.map_shape[0]
        )
        self.runningLock.unlock()
        return valid

    def y_boundary_valid(self):
        self.runningLock.lock()
        valid = (
            self.current_piece_anchor[1] >= 0
            and self.current_piece_anchor[1]
            + self.PIECES[self.current_piece][self.current_piece_orientation].shape[1]
            - 1
            < self.map_shape[1]
        )
        self.runningLock.unlock()
        return valid

    def new_piece(self):
        self.runningLock.lock()
        self.already_updated_flag = True
        self.current_piece = randint(1, 7)
        self.current_piece_orientation = 0
        self.current_piece_anchor = np.array(
            [
                (
                    self.map_shape[0]
                    - self.PIECES[self.current_piece][
                        self.current_piece_orientation
                    ].shape[0]
                )
                // 2,
                0,
            ]
        )
        self.runningLock.unlock()
        valid_pos = (
            self.x_boundary_valid()
            and self.y_boundary_valid()
            and self.is_piece_anchor_valid()
        )
        if not valid_pos:
            self.stop_game()
        self.gameDataSignal.emit(
            (
                self.map,
                self.current_piece,
                self.current_piece_anchor,
                self.current_piece_orientation,
            )
        )

    def init_map(self):
        self.runningLock.lock()
        self.map = np.zeros(self.map_shape)
        self.runningLock.unlock()

    def update_game(self):
        self.runningLock.lock()
        self.current_piece_anchor[1] += 1
        self.runningLock.unlock()
        valid_pos = (
            self.x_boundary_valid()
            and self.y_boundary_valid()
            and self.is_piece_anchor_valid()
        )
        if not valid_pos:
            self.print_piece()
            self.new_piece()

    def print_piece(self):
        self.runningLock.lock()
        self.current_piece_anchor[1] -= 1
        current_piece_map = self.PIECES[self.current_piece][
            self.current_piece_orientation
        ]
        for i in range(current_piece_map.shape[0]):
            for j in range(current_piece_map.shape[1]):
                if current_piece_map[i, j] != 0:
                    self.map[
                        self.current_piece_anchor[0] + i,
                        self.current_piece_anchor[1] + j,
                    ] = self.PIECES[self.current_piece][self.current_piece_orientation][
                        i, j
                    ]
        self.runningLock.unlock()
        self.check_full_rows()

    def check_full_rows(self):
        self.runningLock.lock()
        for i in range(self.map_shape[1]):
            if np.sum(self.map[:, i] != 0) == self.map_shape[0]:
                # self.map[:, i] = 0
                self.map[:, 1 : i + 1] = self.map[:, 0:i]
                self.map[:, 0] = 0

                # self.check_full_rows()
        self.runningLock.unlock()

    def main_cycle(self):
        while self.running:
            self.init_map()
            self.new_piece()
            while self.running and not self.game_started:
                self.thread().msleep(50)
                self.gameDataSignal.emit(
                    (
                        self.map,
                        self.current_piece,
                        self.current_piece_anchor,
                        self.current_piece_orientation,
                    )
                )
            while self.running and self.game_started:
                if not self.already_updated_flag:
                    self.update_game()
                else:
                    self.already_updated_flag = False
                self.gameDataSignal.emit(
                    (
                        self.map,
                        self.current_piece,
                        self.current_piece_anchor,
                        self.current_piece_orientation,
                    )
                )
                # self.update_piece()
                self.thread().msleep(self.tick_time_millis)


class GameThread(QThread):
    def __init__(
        self,
        produce_callback,
        map_shape: np.ndarray,
        tick_time_millis: int,
    ):
        QThread.__init__(self)
        self.engine = GameEngine(
            map_shape=map_shape,
            tick_time_millis=tick_time_millis,
        )
        self.engine.moveToThread(self)
        self.started.connect(self.engine.main_cycle)
        self.engine.gameDataSignal.connect(produce_callback)

    def stop(self):
        self.engine.stop()


class GameWindow(QMainWindow):
    def __init__(
        self,
        map_shape: np.ndarray = np.array([10, 25]),
        pixel_shape: np.ndarray = np.array([25, 25]),
        init_snake_size: int = 3,
        tick_time_millis: int = 500,
        draw_rect_offset: int = 3,
        flag_thoroid: bool = True,
    ):
        QMainWindow.__init__(self)
        self.map_shape = map_shape
        self.pixel_shape = pixel_shape
        self.init_snake_size = init_snake_size
        self.tick_time_millis = tick_time_millis
        self.draw_rect_offset = draw_rect_offset
        self.label = QLabel()
        canvas = QPixmap(
            pixel_shape[0] * map_shape[0],
            pixel_shape[1] * map_shape[1],
        )
        canvas.fill(Qt.white)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)

        self.game_thread = GameThread(
            produce_callback=self.redraw_game_graphics,
            map_shape=map_shape,
            tick_time_millis=tick_time_millis,
        )
        self.game_thread.start()

    def closeEvent(self, evt):
        pass

    def keyPressEvent(self, keyEvent: QKeyEvent) -> None:
        super(QMainWindow, self).keyPressEvent(keyEvent)
        if not self.game_thread.engine.is_game_started():
            self.game_thread.engine.start_game()
        match keyEvent.key():
            case Qt.Key_Space:
                self.game_thread.engine.rotate_piece()
            case Qt.Key_D:
                self.game_thread.engine.move_piece_right()
            case Qt.Key_A:
                self.game_thread.engine.move_piece_left()
            case Qt.Key_S:
                self.game_thread.engine.drop_piece()
            case Qt.Key_Escape:
                self.game_thread.stop()
                self.close()

    def redraw_game_graphics(self, game_data):
        canvas = self.label.pixmap()
        canvas.fill(Qt.white)
        painter = QPainter(canvas)
        pen = QPen()
        pen.setColor(Qt.lightGray)
        painter.setPen(pen)
        game_map = game_data[0]
        current_piece = game_data[1]
        current_piece_anchor = game_data[2]
        current_piece_orientation = game_data[3]

        for i in range(game_map.shape[0]):
            for j in range(game_map.shape[1]):
                if game_map[i, j] != 0:
                    painter.fillRect(
                        i * self.pixel_shape[0] + self.draw_rect_offset,
                        j * self.pixel_shape[1] + self.draw_rect_offset,
                        self.pixel_shape[0] - 2 * self.draw_rect_offset,
                        self.pixel_shape[1] - 2 * self.draw_rect_offset,
                        self.game_thread.engine.PIECE_COLORS[int(game_map[i, j])],
                    )
                else:
                    painter.drawRect(
                        i * self.pixel_shape[0] + self.draw_rect_offset,
                        j * self.pixel_shape[1] + self.draw_rect_offset,
                        self.pixel_shape[0] - 2 * self.draw_rect_offset,
                        self.pixel_shape[1] - 2 * self.draw_rect_offset,
                    )
        current_piece_map = self.game_thread.engine.PIECES[current_piece][
            current_piece_orientation
        ]
        for i in range(current_piece_map.shape[0]):
            for j in range(current_piece_map.shape[1]):
                if current_piece_map[i, j] != 0:
                    painter.fillRect(
                        (i + current_piece_anchor[0]) * self.pixel_shape[0]
                        + self.draw_rect_offset,
                        (j + current_piece_anchor[1]) * self.pixel_shape[1]
                        + self.draw_rect_offset,
                        self.pixel_shape[0] - 2 * self.draw_rect_offset,
                        self.pixel_shape[1] - 2 * self.draw_rect_offset,
                        self.game_thread.engine.PIECE_COLORS[current_piece],
                    )

        painter.end()
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GameWindow()
    win.show()
    app.exec_()
