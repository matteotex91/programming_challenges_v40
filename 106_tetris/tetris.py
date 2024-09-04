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

    def __init__(
        self,
        map_shape: np.ndarray,
        init_snake_size: int,
        tick_time_millis: int,
        flag_thoroid: bool,
    ):
        QObject.__init__(self)
        self.game_started = False
        self.running = True
        self.runningLock = QMutex()
        self.map_shape = map_shape
        self.init_snake_size = init_snake_size
        self.food_position = None
        self.head_position = None
        self.map = None
        self.snake_size = None
        self.snake_direction = None
        self.tick_time_millis = tick_time_millis
        self.flag_thoroid = flag_thoroid

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

    def set_snake_direction(self, dir):
        self.runningLock.lock()
        if self.snake_direction != (dir + 2) % 4:
            self.snake_direction = dir
        self.runningLock.unlock()

    def stop(self):
        self.runningLock.lock()
        self.running = False
        self.runningLock.unlock()

    def init_food(self):
        self.runningLock.lock()
        correct_position_found = False
        while not correct_position_found:
            new_food_position = np.array(
                [randint(0, self.map_shape[0] - 1), randint(0, self.map_shape[1] - 1)]
            )
            if self.map[*new_food_position] == 0:
                self.food_position = new_food_position
                correct_position_found = True
        self.runningLock.unlock()

    def init_map(self):
        self.runningLock.lock()
        self.map = np.zeros(self.map_shape)
        self.snake_size = self.init_snake_size
        self.head_position = [self.map_shape[0] // 2, self.map_shape[1] // 2]
        self.snake_direction = 0
        self.map[*self.head_position] = self.snake_size
        self.runningLock.unlock()

    def update_snake(self):
        self.runningLock.lock()
        self.map -= 1
        self.map = np.maximum(self.map, np.zeros_like(self.map))
        self.map[*self.head_position] = self.snake_size
        match self.snake_direction:
            case 0:
                self.head_position[0] += 1
            case 1:
                self.head_position[1] -= 1
            case 2:
                self.head_position[0] -= 1
            case 3:
                self.head_position[1] += 1
        if self.flag_thoroid:
            self.head_position[0] %= self.map_shape[0]
            self.head_position[1] %= self.map_shape[1]

        if (
            self.head_position[0] < 0
            or self.head_position[0] >= self.map_shape[0]
            or self.head_position[1] < 0
            or self.head_position[1] >= self.map_shape[1]
            or self.map[*self.head_position] != 0
        ):
            self.runningLock.unlock()
            # self.init_map()
            self.stop_game()
        else:
            self.runningLock.unlock()

    def update_food(self):
        self.runningLock.lock()
        if np.array_equal(self.food_position, self.head_position):
            self.snake_size += 1
            self.runningLock.unlock()
            self.init_food()
        else:
            self.runningLock.unlock()

    def main_cycle(self):
        while self.running:
            self.init_map()
            self.init_food()
            while self.running and not self.game_started:
                self.thread().msleep(100)
                self.gameDataSignal.emit((self.map, self.food_position))
            while self.running and self.game_started:
                self.update_snake()
                self.gameDataSignal.emit((self.map, self.food_position))
                self.update_food()
                self.thread().msleep(self.tick_time_millis)


class GameThread(QThread):
    def __init__(
        self,
        produce_callback,
        map_shape: np.ndarray,
        init_snake_size: int,
        tick_time_millis: int,
        flag_thoroid: bool,
    ):
        QThread.__init__(self)
        self.engine = GameEngine(
            init_snake_size=init_snake_size,
            map_shape=map_shape,
            tick_time_millis=tick_time_millis,
            flag_thoroid=flag_thoroid,
        )
        self.engine.moveToThread(self)
        self.started.connect(self.engine.main_cycle)
        self.engine.gameDataSignal.connect(produce_callback)

    def stop(self):
        self.engine.stop()


class GameWindow(QMainWindow):
    def __init__(
        self,
        map_shape: np.ndarray = np.array([25, 25]),
        pixel_shape: np.ndarray = np.array([25, 25]),
        init_snake_size: int = 3,
        tick_time_millis: int = 200,
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
            init_snake_size=init_snake_size,
            map_shape=map_shape,
            tick_time_millis=tick_time_millis,
            flag_thoroid=flag_thoroid,
        )
        self.game_thread.start()

    def closeEvent(self, evt):
        pass

    def keyPressEvent(self, keyEvent: QKeyEvent) -> None:
        super(QMainWindow, self).keyPressEvent(keyEvent)
        if not self.game_thread.engine.is_game_started():
            self.game_thread.engine.start_game()
        match keyEvent.key():
            case Qt.Key_D:
                self.game_thread.engine.set_snake_direction(0)
            case Qt.Key_W:
                self.game_thread.engine.set_snake_direction(1)
            case Qt.Key_A:
                self.game_thread.engine.set_snake_direction(2)
            case Qt.Key_S:
                self.game_thread.engine.set_snake_direction(3)
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
        food_position = game_data[1]
        for i in range(game_map.shape[0]):
            for j in range(game_map.shape[1]):
                if game_map[i, j] != 0:
                    painter.fillRect(
                        i * self.pixel_shape[0] + self.draw_rect_offset,
                        j * self.pixel_shape[1] + self.draw_rect_offset,
                        self.pixel_shape[0] - 2 * self.draw_rect_offset,
                        self.pixel_shape[1] - 2 * self.draw_rect_offset,
                        Qt.darkGreen,
                    )
                else:
                    painter.drawRect(
                        i * self.pixel_shape[0] + self.draw_rect_offset,
                        j * self.pixel_shape[1] + self.draw_rect_offset,
                        self.pixel_shape[0] - 2 * self.draw_rect_offset,
                        self.pixel_shape[1] - 2 * self.draw_rect_offset,
                    )
        painter.fillRect(
            food_position[0] * self.pixel_shape[0] + self.draw_rect_offset,
            food_position[1] * self.pixel_shape[0] + self.draw_rect_offset,
            self.pixel_shape[0] - 2 * self.draw_rect_offset,
            self.pixel_shape[1] - 2 * self.draw_rect_offset,
            Qt.red,
        )
        painter.end()
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GameWindow()
    win.show()
    app.exec_()
