import random
import sys
import numpy as np
from random import randint

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QGraphicsRectItem,
    QWidget,
    QGraphicsScene,
)
from PyQt5.QtCore import QObject, QThread, QMutex, pyqtSignal, Qt
from PyQt5.QtGui import QKeyEvent


class GameEngine(QObject):
    gameDataSignal = pyqtSignal(object)

    def __init__(
        self,
        map_shape: np.ndarray,
        init_snake_size: int,
        tick_time_millis: int,
    ):
        QObject.__init__(self)
        self.game_started = False
        self.running = True
        self.runningLock = QMutex()
        self.map_shape = map_shape
        self.init_snake_size = init_snake_size
        self.food_position = None
        self.map = None
        self.snake_size = None
        self.snake_direction = None
        self.tick_time_millis = tick_time_millis

    def stop(self):
        self.runningLock.lock()
        self.running = False
        self.runningLock.unlock()

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

    def set_snake_direction(self, dir):
        self.runningLock.lock()
        self.snake_direction = dir
        self.runningLock.unlock()

    def init_food(self):
        self.runningLock.lock()
        correct_position_found = False
        while not correct_position_found:
            new_food_position = np.array(
                [randint(0, self.map_shape[0] - 1), randint(0, self.map_shape[1] - 1)]
            )
            if self.map[new_food_position[0], new_food_position[1]] != 0:
                self.food_position = new_food_position
                correct_position_found = True
        self.runningLock.unlock()

    def init_map(self):
        self.runningLock.lock()
        self.map = np.zeros(self.map_shape)
        self.snake_size = self.init_snake_size
        self.snake_direction = 0
        for i in range(self.snake_size):
            self.map[self.map_shape[0] // 2 - i, self.map_shape[1] // 2] = (
                self.init_snake_size - i
            )
        self.runningLock.unlock()

    def update_map(self):
        self.runningLock.lock()
        self.runningLock.unlock()

    def main_cycle(self):
        while self.stillRunning() and not self.is_game_started():
            self.thread().msleep(100)
            print("wait")
        while self.stillRunning() and self.is_game_started():
            self.init_map()
            self.init_food()
            self.update_map()
            self.gameDataSignal.emit((self.map, self.food_position))
            self.thread().msleep(self.tick_time_millis)
            print("cycle")


class GameThread(QThread):
    def __init__(
        self,
        produce_callback,
        map_shape: np.ndarray,
        init_snake_size: int,
        tick_time_millis: int,
    ):
        QThread.__init__(self)
        self.engine = GameEngine(
            init_snake_size=init_snake_size,
            map_shape=map_shape,
            tick_time_millis=tick_time_millis,
        )
        self.engine.moveToThread(self)
        self.started.connect(self.engine.main_cycle)
        self.engine.gameDataSignal.connect(produce_callback)

    def stop(self):
        self.engine.stop()


class GameScene(QWidget):
    def __init__(
        self,
        map_shape: np.ndarray = np.array([25, 25]),
        pixel_shape: np.ndarray = np.array([25, 25]),
        init_snake_size: int = 3,
        tick_time_millis: int = 1000,
        draw_rect_offset: int = 3,
    ):
        QWidget.__init__(self)
        self.map_shape = map_shape
        self.pixel_shape = pixel_shape
        self.init_snake_size = init_snake_size
        self.tick_time_millis = tick_time_millis
        self.draw_rect_offset = draw_rect_offset
        self.scene_layout = QVBoxLayout()
        self.setLayout(self.scene_layout)

        self.game_thread = GameThread(
            produce_callback=self.redraw_game_graphics,
            init_snake_size=init_snake_size,
            map_shape=map_shape,
            tick_time_millis=tick_time_millis,
        )
        self.game_thread.start()

    def redraw_game_graphics(self, gameData):
        rect = QGraphicsRectItem(0, 0, 50, 50)
        rect.setBrush(Qt.red)
        self.scene_layout.addWidget(rect)

    def sceneInterceptCloseEvent(self, evt):
        self.game_thread.stop()
        self.game_thread.wait()
        evt.accept()


class GameWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.scene = GameScene()
        self.setFixedSize(
            self.scene.pixel_shape[0] * self.scene.map_shape[0],
            self.scene.pixel_shape[1] * self.scene.map_shape[1],
        )
        self.setCentralWidget(self.scene)

    def closeEvent(self, evt):
        self.scene.sceneInterceptCloseEvent(evt)

    # def keyPressEvent(self, keyEvent: QKeyEvent) -> None:
    #    super(QMainWindow, self).keyPressEvent(keyEvent)
    #    print("key")
    #    if not self.scene.game_thread.engine.is_game_started():
    #        self.scene.game_thread.engine.start_game()
    #    match keyEvent.key():
    #        case Qt.Key_D:
    #            self.scene.game_thread.engine.set_snake_direction(0)
    #        case Qt.Key_W:
    #            self.scene.game_thread.engine.set_snake_direction(1)
    #        case Qt.Key_A:
    #            self.scene.game_thread.engine.set_snake_direction(2)
    #        case Qt.Key_S:
    #            self.scene.game_thread.engine.set_snake_direction(3)
    #        case Qt.Key_Escape:
    #            self.scene.game_thread.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GameWindow()
    win.show()
    app.exec_()
