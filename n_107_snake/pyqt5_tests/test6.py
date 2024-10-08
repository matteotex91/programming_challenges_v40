import numpy as np
from threading import Thread
from random import randint
from typing import Optional
import sys
from time import sleep
from PyQt5.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QGraphicsRectItem,
    QApplication,
)
from PyQt5.QtGui import QBrush, QPen
from PyQt5.QtCore import Qt, QRunnable, pyqtSlot, QThreadPool
from PyQt5.QtGui import QKeyEvent, QCloseEvent


""" This class contains all the informations and functions needed to plot and collect keyboard events
Snake directions :
0 -> right
1 -> up
2 -> left
3 -> down
"""


class SnakeGraphicsView(QGraphicsView):
    def __init__(
        self,
        map_shape: np.ndarray = np.array([25, 25]),
        pixel_shape: np.ndarray = np.array([25, 25]),
        draw_rect_offset: int = 3,
    ) -> None:
        QGraphicsView.__init__(self)
        self._pixel_shape = pixel_shape
        self._graphics_shape = map_shape * pixel_shape
        self._draw_rect_offset = draw_rect_offset
        self._scene = QGraphicsScene(
            0, 0, self._graphics_shape[0], self._graphics_shape[1]
        )
        self.setScene(self._scene)
        self._snake_direction = 0
        self._running = True
        self._snake_brush = QBrush(Qt.blue)
        self._food_brush = QBrush(Qt.red)
        self._brushes = [self._snake_brush, self._food_brush]
        self._key_trigger = False
        self._snake_engine = SnakeEngine(self, map_shape)
        self.threadpool = QThreadPool()
        self.threadpool.start(self._snake_engine)
        self.show()

    def keyPressEvent(self, keyEvent: QKeyEvent) -> None:
        self._key_trigger = False
        # Process the event as this function was not overdriven
        super(SnakeGraphicsView, self).keyPressEvent(keyEvent)
        match keyEvent.key():
            case Qt.Key_D:
                self._snake_direction = 0
            case Qt.Key_W:
                self._snake_direction = 1
            case Qt.Key_A:
                self._snake_direction = 2
            case Qt.Key_S:
                self._snake_direction = 3
            case Qt.Key_Escape:
                self._running = False
                self.close()

    def closeEvent(self, closeEvent: QCloseEvent):
        super(SnakeGraphicsView, self).closeEvent(closeEvent)
        self._running = False
        self.close()

    """ This function draws a color in the scene
    0 -> snake
    1 -> food
    """

    def draw_rect(self, pixel_x: int, pixel_y: int, color: int):
        rect = QGraphicsRectItem(
            pixel_y * self._pixel_shape[1] + self._draw_rect_offset,
            pixel_x * self._pixel_shape[0] + self._draw_rect_offset,
            self._pixel_shape[1] - 2 * self._draw_rect_offset,
            self._pixel_shape[0] - 2 * self._draw_rect_offset,
        )
        rect.setBrush(self._brushes[color])
        self._scene.addItem(rect)

    def redraw_snake_graphics(self) -> None:
        self._scene.clear()
        map_shape = self._snake_engine._map.shape
        for ix in range(map_shape[0]):
            for iy in range(map_shape[1]):
                if map[ix, iy] != 0:
                    self.draw_rect(ix, iy, 0)

        self.draw_rect(
            self._snake_engine._food_position[0],
            self._snake_engine._food_position[1],
            1,
        )
        # self.setScene(self._scene)
        # self.show()


class SnakeEngine(QRunnable):
    INIT_SIZE = 3

    def __init__(
        self,
        graphics: SnakeGraphicsView,
        map_shape: np.ndarray = np.array([25, 25]),
        pixel_shape: np.ndarray = np.array([25, 25]),
        draw_rect_offset: int = 3,
    ):
        QRunnable.__init__(self)
        self._map_shape = map_shape
        self._pixel_shape = pixel_shape
        self._draw_rect_offset = draw_rect_offset
        # head position
        self._alive = True
        self._graphics = graphics

    @pyqtSlot()
    def run(self):
        while self._graphics._running:
            self.init_game()
            self._graphics.redraw_snake_graphics(self._map, self._food_position)
            self._graphics._key_trigger = True
            while self._graphics._key_trigger:
                sleep(0.1)
            while self._alive:
                sleep(1)
                self._graphics.redraw_snake_graphics(self._map, self._food_position)

    def init_food(self):
        correct_position_found = False
        while not correct_position_found:
            new_food_position = np.array(
                [randint(0, self._map_shape[0] - 1), randint(0, self._map_shape[1] - 1)]
            )
            if self._map[new_food_position[0], new_food_position[1]] != 0:
                self._food_position = new_food_position
                correct_position_found = True

    def init_game(self):
        self._map = np.zeros(self._map_shape)
        self._size = self.INIT_SIZE
        for i in range(self._size):
            self._map[self._map_shape[0] // 2 - i, self._map_shape[1] // 2] = (
                self.INIT_SIZE - i
            )
        self.init_food()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    game = SnakeGraphicsView()
    app.exec_()
