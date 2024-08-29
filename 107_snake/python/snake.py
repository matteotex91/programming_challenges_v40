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
from PyQt5.QtCore import Qt
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

    def redraw_snake_graphics(self, map: np.ndarray, food_position: np.ndarray) -> None:
        self._scene.clear()
        map_shape = map.shape
        for ix in range(map_shape[0]):
            for iy in range(map_shape[1]):
                if map[ix, iy] != 0:
                    self.draw_rect(ix, iy, 0)

        self.draw_rect(food_position[0], food_position[1], 1)
        # self.setScene(self._scene)
        # self.show()


class SnakeEngine:
    INIT_SIZE = 3

    def __init__(
        self,
        map_shape: np.ndarray = np.array([25, 25]),
        pixel_shape: np.ndarray = np.array([25, 25]),
        draw_rect_offset: int = 3,
    ):
        self._map_shape = map_shape
        self.init_game()
        self._graphics = SnakeGraphicsView(map_shape, pixel_shape, draw_rect_offset)
        self._alive = True
        game_thread = Thread(target=self.game_cycle)
        game_thread.start()

    def game_cycle(self):
        while self._graphics._running:
            self._graphics._key_trigger = True
            while self._graphics._key_trigger:
                sleep(0.1)
            while self._alive:
                print("test")
                sleep(1)

    def init_game(self):
        self._map = np.zeros(self._map_shape)
        self._size = self.INIT_SIZE
        for i in range(self._size):
            self._map[self._map_shape[0] // 2 - i, self._map_shape[1] // 2] = (
                self.INIT_SIZE - i
            )
        self._food_position=
        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    game = SnakeEngine()
    app.exec_()
