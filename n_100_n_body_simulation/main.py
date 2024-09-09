import sys
import numpy as np
from numpy.linalg import norm
from random import random

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
)
from PyQt5.QtCore import QObject, QThread, QMutex, pyqtSignal, Qt
from PyQt5.QtGui import QKeyEvent, QPixmap, QPainter, QPen


class Body:
    position: np.ndarray
    speed: np.ndarray
    mass: float
    radius: float

    def __init__(
        self, position: np.ndarray, speed: np.ndarray, mass: float, radius: float
    ):
        self.position = position
        self.speed = speed
        self.mass = mass
        self.radius = radius

    def bounce(self, body) -> None:
        if norm(self.position - body.position) < self.radius + body.radius:
            # t_hit = (
            #    -2
            #    * np.dot(self.position - body.position, self.speed - body.speed)
            #    / np.dot(self.speed - body.speed, self.speed - body.speed)
            # )
            hit_direction = (
                body.position - self.position  # + (body.speed - self.speed) * t_hit
            )
            hit_direction /= norm(hit_direction)
            hit_direction_projector = np.tensordot(hit_direction, hit_direction, axes=0)
            hit_direction_orthogonal_projector = np.eye(2) - hit_direction_projector
            v1 = np.dot(hit_direction, self.speed)
            v2 = np.dot(hit_direction, body.speed)
            v1_par = np.dot(hit_direction_orthogonal_projector, self.speed)
            v2_par = np.dot(hit_direction_orthogonal_projector, body.speed)
            v_cdm = (self.mass * v1 + body.mass * v2) / (self.mass + body.mass)
            v1_cdm = v1 - v_cdm
            v2_cdm = v2 - v_cdm
            if v1_cdm < 0 and v2_cdm > 0:
                return  # bodies are not approaching
            kin_en_cdm = self.mass * np.dot(v1_cdm, v1_cdm) + body.mass * np.dot(
                v2_cdm, v2_cdm
            )
            new_v1 = (
                (
                    np.sqrt(
                        kin_en_cdm * body.mass / (self.mass * (self.mass + body.mass))
                    )
                    + v_cdm
                )
                * hit_direction
                * (-1)
                * np.sign(v1)
            )
            new_v2 = (
                (
                    np.sqrt(
                        kin_en_cdm * self.mass / (body.mass * (self.mass + body.mass))
                    )
                    + v_cdm
                )
                * hit_direction
                * (-1)
                * np.sign(v2)
            )
            self.speed = new_v1 + v1_par
            body.speed = new_v2 + v2_par

    def move(self, delta_time, gravity) -> None:
        self.position = self.position + self.speed * delta_time
        self.speed[1] = self.speed[1] + gravity * delta_time

    def rebound_x_min(self, xmin) -> None:
        if self.position[0] - self.radius < xmin:
            self.speed[0] *= -1

    def rebound_x_max(self, xmax) -> None:
        if self.position[0] + self.radius > xmax:
            self.speed[0] *= -1

    def rebound_y_min(self, ymin) -> None:
        if self.position[1] - self.radius < ymin:
            self.speed[1] *= -1

    def rebound_y_max(self, ymax) -> None:
        if self.position[1] + self.radius > ymax:
            self.speed[1] *= -1


class GameEngine(QObject):
    updateGeometrySignal = pyqtSignal(object)

    def __init__(
        self,
        map_shape: np.ndarray,
        integration_time: float,
        nbodies: int,
        max_radius: float,
        gravity: float,
    ):
        QObject.__init__(self)
        self.running = True
        self.runningLock = QMutex()
        self.map_shape = map_shape
        self.integration_time = integration_time
        self.integration_time_millis = int(integration_time * 1000)
        self.gravity = gravity
        self.bodies = []
        for i in range(nbodies):
            # radius = random() * max_radius
            radius = max_radius
            self.bodies.append(
                Body(
                    np.array(
                        [
                            radius + random() * (self.map_shape[0] - 2 * radius),
                            radius + random() * (self.map_shape[1] - 2 * radius),
                        ]
                    ),
                    np.array([0, 0]),
                    radius,
                    radius,
                )
            )
        self.update_dynamics()

    def stillRunning(self):
        self.runningLock.lock()
        value = self.running
        self.runningLock.unlock()
        return value

    def stop(self):
        self.runningLock.lock()
        self.running = False
        self.runningLock.unlock()

    def update_dynamics(
        self,
    ):
        self.runningLock.lock()
        for b in self.bodies:
            b.move(self.integration_time, self.gravity)
            b.rebound_x_min(0)
            b.rebound_y_min(0)
            b.rebound_x_max(self.map_shape[0])
            b.rebound_y_max(self.map_shape[1])
        for i, b1 in enumerate(self.bodies):
            for j, b2 in enumerate(self.bodies):
                if j < i:
                    b1.bounce(b2)
        self.runningLock.unlock()

    def main_cycle(self):
        while self.stillRunning():
            self.update_dynamics()
            self.updateGeometrySignal.emit(self.bodies)
            self.thread().msleep(100)


class MyThread(QThread):
    def __init__(
        self,
        produce_callback: callable,
        integration_time: float,
        map_shape: np.ndarray,
        nbodies: int,
        gravity: float,
        max_radius: float,
    ):
        QThread.__init__(self)
        self.engine = GameEngine(
            integration_time=integration_time,
            gravity=gravity,
            map_shape=map_shape,
            nbodies=nbodies,
            max_radius=max_radius,
        )
        self.engine.moveToThread(self)
        self.started.connect(self.engine.main_cycle)
        self.engine.updateGeometrySignal.connect(produce_callback)

    def stop(self):
        self.engine.stop()


class GameWindow(QMainWindow):
    def __init__(
        self,
    ):
        QMainWindow.__init__(self)
        self.label = QLabel()
        canvas = QPixmap(500, 500)
        canvas.fill(Qt.white)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)

        self.engine_thread = MyThread(
            produce_callback=self.redraw_game_graphics,
            integration_time=0.01,
            max_radius=5,
            map_shape=np.array([500, 500]),
            gravity=300,
            nbodies=100,
        )
        self.engine_thread.start()

    def keyPressEvent(self, keyEvent: QKeyEvent) -> None:
        super(QMainWindow, self).keyPressEvent(keyEvent)
        match keyEvent.key():
            case Qt.Key_D:
                ...
            case Qt.Key_W:
                ...
            case Qt.Key_A:
                ...
            case Qt.Key_S:
                ...
            case Qt.Key_Escape:
                self.engine_thread.stop()
                self.close()
            case Qt.Key_Space:
                ...

    def redraw_game_graphics(self, bodies):
        canvas = self.label.pixmap()
        canvas.fill(Qt.darkGray)
        painter = QPainter(canvas)
        pen = QPen()
        pen.setColor(Qt.black)
        painter.setPen(pen)
        for b in bodies:
            painter.drawEllipse(int(b.position[0]) - 5, int(b.position[1]) - 5, 10, 10)
        painter.end()
        self.update()


if __name__ == "__main__":
    # b1 = Body(np.array([0, 0]), np.array([1, 0]), 1, 10)
    # b2 = Body(np.array([23, 1]), np.array([-1, 0]), 2, 12)
    # for i in range(500):
    #    b1.bounce(b2)
    #    b1.move(0.01, 0)
    #    b2.move(0.01, 0)
    #    print(
    #        f"{b1.position[0]:2f} - {b1.position[1]:2f} - {b2.position[0]:2f} - {b2.position[1]:2f} - min dist {norm(b1.position-b2.position)-b1.radius-b2.radius}"
    #    )
    #
    app = QApplication(sys.argv)
    win = GameWindow()
    win.show()
    app.exec_()
