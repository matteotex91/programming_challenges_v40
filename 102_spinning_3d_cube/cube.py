import sys
import numpy as np
from numpy.linalg import norm

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
)
from PyQt5.QtCore import QObject, QThread, QMutex, pyqtSignal, Qt
from PyQt5.QtGui import QKeyEvent, QPixmap, QPainter, QPen


class GameEngine(QObject):
    updateGeometrySignal = pyqtSignal(object)
    CUBE_VERTEX = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, -1],
            [1, -1, 1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, -1],
            [-1, -1, 1],
        ]
    ).T

    CUBE_EDGES_MAP = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )

    def __init__(self, init_ang_speed: np.ndarray, integration_time: float):
        QObject.__init__(self)
        self.running = True
        self.runningLock = QMutex()
        self.angular_speed = init_ang_speed
        self.cube_ref_sys = np.eye(3)
        self.cube_vertex = np.copy(self.CUBE_VERTEX)
        self.integration_time = integration_time
        self.integration_time_millis = int(integration_time * 1000)
        self.update_rotation_matrix()

    def stillRunning(self):
        self.runningLock.lock()
        value = self.running
        self.runningLock.unlock()
        return value

    def stop(self):
        self.runningLock.lock()
        self.running = False
        self.runningLock.unlock()

    def stop_rotation(self):
        self.runningLock.lock()
        self.angular_speed = np.array([0, 0, 0.000000000001])
        self.update_rotation_matrix()
        self.cube_vertex = np.copy(self.CUBE_VERTEX)
        self.runningLock.unlock()

    def change_angular_speed(self, delta_ang_speed):
        self.runningLock.lock()
        self.angular_speed += np.matmul(self.rot_mat.T, delta_ang_speed)
        self.runningLock.unlock()
        self.update_rotation_matrix()

    def update_rotation_matrix(
        self,
    ):
        theta = norm(self.angular_speed) * self.integration_time
        ang_speed_versor = self.angular_speed / norm(self.angular_speed)
        self.rot_mat = (
            np.cos(theta) * np.eye(3)
            + (1 - np.cos(theta))
            * np.tensordot(ang_speed_versor, ang_speed_versor, axes=0)
            + np.sin(theta)
            * np.array(
                [
                    [0, -ang_speed_versor[2], ang_speed_versor[1]],
                    [ang_speed_versor[2], 0, -ang_speed_versor[0]],
                    [-ang_speed_versor[1], ang_speed_versor[0], 0],
                ]
            )
        )

    def update_coordinates(self):
        self.runningLock.lock()
        self.cube_vertex = np.matmul(self.rot_mat, self.cube_vertex)
        self.runningLock.unlock()

    def main_cycle(self):
        while self.stillRunning():
            self.update_coordinates()
            self.updateGeometrySignal.emit(self.cube_vertex)
            self.thread().msleep(self.integration_time_millis)


class MyThread(QThread):
    def __init__(
        self,
        produce_callback: callable,
        init_ang_speed: np.ndarray,
        integration_time: float,
    ):
        QThread.__init__(self)
        self.engine = GameEngine(
            init_ang_speed=init_ang_speed, integration_time=integration_time
        )
        self.engine.moveToThread(self)
        self.started.connect(self.engine.main_cycle)
        self.engine.updateGeometrySignal.connect(produce_callback)

    def stop(self):
        self.engine.stop()


class GameWindow(QMainWindow):
    def __init__(
        self,
        fov: float = 1,
        view_plane_x_position: float = 3,
        init_ang_speed: np.ndarray = np.array([0, 0, 1.0]),
        integration_time: float = 0.01,
    ):
        QMainWindow.__init__(self)
        self.fov = fov
        self.view_plane_x_position = view_plane_x_position
        self.label = QLabel()
        canvas = QPixmap(500, 500)
        canvas.fill(Qt.white)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)

        self.engine_thread = MyThread(
            produce_callback=self.redraw_game_graphics,
            init_ang_speed=init_ang_speed,
            integration_time=integration_time,
        )
        self.engine_thread.start()

    def keyPressEvent(self, keyEvent: QKeyEvent) -> None:
        super(QMainWindow, self).keyPressEvent(keyEvent)
        match keyEvent.key():
            case Qt.Key_D:
                self.engine_thread.engine.change_angular_speed(np.array([0, 0, -0.2]))
            case Qt.Key_W:
                self.engine_thread.engine.change_angular_speed(np.array([0, -0.2, 0]))
            case Qt.Key_A:
                self.engine_thread.engine.change_angular_speed(np.array([0, 0, 0.2]))
            case Qt.Key_S:
                self.engine_thread.engine.change_angular_speed(np.array([0, 0.2, 0]))
            case Qt.Key_Escape:
                self.engine_thread.stop()
                self.close()
            case Qt.Key_Space:
                self.engine_thread.engine.stop_rotation()

    def project_vector(self, v):
        return (
            np.array([v[1], v[2]])
            * self.fov
            / (v[0] - self.view_plane_x_position - self.fov)
        )

    OFFSET = 250
    MAGNIFICATION = 400

    def redraw_game_graphics(self, cube_vertex):
        canvas = self.label.pixmap()
        canvas.fill(Qt.darkGray)
        painter = QPainter(canvas)
        pen = QPen()
        pen.setWidth(5)
        pen.setColor(Qt.magenta)
        painter.setPen(pen)
        projected_vectors = self.project_vector(cube_vertex)
        for line in self.engine_thread.engine.CUBE_EDGES_MAP:
            p1 = projected_vectors.T[line[0]]
            p2 = projected_vectors.T[line[1]]
            painter.drawLine(
                int(self.OFFSET + self.MAGNIFICATION * p1[0]),
                int(self.OFFSET + self.MAGNIFICATION * p1[1]),
                int(self.OFFSET + self.MAGNIFICATION * p2[0]),
                int(self.OFFSET + self.MAGNIFICATION * p2[1]),
            )

        painter.end()
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GameWindow()
    win.show()
    app.exec_()
