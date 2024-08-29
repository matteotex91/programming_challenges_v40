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
    QWidget,
)
from PyQt5.QtGui import QBrush, QPen
from PyQt5.QtCore import (
    Qt,
    QRunnable,
    pyqtSlot,
    QThreadPool,
    QObject,
    QMutex,
    pyqtSignal,
    QThread,
)
from PyQt5.QtGui import QKeyEvent, QCloseEvent


class ColorEngine(QObject):
    def __init__(self):
        QObject.__init__(self)
        self.running = True
        self.runningLock = QMutex()
        self.colorProducerSignal = pyqtSignal(object)

    # Might be called from both threads.
    def stop(self):
        self.runningLock.lock()
        self.running = False
        self.runningLock.unlock()

    # Might be called from both threads.
    def stillRunning(self):
        self.runningLock.lock()
        value = self.running
        self.runningLock.unlock()
        return value

    def main(self):
        index = 0
        while self.stillRunning:
            if index % 2 == 0:
                self.colorProducerSignal.emit(Qt.red)
            else:
                self.colorProducerSignal.emit(Qt.blue)
            self.thread().msleep(1000)
        self.thread().quit()


class MyThread(QThread):
    def __init__(self, produce_callback):
        self.engine = ColorEngine()
        self.engine.moveToThread(self)
        self.started.connect(self.engine.main)
        self.engine.colorProducerSignal.connect(produce_callback)

    def stop(self):
        self.engine.stop()


class MyScene(QWidget):
    def __init__(self):
        QWidget.__init__(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    game = GraphicsView()
    app.exec_()
