from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QObject, QThread, QMutex, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor


import random
import sys


# -------------------------------------------------------------------------------------


# Here we create the Worker Object. Everything inside `Worker.main(self)` will be
# executed in another thread: ChangeColorThread.
#
# As we might inspect / change the value of `Worker.running` on both threads at the same time
# (Main Thread and ChangeColorThread), I recommend using a QMutex to restrict access
# to this variable.
class Worker(QObject):
    # Note the signal possesses a `object` as a parameter. It signalizes the QObject
    # that you want to pass an object type to the signal's receiver.
    #
    # In this case, it's a tuple (red, green, blue) as `Scene.setRandomColor` has the
    # `color` parameter.
    produce = pyqtSignal(object)

    # Called from main thread only to construct the Worker instance.
    def __init__(self, delay):
        QObject.__init__(self)
        self.running = True
        self.delay = delay
        self.runningLock = QMutex()

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

    # Executes on ChangeColorThread only.
    def main(self):
        while self.stillRunning():
            # Generate the random colors from the ChangeColorThread.
            red = random.randint(0, 255)
            green = random.randint(0, 255)
            blue = random.randint(0, 255)

            # Emit the `produce` signal to safely call the `Scene.setRandomColor()`
            # on the Main Thread.
            #
            # In order to update any GUI or call any QWidget method, you must emit
            # a connected signal to the main thread, so you don't raise any exceptions
            # or segmentation faults (or worse, silent crashes).
            self.produce.emit((red, green, blue))

            # Tell the ChangeColorThread to sleep for delay microseconds
            # (aka a value of 1000 == 1 second)
            self.thread().msleep(self.delay)

        print("Quit from ChangeColorThread Worker.main()")
        self.thread().quit()


# Here we have the other thread we will instantiate. The thread by itself
# is nothing special. It acts like a QObject until `QThread.start()` is
# called. It also has a few important properties you might want to take a look:
#    - started (calls one or more connected functions once the thread starts executing).
#    - finished (calls one or more connected functions once the thread properly finishes).
#
# When `start()` is called, as we connected `self.started` to `Worker.main`,
# `Worker.main` will start executing on the other thread until a `stop` call
# is requested.
class ChangeColorThread(QThread):
    def __init__(self, produce_callback, delay=100):
        QThread.__init__(self)

        self.worker = Worker(delay)
        self.worker.moveToThread(self)
        self.started.connect(self.worker.main)
        self.worker.produce.connect(produce_callback)

    # Stops the worker object's main loop from the main thread.
    def stop(self):
        self.worker.stop()


# Here we will create the GUI. I kept it simple, it contains a single QLabel.
#
# It also starts the ChangeColorThread thread.
class Scene(QLabel):
    def __init__(self):
        QLabel.__init__(self)

        canvas = QPixmap(200, 200)
        canvas.fill(Qt.white)
        self.setPixmap(canvas)

        self.thread = ChangeColorThread(self.setRandomColor, 500)
        self.thread.start()

    # This function will execute on the Main Thread only.
    #
    # However, this function is not called by the user. It is scheduled to execute at some point
    # by PyQt5 once `Worker.produce` is emitted.
    def setRandomColor(self, color):
        # self.label.setStyleSheet(
        #    "background-color:rgb(%d,%d,%d)" % (color[0], color[1], color[2])
        # )
        painter = QPainter(self.pixmap())

        pen = QPen()
        pen.setWidth(4)
        pen.setColor(QColor(color[0], color[1], color[2]))
        painter.setPen(pen)
        painter.drawRect(50, 50, 50, 50)
        painter.end()
        self.update()

    # As we don't know wether ChangeColorThread has stopped or not, we signalize
    # it to stop before closing the application.
    #
    # This step is important, because even if PyQt5 / PySide2 tries its best to close the
    # remaining thread, once the window is closed, this process can fail. The result is a
    # hidden process still running even after the program is closed.
    def sceneInterceptCloseEvent(self, evt):
        self.thread.stop()
        self.thread.wait()
        evt.accept()


# Here we create the Main Window. It contains a single scene with the QLabel inside it.
class Window(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.scene = Scene()
        self.setCentralWidget(self.scene)

    # We call Scene.sceneInterceptCloseEvent in order to signalize the running thread
    # to stop its main loop before the application is closed.
    def closeEvent(self, evt):
        self.scene.sceneInterceptCloseEvent(evt)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    app.exec_()
