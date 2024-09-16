import tensorflow as tf
import numpy as np
import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QKeyEvent, QPixmap, QPainter, QPen, QMouseEvent
import sys

LOW_TONE = 255  # was 100
MID_TONE = 255  # was 175
HIG_TONE = 255  # was 255


class NeuralEngine:

    def __init__(self, path: str = None):
        if path is not None:
            self.model = tf.keras.models.load_model(path)
        else:
            mnist = tf.keras.datasets.mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            input_shape = (28, 28, 1)
            x_train = x_train.reshape(
                x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
            )
            x_train = x_train / 255.0
            x_test = x_test.reshape(
                x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
            )
            x_test = x_test / 255.0
            y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
            y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
            batch_size = 64
            num_classes = 10
            epochs = 5  # set to 5
            model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        32,
                        (5, 5),
                        padding="same",
                        activation="relu",
                        input_shape=input_shape,
                    ),
                    tf.keras.layers.Conv2D(
                        32, (5, 5), padding="same", activation="relu"
                    ),
                    tf.keras.layers.MaxPool2D(),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Conv2D(
                        64, (3, 3), padding="same", activation="relu"
                    ),
                    tf.keras.layers.Conv2D(
                        64, (3, 3), padding="same", activation="relu"
                    ),
                    tf.keras.layers.MaxPool2D(strides=(2, 2)),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(num_classes, activation="softmax"),
                ]
            )
            model.compile(
                optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08),
                loss="categorical_crossentropy",
                metrics=["acc"],
            )

            class myCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs={}):
                    if logs.get("acc") > 0.995:
                        self.model.stop_training = True

            callbacks = myCallback()

            model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.1,
                callbacks=[callbacks],
            )

            self.model = model

    def predict(self, image: np.ndarray) -> int:
        wrapped_image = np.zeros((1, *image.shape, 1))
        wrapped_image[0, :, :, 0] = image.T / 255.0
        prediction = self.model.predict(wrapped_image)
        # print(prediction)
        return np.argmax(prediction[0])


class GameWindow(QMainWindow):
    def __init__(
        self,
        nn_model_path: str,
        map_shape: np.ndarray = np.array([28, 28]),
        pixel_shape: np.ndarray = np.array([20, 20]),
        pixel_offset: int = 2,
    ):
        QMainWindow.__init__(self)
        self.neural_engine = NeuralEngine(nn_model_path)
        self.map_shape = map_shape
        self.map = np.zeros(map_shape)
        self.pixel_shape = pixel_shape
        self.pixel_offset = pixel_offset
        self.label = QLabel()
        self.setFixedSize(*(map_shape * pixel_shape))
        canvas = QPixmap(*(map_shape * pixel_shape))
        canvas.fill(Qt.lightGray)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)
        self.redraw_game_graphics()
        self.clicked = False

    def redraw_game_graphics(self):
        canvas = self.label.pixmap()
        canvas.fill(Qt.lightGray)
        painter = QPainter(canvas)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(Qt.darkGray)
        painter.setPen(pen)
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                painter.drawRect(
                    i * self.pixel_shape[0],
                    j * self.pixel_shape[1],
                    self.pixel_shape[0],
                    self.pixel_shape[1],
                )
                color = None
                match self.map[i, j]:
                    case 100:
                        color = Qt.lightGray
                    case 175:
                        color = Qt.gray
                    case 255:
                        color = Qt.darkGray
                if color is not None:
                    painter.fillRect(
                        i * self.pixel_shape[0] + self.pixel_offset,
                        j * self.pixel_shape[1] + self.pixel_offset,
                        self.pixel_shape[0] - 2 * self.pixel_offset,
                        self.pixel_shape[1] - 2 * self.pixel_offset,
                        color,
                    )

        prediction = int(self.neural_engine.predict(self.map))
        painter.drawText(
            QRect(
                0 * self.pixel_shape[0] + self.pixel_offset,
                0 * self.pixel_shape[1] + self.pixel_offset,
                self.pixel_shape[0] - 2 * self.pixel_offset,
                self.pixel_shape[1] - 2 * self.pixel_offset,
            ),
            0,
            str(prediction),
        )
        painter.end()
        self.update()

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        self.clicked = True

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        self.clicked = False

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        if self.clicked:
            x = int(a0.x() / self.pixel_shape[0])
            y = int(a0.y() / self.pixel_shape[1])
            self.map[x, y] = HIG_TONE
            if x > 0 and y > 0:
                self.map[x - 1, y - 1] = max(LOW_TONE, self.map[x - 1, y - 1])
            if x < 28 and y < 28:
                self.map[x + 1, y + 1] = max(LOW_TONE, self.map[x + 1, y + 1])
            if x > 0 and y < 28:
                self.map[x - 1, y + 1] = max(LOW_TONE, self.map[x - 1, y + 1])
            if x < 28 and y > 0:
                self.map[x + 1, y - 1] = max(LOW_TONE, self.map[x + 1, y - 1])
            if x > 0:
                self.map[x - 1, y] = max(MID_TONE, self.map[x - 1, y])
            if x < 28:
                self.map[x + 1, y] = max(MID_TONE, self.map[x + 1, y])
            if y > 0:
                self.map[x, y - 1] = max(MID_TONE, self.map[x, y - 1])
            if y < 28:
                self.map[x, y + 1] = max(MID_TONE, self.map[x, y + 1])
            self.redraw_game_graphics()

    def keyReleaseEvent(self, a0: QKeyEvent | None) -> None:
        match a0.key():
            case Qt.Key_Space:
                self.map = np.zeros(self.map_shape)
                self.redraw_game_graphics()


if __name__ == "__main__":
    nn_model_path = os.getcwd() + "/n_76_nn_MNIST/model/mnist_fit.h5"
    app = QApplication(sys.argv)
    win = GameWindow(nn_model_path=nn_model_path)
    win.show()
    app.exec_()
