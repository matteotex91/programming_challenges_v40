import tensorflow as tf
import tensorflow_datasets as tfds
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

import matplotlib.pyplot as plt


LOW_TONE = 50  # was 100
MID_TONE = 150  # was 175
HIG_TONE = 255  # was 255


class NeuralEngine:

    def __init__(self, path: str = None):
        if path is not None:
            self.model = tf.keras.models.load_model(path)
        else:
            (ds_train, ds_test), ds_info = tfds.load(
                "mnist",
                split=["train", "test"],
                shuffle_files=True,
                as_supervised=True,
                with_info=True,
            )

            def normalize_img(image, label):
                """Normalizes images: `uint8` -> `float32`."""
                return tf.cast(image, tf.float32) / 255.0, label

            ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
            ds_train = ds_train.cache()
            ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
            ds_train = ds_train.batch(128)
            ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

            ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
            ds_test = ds_test.batch(128)
            ds_test = ds_test.cache()
            ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

            model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Flatten(input_shape=(28, 28)),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dense(10),
                ]
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            )

            model.fit(
                ds_train,
                epochs=6,
                validation_data=ds_test,
            )

            self.model = model

    def predict(self, image: np.ndarray) -> int:
        wrapped_image = np.zeros((1, *image.shape, 1))
        wrapped_image[0, :, :, 0] = image / 255.0
        prediction = self.model.predict(wrapped_image)
        # print(prediction)
        return np.argmax(prediction[0])


class GameWindow(QMainWindow):
    def __init__(
        self,
        nn_model_path: str = None,
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
                    case 50:
                        color = Qt.lightGray
                    case 150:
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

        prediction = int(self.neural_engine.predict(self.map.T))
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
            if x < 0 or x > 27 or y < 0 or y > 27:
                return
            self.map[x, y] = HIG_TONE
            if x > 0 and y > 0:
                self.map[x - 1, y - 1] = max(LOW_TONE, self.map[x - 1, y - 1])
            if x < 27 and y < 27:
                self.map[x + 1, y + 1] = max(LOW_TONE, self.map[x + 1, y + 1])
            if x > 0 and y < 27:
                self.map[x - 1, y + 1] = max(LOW_TONE, self.map[x - 1, y + 1])
            if x < 27 and y > 0:
                self.map[x + 1, y - 1] = max(LOW_TONE, self.map[x + 1, y - 1])
            if x > 0:
                self.map[x - 1, y] = max(MID_TONE, self.map[x - 1, y])
            if x < 27:
                self.map[x + 1, y] = max(MID_TONE, self.map[x + 1, y])
            if y > 0:
                self.map[x, y - 1] = max(MID_TONE, self.map[x, y - 1])
            if y < 27:
                self.map[x, y + 1] = max(MID_TONE, self.map[x, y + 1])
            self.redraw_game_graphics()

    def keyReleaseEvent(self, a0: QKeyEvent | None) -> None:
        match a0.key():
            case Qt.Key_Space:
                self.map = np.zeros(self.map_shape)
                self.redraw_game_graphics()
            case Qt.Key_Tab:
                plt.pcolormesh(self.map.T / 255.0)
                plt.show()


if __name__ == "__main__":
    nn_model_path = os.getcwd() + "/n_76_nn_MNIST/model/mnist_fit.h5"
    app = QApplication(sys.argv)
    # win = GameWindow(nn_model_path=nn_model_path)
    win = GameWindow()
    win.show()
    app.exec_()
