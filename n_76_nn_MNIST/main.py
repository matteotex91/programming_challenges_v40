import tensorflow as tf
import numpy as np
import os


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
            epochs = 3  # set to 5
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
                        print("\nReached 99.5% accuracy so cancelling training!")
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
        return max(self.model.predict(np.array([image])))


n = NeuralEngine(os.getcwd() + "/n_76_nn_MNIST/model/mnist_fit.h5")
test = np.array([])
n.predict
