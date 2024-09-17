import tensorflow as tf
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from matplotlib.backend_bases import KeyEvent

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# sns.countplot(y_train)

input_shape = (28, 28, 1)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test = x_test / 255.0

y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)


batch_size = 64
num_classes = 10
epochs = 5  # set to 5

test_index = 0
fig, ax = plt.subplots()


def on_release(event: KeyEvent):
    global fig, ax, x_test, test_index, y_test
    if event.key == "left":
        test_index = test_index - 1
    elif event.key == "right":
        test_index = test_index + 1
    ax.pcolormesh(x_test[test_index, :, :, 0])
    ax.set_title(str(y_test[test_index]))
    fig.show()


conn_id = fig.canvas.mpl_connect("key_release_event", on_release)
plt.show()
fig.canvas.mpl_disconnect(conn_id)


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (5, 5), padding="same", activation="relu", input_shape=input_shape
        ),
        tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
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

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    callbacks=[callbacks],
)


test_loss, test_acc = model.evaluate(x_test, y_test)


# Predict the values from the testing dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)
# Convert testing observations to one hot vectors
Y_true = np.argmax(y_test, axis=1)
# compute the confusion matrix
confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes)


savepath = os.getcwd() + "/n_76_nn_MNIST/model/mnist_fit.h5"
model.save(savepath)

# to load model : new_model = tf.keras.models.load_model('saved_model/my_model')

print("stop here")
