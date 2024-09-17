# libraries needed
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(f"Train Images:{train_images.shape}")
print(f"Train Labels:{len(train_labels)}")
# Train Images:(60000, 28, 28)
# Train Labels:60000


# set a proper figure dimension
plt.figure(figsize=(10, 10))

# pick 36 random digits in range 0-59999
# inner bound is inclusive, outer bound exclusive
random_inds = np.random.choice(60000, 36)

flag_plot = False
if flag_plot:
    for i in range(36):
        plt.subplot(6, 6, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image_ind = random_inds[i]
        # show images using a binary color map (i.e. Black and White only)
        plt.imshow(train_images[image_ind], cmap=plt.cm.binary)
        # set the image label
        plt.xlabel(train_labels[image_ind])
    plt.show()

    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

train_images = (np.expand_dims(train_images, axis=-1) / 255.0).astype(np.float32)
train_labels = (train_labels).astype(np.int64)
test_images = (np.expand_dims(test_images, axis=-1) / 255.0).astype(np.float32)
test_labels = (test_labels).astype(np.int64)


def build_fc_model():
    fc_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return fc_model


model = build_fc_model()

learning_rate = 1e-1
optimizer = tf.keras.optimizers.SGD(learning_rate)
loss = "sparse_categorical_crossentropy"
metrics = ["accuracy"]

model.compile(
    optimizer,
    loss,
    metrics,
)

batch_size = 64
epochs = 10

model.fit(
    train_images,
    train_labels,
    batch_size,
    epochs,
)

# test_loss, test_acc = model.evaluate(test_images, test_labels)

print("stop here")
