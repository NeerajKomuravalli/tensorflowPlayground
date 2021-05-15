import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.core import Flatten


class customCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if logs.get("accuracy") > 0.93:
            print("Stopping training because accuracy is more than 85%")
            self.model.stop_training = True


if __name__ == "__main__":
    callback = customCallBack()
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    train_images = train_images.reshape(60000, 28, 28, 1)
    train_images = train_images/255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images/255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam', 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )

    model.fit(train_images, train_labels, epochs=5, callbacks=[callback])

    model.evaluate(test_images, test_labels)