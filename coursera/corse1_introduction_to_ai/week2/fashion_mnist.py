import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.python.ops.gen_math_ops import mod

class customCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if logs.get("accuracy") > 0.85:
            print("Stopping training because accuracy is more than 85%")
            self.model.stop_training = True


if __name__ == "__main__":
    # Load data
    mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # Noramalize data
    train_images = train_images/255.0
    test_images = test_images/255.0
    
    # Callback 
    callback = customCallBack()

    # Declare model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Train
    model.fit(train_images, train_labels, epochs=5, callbacks=[callback])

    # Evaluate
    model.evaluate(test_images, test_labels)

    # Predict
    classification = model.predict(test_images)

    print("lable : {} :: prediction : {}".format(test_labels[0], classification[0]))
