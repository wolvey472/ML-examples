import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import os
import time as t

import random as rd


if tf.__version__:
    print(f"WORKS VERSION: {tf.__version__}")
else:
    print("SUM WENT WRONG") # liklely not on 3.12

data = tf.keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = data.load_data()
#  X   -> images
#  y   -> labels

#Label	Class
#0	 T-shirt/top
#1	 Trouser
#2	 Pullover
#3	 Dress
#4	 Coat
#5	 Sandal
#6	 Shirt
#7	 Sneaker
#8	 Bag
#9	 Ankle boot

print(X_train.shape)
#60000, 28, 28

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot():
    plt.figure()
    plt.imshow(X_train[0])
    plt.colorbar()
    plt.grid(visible=False)
    plt.legend()

    plt.text(x=5, y=-2, s=y_train[0])
    plt.show()
    return None





X_train = X_train / 255.0 #scale values to range from 0-1
X_test = X_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #unstacks pixels by taking 28 rows by 28 colums
    tf.keras.layers.Dense(128, activation='relu'), #128 neurons
    tf.keras.layers.Dense(10) # 0-9 neurons for y output
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)
print(f"{test_acc * 100}")

csv_logger = tf.keras.callbacks.CSVLogger(
    "training_log.csv",
    append=True
)

try:
    while True:
        model.fit(
            X_train,
            y_train,
            epochs=1,
            validation_data=(X_test, y_test),
            callbacks=[csv_logger]
        )

except KeyboardInterrupt:
    print("Stopped training.")
