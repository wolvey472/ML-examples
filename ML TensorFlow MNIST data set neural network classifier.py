import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

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
print(f"{test_acc * 100}:,.2f")

def predictions():
    for i in range(2):
        plt.figure()
        plt.imshow(X_test[i])
        plt.colorbar()
        plt.grid(visible=False)

        probability_model = tf.keras.Sequential([model,
                                    tf.keras.layers.Softmax()])
        predictions = probability_model.predict(X_test)
        plt.title(f"Prediction: {np.argmax(predictions[i])}, Actual: {y_test[i]}")
        plt.show()
        t.sleep(1)
        plt.close(fig="all")
    

predictions()

def plot_incorrect(choice, predicted):
    plt.figure()
    plt.imshow(X_test[choice], cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)

    actual_label = y_test[choice]

    actual_name = class_names[actual_label]
    predicted_name = class_names[predicted]

    plt.title(f"Prediction: {predicted_name}, Actual: {actual_name}")
    plt.show()
    rand_test()


def rand_test():
    while True:
        choice = rd.randint(0, 9999)

        probability_model = tf.keras.Sequential([
            model,
            tf.keras.layers.Softmax()
        ])

        predictions = probability_model.predict(X_test)

        actual_label = y_test[choice]
        predicted_label = np.argmax(predictions[choice])

        actual_name = class_names[actual_label]
        predicted_name = class_names[predicted_label]

        print(f"Image index: {choice}")
        print(f"Actual: {actual_name}")
        print(f"Prediction: {predicted_name}")

        if actual_label != predicted_label:
            print("Model was wrong.")
            plot_incorrect(choice, predicted_label)
            break
        else:
            print("Model was correct.")



rand_test()
