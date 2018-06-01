# -*- coding: utf-8 -*-
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical


def main():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1),
                            strides=1, padding='same', kernel_initializer='he_normal'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            strides=1, padding='same', kernel_initializer='he_normal'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    print(model.summary())


    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255.0

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255.0

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.fit(train_images, train_labels, epochs=20, batch_size=100)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test loss:{}, acc:{}'.format(test_loss, test_acc))


if __name__ ==  '__main__':
    main()

