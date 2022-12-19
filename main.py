import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')


(training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

# training_images = training_images[:20000] # you can save a lot of time [delete this 4 rows]
# training_labels = training_labels[:20000]
# testing_images = testing_images[:4000]
# testing_labels = testing_labels[:4000]
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
#
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(training_images, training_labels, epochs=20, validation_data=(testing_images, testing_labels))
#
# loss = model.evaluate(testing_images, testing_labels)
# print('Test loss:', loss)
# accuracy = model.evaluate(testing_images, testing_labels)
# print('Test accuracy:', accuracy)
#
# model.save('image_classifier.model-19-12-2022')
model = tf.keras.models.load_model('image_classifier.model-19-12-2022')

img = cv2.imread('deer.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')

plt.show()
