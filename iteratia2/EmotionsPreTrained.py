import os

import numpy as np
from keras import Sequential
from keras.applications import VGG16
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras import backend as K
from keras.models import load_model

def preTrained():
    data_dir = "C:\Proiecte SSD\Python\lab11AI\data\emotions"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    emotions = os.listdir(train_dir)
    num_classes = len(emotions)
    image_size = 48
    batch_size = 32

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True
    )

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False
    )

    pretrained_model = VGG16(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))

    model = Sequential()
    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    # Freeze the pre-trained layers
    for layer in pretrained_model.layers:
        layer.trainable = False

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        train_data,
        steps_per_epoch=train_data.samples // batch_size,
        epochs=10
    )

    test_data.reset()  # Reset the test_data generator
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)

    true_labels = test_data.classes
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("Accuracy:", accuracy)


def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 1

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))

def ferPreTrained():
    data_dir = "C:\Proiecte SSD\Python\lab11AI\data\emotions"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    emotions = os.listdir(train_dir)
    num_classes = len(emotions)
    image_size = 48
    batch_size = 32

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False
    )

    model = load_model("C:\Proiecte SSD\Python\lab11AI\\facenet\\model.h5")

    # Make predictions on the preprocessed image(s)
    predictions = model.predict(test_data)

    # Get the predicted emotion class for each image
    predicted_classes = tf.argmax(predictions, axis=1)

    # Map the predicted class indices to actual emotion labels
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    predicted_emotions = [emotion_labels[prediction] for prediction in predicted_classes]
    print(predicted_emotions)