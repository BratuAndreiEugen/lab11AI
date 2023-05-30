import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import tensorflow as tf

from feat import Detector


def emotions():
    # Set the path to the dataset directory
    data_dir = "C:\Proiecte SSD\Python\lab11AI\data\emotions"

    # Set the paths to the train and test directories
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # Set the image size, batch size, and number of classes
    image_size = (48, 48)
    batch_size = 32
    num_classes = 7

    # Create data generators for train and test sets
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    # Build the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=10
    )

    # Evaluate the model on the test set
    _, accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
    print("Test Accuracy:", accuracy)


def extract_photo_paths(directory):
    photo_paths = []
    photo_extensions = ['.jpg', '.jpeg', '.png', '.gif']  # Add more extensions if needed

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()  # Get the file extension in lowercase
            if file_extension in photo_extensions:
                photo_path = os.path.join(root, file)
                photo_paths.append(photo_path)

    return photo_paths

def emotionsAutomatedExtract(): # rezolva si partea de extragere automata si partea de pre train
    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model='xgb',
        emotion_model="resmasknet",
        facepose_model="img2pose",
    )

    from feat.utils.io import get_test_data_path
    from feat.plotting import imshow
    import os

    # Helper to point to the test data folder
    test_data_dir = "C:\Proiecte SSD\Python\lab11AI\data\emotions\\test"

    #detector.detect_faces(single_face_img_path)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    accuracy = 0
    length = 0
    i = 0
    for emotion in emotions:
        em_path = os.path.join(test_data_dir, emotion)
        paths = extract_photo_paths(em_path)
        only10 = 0
        for path in paths:
            length +=1
            single_face_prediction = detector.detect_image(path)
            print(single_face_prediction.features)
            #print(single_face_prediction.emotions)
            print(single_face_prediction.emotions)
            v = np.argmax(single_face_prediction.emotions.values.tolist())
            print(v)
            if v == i:
                accuracy += 1
            only10+=1
            if only10 == 10:
                break
        i+=1

    print("Accuracy : ", accuracy/length)

def emotionsOpenCV():
    image_path = 'path/to/image.jpg'  # Replace with the path to your input image
    image = cv2.imread(image_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))