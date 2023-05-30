import os
import random
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def emoji():
    # Set the paths to your happy and sad folders
    happy_folder = "C:\Proiecte SSD\Python\lab11AI\data\emoji\happy"
    sad_folder = "C:\Proiecte SSD\Python\lab11AI\data\emoji\sad"

    data = []
    labels = []

    positive_images = os.listdir(happy_folder)
    for image_file in positive_images:
        image_path = os.path.join(happy_folder, image_file)
        image = Image.open(image_path)
        image_array = np.array(image)
        data.append(image_array)
        labels.append(1)  # Happy label is 1

    negative_images = os.listdir(sad_folder)
    for image_file in negative_images:
        image_path = os.path.join(sad_folder, image_file)
        image = Image.open(image_path)
        image_array = np.array(image)
        data.append(image_array)
        labels.append(0)  # Sad label is 0

    # data split
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # flatten
    x_train = np.array(x_train).reshape(len(x_train), -1)
    x_test = np.array(x_test).reshape(len(x_test), -1)

    # support vector machine (SVM) classifier
    #classifier = SVC()
    # multiple decision trees
    classifier = RandomForestClassifier()

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    while 1:
        link = input("Link catre poza de test : ")
        if link == "exit":
            return
        img = Image.open(link)
        img.show()
        arr = np.array(img)
        d = arr.reshape(1, -1)
        pred = classifier.predict(d)
        if pred[0] == 0:
            print("SAD")
        else:
            print("HAPPY")