import numpy as np

from feat import Detector
from iteratia3.Emotions import extract_photo_paths


def multiLabel():
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
    test_data_dir = "C:\Proiecte SSD\Python\lab11AI\data\multilabel"
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    paths = extract_photo_paths(test_data_dir)
    print(paths)

    i = 1
    for path in paths:
        print(path)
        prediction = detector.detect_image(path)
        print(prediction.emotions)
        l = []
        for face_predict in prediction.emotions.values:
            l.append(emotions[np.argmax(face_predict)])
        l = set(l)
        print("Detected emotions : ", l)
        i+=1