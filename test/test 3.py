'''
CLEARING THE ARRAY AFTER EACH PREDICTION
'''

from cv2 import VideoCapture, waitKey, imshow, cvtColor, flip, COLOR_BGR2RGB, COLOR_RGB2BGR, putText, FONT_HERSHEY_SIMPLEX, resize
from numpy import ndarray, array, argmax, expand_dims
from pathlib import Path
from shutil import rmtree
from tensorflow.keras.models import load_model
from collections import deque

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

from ASL_UTILS.extractor import MediapipeExtractor
from ASL_UTILS.collection import ImageHandler

cap = VideoCapture(0)
me = MediapipeExtractor()
im = ImageHandler()

out_dir = Path('test')
if out_dir.exists(): rmtree(out_dir)
out_dir.mkdir()

DQ = deque(maxlen=5)

Q = []
Q_limit = 40
Q_status = 0
klasses = ['before', 'cool', 'hands_down']


QR = []
QR_limit = 35
QR_status = 0

model = 'Model/IDB_3_95'
model = load_model(str(model))


text1 = ''

with mp_holistic.Holistic(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as holistic:

    while cap.isOpened():
        
        image: ndarray
        success, image = cap.read()

        if not success: continue

        image = flip(image, 1)

        image = cvtColor(image, COLOR_BGR2RGB)

        image.flags.writeable = False
        results = holistic.process(image)

        image = cvtColor(image, COLOR_RGB2BGR)
        image = im.draw_results(image, results)

        landmarks = me.extract_landmarks(results)


        if QR_status == QR_limit:

            QR_vid = expand_dims(array(QR), 0)
            results = model.predict(QR_vid)

            DQ.append(results[0])
            results = array(DQ).mean(axis=0)

            predicted_kls = klasses[results.argmax(axis=0)]
            text1 = f"{predicted_kls}"

            QR_status = 0
            QR = []
        else:

            QR.append(landmarks)
            QR_status += 1

            text2 = f"{'|'*QR_status}"


        image = resize(image, (1280, 1024))
        image = putText(image, text1, (30, image.shape[0] - 90), FONT_HERSHEY_SIMPLEX, 2, (11,209,252), 3)
        image = putText(image, text2, (30, image.shape[0] - 30), FONT_HERSHEY_SIMPLEX, 2, (11,209,252), 3)
            
        imshow('out', image)

        if waitKey(5) & 0xFF == 27: break # 27 is escape!

    cap.release()