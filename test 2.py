'''
REMOVING 1 FRAME APPENDING 1 FRAME
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
# klasses = ['before', 'cool', 'hands_down']
klasses = ['before', 'blue', 'can', 'cool', 'drink', 'hands_down']

# model = 'Model/IDB_3_95'
model = 'Explore/model_6_95'
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

        if Q_status == Q_limit:
            Q.pop(0)
            Q.append(landmarks)
            Q_vid = expand_dims(array(Q), 0)


            results = model.predict(Q_vid)


            DQ.append(results[0])
            results = array(DQ).mean(axis=0)

            predicted_kls = klasses[results.argmax(axis=0)]
            text = f"{predicted_kls}"

        else:
            Q.append(landmarks)
            Q_status += 1


            text = f"{'|'*Q_status}"


        image = resize(image, (1280, 1024))
        image = putText(image, text, (30, image.shape[0] - 60), FONT_HERSHEY_SIMPLEX, 2, (11,209,252), 3)
            
        imshow('out', image)

        if waitKey(5) & 0xFF == 27: break # 27 is escape!

    cap.release()