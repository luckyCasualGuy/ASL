'''
CLEARING THE ARRAY AFTER EACH PREDICTION
'''

from cv2 import VideoCapture, waitKey, imshow, cvtColor, flip, COLOR_BGR2RGB, COLOR_RGB2BGR, putText, FONT_HERSHEY_SIMPLEX, resize, rectangle, line
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
klasses = ['before', 'bird', 'blue', 'can', 'cool', 'dog', 'drink', 'fish', 'hands_down']


QR = []
QR_limit = 70
QR_status = 0

# model = 'Model/IDB_3_95'
model = 'Model/ADB_9_95_675'
model = load_model(str(model))

width = 640
height = 480

width_from_center = 175
height_from_center_top = 200
height_from_center_bottom = int(height / 2)

point1 = (int((width/2) - width_from_center), int((height/2) - height_from_center_top))
point2 = (int((width/2) + width_from_center), int((height/2) + height_from_center_bottom))

lheight = int((height/2) - 10)
lpoint1 = (point1[0], lheight)
lpoint2 = (point2[0], lheight)

fwidth = 100
fpoint1 = (int(width/2 - fwidth), point1[1])
fpoint2 = ((int(width/2 + fwidth), lpoint1[1]))


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

        original = image.copy()

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

        try:
            image = putText(image, str((sorted(zip(klasses, results), key=lambda x: x[1], reverse=True))), (30, image.shape[0] - 150), FONT_HERSHEY_SIMPLEX, 1, (255,174,0), 2)
        except Exception as err:
            # pass
            image = putText(image, str(results), (30, image.shape[0] - 200), FONT_HERSHEY_SIMPLEX, 1, (255,174,0), 2)

        original = putText(original, str(original.shape), (30, 30), FONT_HERSHEY_SIMPLEX, 1, (11,209,252), 2)
        original = rectangle(original, point1, point2, (211, 66, 242), 2)
        original = line(original, lpoint1, lpoint2, (211, 66, 242), 2)
        original = rectangle(original, fpoint1, fpoint2, (92, 206, 17), 2)
            
        imshow('out', image)
        imshow('original', original)

        if waitKey(5) & 0xFF == 27: break # 27 is escape!

    cap.release()