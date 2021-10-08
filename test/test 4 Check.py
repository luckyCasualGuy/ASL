from cv2 import VideoCapture, waitKey, imshow, cvtColor, flip, COLOR_BGR2RGB, COLOR_RGB2BGR, putText, FONT_HERSHEY_SIMPLEX, resize, rectangle, line
from numpy import ndarray, array
from pathlib import Path
from shutil import rmtree

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

from ASL_UTILS.extractor import MediapipeExtractor
from ASL_UTILS.collection import Writer, ImageHandler

cap = VideoCapture(0)
me = MediapipeExtractor()
w = Writer()
im = ImageHandler()


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

while cap.isOpened():
        
    image: ndarray
    success, image = cap.read()

    if not success: continue

    original = image.copy()

    image = flip(image, 1)


    image = resize(image, (1280, 1024))
    image = putText(image, 'TESTING', (30, image.shape[0] - 30), FONT_HERSHEY_SIMPLEX, 2, (11,209,252), 3)
    image = putText(image, str(image.shape), (30, 30), FONT_HERSHEY_SIMPLEX, 1, (11,209,252), 3)
    
    original = putText(original, str(original.shape), (30, 30), FONT_HERSHEY_SIMPLEX, 1, (11,209,252), 2)
    original = rectangle(original, point1, point2, (211, 66, 242), 2)
    original = line(original, lpoint1, lpoint2, (211, 66, 242), 2)
    original = rectangle(original, fpoint1, fpoint2, (92, 206, 17), 2)
    
    imshow('out', image)
    imshow('original', original)

    if waitKey(5) & 0xFF == 27: break # 27 is escape!

cap.release()