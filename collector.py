from distutils.core import setup, Extension

from cv2 import VideoCapture, imshow, waitKey
from cv2 import VideoWriter, VideoWriter_fourcc
from cv2 import flip, cvtColor, resize
from cv2 import COLOR_BGR2RGB, COLOR_RGB2BGR

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

from ASL_UTILS.extractor import MediapipeExtractor
from ASL_UTILS.collection import ImageHandler
me = MediapipeExtractor()
im = ImageHandler()

capture = VideoCapture(0)
writer = VideoWriter('out.avi', VideoWriter_fourcc('M','J','P','G'), 30, (640, 480)) #30fps

C_video_size = (640, 480)
C_write_mode = True

F_stop = False
F_pause = False

with mp_holistic.Holistic(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as holistic:


    while not F_stop:

        if not F_pause:
            ret, frame = capture.read()
            frame = resize(frame, C_video_size)
            if C_write_mode: writer.write(frame)
        else:
            print('stream Paused')

        if ret: 
            # Mediapipe frame handler here
            # frame = flip(frame, 1)
            # frame = cvtColor(frame, COLOR_BGR2RGB)
            # frame.flags.writeable = False
            # results = holistic.process(frame)
            # frame = cvtColor(frame, COLOR_RGB2BGR)
            # frame = im.draw_results(frame, results)

            imshow('Video Capture', frame)
        
        key = waitKey(10)
        if key == 27: F_stop = True #esc
        elif key == 112: F_pause = not F_pause #pause