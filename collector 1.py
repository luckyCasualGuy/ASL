from cv2 import VideoCapture, waitKey, imshow, cvtColor, flip, COLOR_BGR2RGB, COLOR_RGB2BGR, putText, FONT_HERSHEY_SIMPLEX, resize
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

out_dir = Path('test')
if out_dir.exists(): rmtree(out_dir)
out_dir.mkdir()

out_prefix = 'test'
klass = 'test'
frame_count = 70
video_count = 2

wait_flag = True
wait_count = 150
count_flag = False


# video_original = [] #
with mp_holistic.Holistic(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as holistic:

    while cap.isOpened():
        
        image: ndarray
        success, image = cap.read()

        if not success: continue

        image = flip(image, 1)

        if not wait_flag:
            # original = image.copy() #
            # video_original.append(original) #

            image = cvtColor(image, COLOR_BGR2RGB)

            image.flags.writeable = False
            results = holistic.process(image)

            image = cvtColor(image, COLOR_RGB2BGR)
            image = im.draw_results(image, results)

            landmarks = me.extract_landmarks(results)
            w.write_to_csv(landmarks, out_dir /  f'{out_prefix}_{video_count}.csv')

            frame_count -= 1
            if not frame_count: 
                wait_flag = True
                frame_count = 70

            text = f"Taking video {video_count}"

        else:
            # if len(video_original) == 70: #
            #     print('h')
            #     w.write_as_video(array(video_original), str(out_dir /  f'{out_prefix}_{video_count}.avi'))
            #     video_original = []

            wait_count -= 1
            if not wait_count:
                wait_count = 200
                wait_flag = False
                video_count -= 1
                if video_count == 0: break
            
            text = "Waiting"

            if wait_count < 100: image = putText(image, 'BE READY' if video_count != 1 else 'STOPPING', (int(image.shape[1]/2 - 100), int(image.shape[1]/2)), FONT_HERSHEY_SIMPLEX, 1, (146,35,255), 2)

        
        image = resize(image, (1280, 1024))
        image = putText(image, text, (int(image.shape[1]/2 - 300), 60), FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        image = putText(image, f"frame count {frame_count}", (image.shape[1] - 300, image.shape[0] - 90), FONT_HERSHEY_SIMPLEX, 1, (147, 248, 80), 3)
        image = putText(image, f"video count {video_count}", (image.shape[1] - 300, image.shape[0] - 60), FONT_HERSHEY_SIMPLEX, 1, (248, 115,80), 3)
        image = putText(image, f"wait count {wait_count}", (image.shape[1] - 300, image.shape[0] - 30), FONT_HERSHEY_SIMPLEX, 1, (88, 88, 252), 3)

        if video_count == 1:
            image = putText(image, 'LAST VIDEO' if frame_count != 70 else 'COMPLETE', (520, 150), FONT_HERSHEY_SIMPLEX, 1, (13,219,255), 2)
            
        imshow('out', image)

        if waitKey(5) & 0xFF == 27: break # 27 is escape!

    cap.release()