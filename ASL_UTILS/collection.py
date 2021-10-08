from pathlib import Path
from csv import writer
from numpy import ndarray

from cv2 import VideoWriter_fourcc, VideoWriter

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

from shutil import move

class Writer:
    def write_to_csv(self, result: ndarray, out_csv: str):
        out_csv = Path(out_csv)

        with out_csv.open('a+') as csv:
            csv_writer = writer(csv)
            csv_writer.writerow(result)

    def write_as_video(self, array: ndarray, out_video: str, frame_rate=30):
        fourcc = VideoWriter_fourcc(*'RGBA')
        # frame_size = array.shape[1:3][::-1]
        frame_size = array.shape[1:3]
        print(frame_size)
        return VideoWriter(out_video, fourcc, frame_rate, frame_size)


class ImageHandler:
    def draw_results(self, image, results):
        image.flags.writeable = True

        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        return image


class Maker:
    def combine_csvs(self): pass


    def append_csvs_to_dir(self, dir_: Path, append_dir: Path):
        dir_ = Path(dir_)
        last_entry = list(dir_.glob('*'))[-1]

        last_entry_count = int(last_entry.name.split('_')[-1].split('.')[0])


        append_dir = Path(append_dir)
        for path in append_dir.glob('*'):
            last_entry_count += 1
            prefix = '_'.join(path.name.split('_')[:-1])
            name = path.parent / f"{prefix}_{last_entry_count}{path.suffix}"
            # print(prefix)
            # print(name)
            # path.rename('temp.csv')
            path.rename(name)
            # move(str(path), str(dir_))
