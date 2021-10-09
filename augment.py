
# from pathlib import Path
# from cv2 import VideoCapture

# video_dir = ''
# video_dir = Path(video_dir)

# out_dir = ''
# out_dir = Path(out_dir)

# def augment_video_from_dir(video_dir, out_dir, extension='avi'):
#     video_dir = Path(video_dir)
#     out_dir = Path(out_dir)

#     all_videos = video_dir.glob(f'*.{extension}')
#     for video in all_videos:
#         capture = VideoCapture(str(video))

#         while capture.isOpened(): pass


from pathlib import Path
from vidaug import augmentors as va
import skvideo.io
import cv2
import random
random.seed(1)
from ASL_UTILS.collection import Writer
vr = Writer()

aug_size_per_class = 200

source = ""
out = ""

classes = ["can", "before"]
print(classes)
source_ds = Path(source)
classes = list(x.name for x in source_ds.glob('*'))
print(classes)
aug_ds = Path(out)

all_source_ds = []
for category in classes:
    cat_dir = list((source_ds/category).glob("*"))
    all_source_ds.append(cat_dir)

counter_class = len(classes)
for i in range(len(classes)):
    counter_vid = aug_size_per_class
    category = classes[i]
    current_ds = all_source_ds[i]

    for _ in range(aug_size_per_class):
        cap = cv2.VideoCapture(str(random.choice(current_ds)))
        video_frames = []

        while True:
            status, frame = cap.read()
            if not status: break
            video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        video_shape = video_frames[0].shape

        seq = va.Sequential([va.RandomTranslate(int(video_shape[1]/4),50), va.RandomShear(0.07,0.07)])

        video_aug = seq(video_frames)
        aug_vid_name = f"{aug_ds}/{category}/{random.randint(100000,999999)}.mp4"
        print(aug_vid_name, type(aug_vid_name))
        print('saving')
        for aug_frame in video_aug:
            vr.write_as_video(aug_vid_name, aug_frame)
        # skvideo.io.vwrite(aug_vid_name, video_aug)
        # print("class:",counter_class,"||","vidno:",counter_vid,"/",aug_size_per_class,"||","filename:",aug_vid_name)
        counter_vid -=1

    counter_class -=1