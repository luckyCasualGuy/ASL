'''
PREDICTIONS INSIDE FOR LOOP
'''

from numpy import expand_dims

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

from tensorflow.keras.models import load_model
from collections import deque
Q = deque(maxlen=5)

from ASL_UTILS.extractor import MediapipeExtractor
me = MediapipeExtractor()

from pandas import read_csv

from pathlib import Path

model = 'Explore/model_6_95'
model = load_model(str(model))

klasses = ['before', 'blue', 'can', 'cool', 'drink', 'hands_down']
test = Path('test')

for csv in test.glob('*'):
    data = read_csv(str(csv), header=None).values
    data = expand_dims(data, 0)
    results = model.predict(data)
    predicted_kls = klasses[results[0].argmax(axis=0)]
    print('----------------')
    print(csv)
    print(data.shape)
    print("PREDICTION --> ", results[0])
    print("PREDICTION --> ", predicted_kls)
    print('----------------')