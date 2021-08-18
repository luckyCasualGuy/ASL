from pandas import DataFrame, read_csv, get_dummies
from pathlib import Path
from numpy import array, random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.core import Dropout


db = Path('DBS')
selected_db = db / 'ADB'
model_out = Path('Model')
model_name = model_out / 'ADB_9_95'

ACCURACY_THRESHOLD = 0.95

y = []
x = []
for klass in selected_db.glob('*'):
    print(klass.name)
    
    for csv in klass.glob('*'):
        y.append(klass.name)
        print(csv)
        x.append(read_csv(str(csv), header=None).values)
    

y = get_dummies(DataFrame(y)).values
x = array(x)

p = random.permutation(len(y))
y = y[p]
x = x[p]

print('Data Shape Check', x.shape, y.shape)

print('Model will stop training after reacing: ', ACCURACY_THRESHOLD)
class stop_(Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy') > ACCURACY_THRESHOLD):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))   
            self.model.stop_training = True
            
stop_training = stop_()

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=x.shape[1:]))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))
model.summary()

a = Adam(learning_rate=0.00001)
H = model.compile(optimizer=a, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x, y, epochs=1000, verbose=1, callbacks=[stop_training])

print('Saving model')
model.save(str(model_name))