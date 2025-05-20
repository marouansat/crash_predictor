import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

COLORS_MAP = {
    'blue': 0,
    'green': 1,
    'orange': 2,
    'red': 3
}

INV_COLORS_MAP = {v: k for k, v in COLORS_MAP.items()}

SEQ_LENGTH = 11

def prepare_data(color_list):
    # تحويل الألوان إلى أرقام
    nums = [COLORS_MAP[c] for c in color_list]
    X, y = [], []
    for i in range(len(nums) - SEQ_LENGTH):
        X.append(nums[i:i+SEQ_LENGTH])
        y.append(nums[i+SEQ_LENGTH])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = to_categorical(y, num_classes=4)
    return X, y

def train_model(color_list):
    if len(color_list) < SEQ_LENGTH + 1:
        return None
    X, y = prepare_data(color_list)
    model = Sequential()
    model.add(LSTM(50, input_shape=(SEQ_LENGTH,1)))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=30, batch_size=16, verbose=0)
    return model

def predict_next_color(model, last_colors):
    nums = np.array([COLORS_MAP[c] for c in last_colors]).reshape((1, SEQ_LENGTH,1))
    pred = model.predict(nums, verbose=0)
    idx = np.argmax(pred)
    return INV_COLORS_MAP[idx]
