import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import os

# إعداد المتغيرات
window_size = 5
data_file = 'rounds_data.csv'
model_path = 'lstm_crash_model.keras'

# التحقق من وجود ملف البيانات
if not os.path.exists(data_file):
    print("⚠️ ملف البيانات rounds_data.csv غير موجود.")
    exit()

# تحميل البيانات
df = pd.read_csv(data_file)

if len(df) <= window_size:
    print("⚠️ لا توجد بيانات كافية لتدريب النموذج.")
    exit()

# تحضير البيانات
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

stages = df['stage'].values
X, y = create_sequences(stages, window_size)

X = X.reshape((X.shape[0], X.shape[1], 1))

# بناء وتدريب النموذج
model = Sequential([
    LSTM(50, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=30, verbose=1)

# حفظ النموذج
model.save(model_path)
print(f"✅ تم تدريب النموذج بنجاح وحُفظ في: {model_path}")
