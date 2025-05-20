import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

color_map = {'blue': 0, 'green': 1, 'orange': 2, 'red': 3}
inv_color_map = {v: k for k, v in color_map.items()}

# بيانات الألوان - استبدلها ببياناتك الحقيقية
colors = [
    'green', 'blue', 'green', 'blue', 'blue', 'blue', 'blue', 'blue', 'green', 'blue',
    'blue', 'blue', 'blue', 'orange', 'green', 'blue', 'orange', 'blue', 'red', 'orange',
    # أضف باقي بيانات الألوان هنا حسب تاريخ الجولات
]

colors_num = [color_map[c] for c in colors]
window_size = 7

def prepare_sequences(colors, window_size):
    X, y = [], []
    for i in range(len(colors) - window_size):
        seq_in = colors[i:i + window_size]
        seq_out = colors[i + window_size]
        X.append(seq_in)
        y.append(seq_out)
    return np.array(X), np.array(y)

X, y = prepare_sequences(colors_num, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))
y_onehot = to_categorical(y, num_classes=4)

model_path = "crash_color_lstm_model.h5"

if os.path.exists(model_path):
    print("تحميل النموذج من الملف...")
    model = load_model(model_path)
else:
    print("بناء وتدريب النموذج...")
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window_size,1)))
    model.add(LSTM(32))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y_onehot, epochs=50, validation_split=0.2)
    model.save(model_path)
    print("تم حفظ النموذج في:", model_path)

# تجربة التنبؤ بناءً على إدخال المستخدم
def predict_next_color(input_colors):
    # تحويل الألوان إلى أرقام
    try:
        input_nums = [color_map[c] for c in input_colors]
    except KeyError:
        return "يوجد لون غير معروف. استخدم فقط: blue, green, orange, red"
    if len(input_nums) != window_size:
        return f"يجب إدخال {window_size} ألوان بالتتابع."
    input_arr = np.array(input_nums).reshape((1, window_size, 1))
    pred = model.predict(input_arr)
    pred_color_num = np.argmax(pred)
    return inv_color_map[pred_color_num]

# مثال على الاستخدام:
print("\nجرب توقع لون الجولة القادمة بإدخال 7 ألوان مفصولة بمسافة (مثلاً: blue green orange blue red green blue):")
user_input = input()
user_colors = user_input.strip().lower().split()
prediction = predict_next_color(user_colors)
print("اللون المتوقع:", prediction)
