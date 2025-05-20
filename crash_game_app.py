import os
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

# إعدادات المسارات
DB_FILE = 'colors_db.json'
MODEL_FILE = 'color_predictor_model.keras'
CONFIG_FILE = 'model_config.json'

# ألوان اللعبة
COLOR_LIST = ['red', 'blue', 'green', 'yellow']
COLOR_TO_INT = {c: i for i, c in enumerate(COLOR_LIST)}
INT_TO_COLOR = {i: c for i, c in enumerate(COLOR_LIST)}

# طول التسلسل للتدريب والتوقع
SEQUENCE_LENGTH = 6

# تحميل البيانات من قاعدة البيانات
def load_data():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, 'r') as f:
        return json.load(f)

# حفظ البيانات في قاعدة البيانات
def save_data(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f)

# حفظ إعدادات النموذج
def save_model_config(sequence_length):
    with open(CONFIG_FILE, 'w') as f:
        json.dump({'sequence_length': sequence_length}, f)

# تحميل إعدادات النموذج
def load_model_config():
    if not os.path.exists(CONFIG_FILE):
        return None
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f).get('sequence_length')

# إنشاء نموذج LSTM جديد
def create_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(SEQUENCE_LENGTH, len(COLOR_LIST))))
    model.add(Dense(len(COLOR_LIST), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# تدريب النموذج على البيانات الحالية
def train_model():
    data = load_data()
    if len(data) <= SEQUENCE_LENGTH:
        print("ليس هناك بيانات كافية للتدريب.")
        return None

    previous_seq_len = load_model_config()
    if previous_seq_len is not None and previous_seq_len != SEQUENCE_LENGTH:
        print(f"تغيير في SEQUENCE_LENGTH من {previous_seq_len} إلى {SEQUENCE_LENGTH}، سيتم حذف النموذج.")
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)

    print(f"بدء تدريب النموذج على {len(data)} جولات...")

    encoded = [COLOR_TO_INT[color] for color in data]
    X = []
    y = []
    for i in range(len(encoded) - SEQUENCE_LENGTH):
        X.append(encoded[i:i+SEQUENCE_LENGTH])
        y.append(encoded[i+SEQUENCE_LENGTH])

    X = np.array([to_categorical(seq, num_classes=len(COLOR_LIST)) for seq in X])
    y = to_categorical(y, num_classes=len(COLOR_LIST))

    model = create_model()
    model.fit(X, y, epochs=150, verbose=0)
    model.save(MODEL_FILE)
    save_model_config(SEQUENCE_LENGTH)
    print("انتهى التدريب وحفظ النموذج.")
    return model

# التوقع
def predict_next_color():
    data = load_data()
    if len(data) < SEQUENCE_LENGTH:
        return None
    if not os.path.exists(MODEL_FILE):
        return None

    model = load_model(MODEL_FILE)
    sequence = data[-SEQUENCE_LENGTH:]
    encoded_seq = [COLOR_TO_INT[color] for color in sequence]
    X = np.array([to_categorical(encoded_seq, num_classes=len(COLOR_LIST))])
    prediction = model.predict(X, verbose=0)[0]
    predicted_index = np.argmax(prediction)
    return INT_TO_COLOR[predicted_index]

@app.route('/')
def index():
    data = load_data()
    prediction = predict_next_color()
    return render_template('index.html',
                           colors=data,
                           prediction=prediction,
                           total=len(data),
                           enumerate=enumerate,  # تمرير enumerate للقالب
                           color_options=COLOR_LIST)

@app.route('/add', methods=['POST'])
def add_color():
    color = request.form['color']
    data = load_data()
    if color in COLOR_LIST:
        data.append(color)
        save_data(data)
        train_model()  # إعادة تدريب النموذج بعد كل إضافة
    return redirect(url_for('index'))

@app.route('/delete/<int:index>')
def delete_color(index):
    data = load_data()
    if 0 <= index < len(data):
        data.pop(index)
        save_data(data)
        train_model()
    return redirect(url_for('index'))

@app.route('/edit/<int:index>', methods=['POST'])
def edit_color(index):
    new_color = request.form['new_color']
    data = load_data()
    if new_color in COLOR_LIST and 0 <= index < len(data):
        data[index] = new_color
        save_data(data)
        train_model()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
