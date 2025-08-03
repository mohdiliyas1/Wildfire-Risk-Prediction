from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import sqlite3
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load model and preprocessor
model = joblib.load("knn_model.pkl")
scaler = joblib.load("knn_scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

def save_prediction_to_db(dc, temp, dec, mar, prediction, latitude=0.0, longitude=0.0):
    conn = sqlite3.connect('wildfire_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dc REAL,
            temp REAL,
            month_dec INTEGER,
            month_mar INTEGER,
            prediction INTEGER,
            latitude REAL,
            longitude REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        INSERT INTO predictions (dc, temp, month_dec, month_mar, prediction, latitude, longitude)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (dc, temp, dec, mar, prediction, latitude, longitude))
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict-page')
def predict_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        dc = float(data['DC'])
        temp = float(data['temp'])
        dec = int(data['month_dec'])
        mar = int(data['month_mar'])
        latitude = float(data.get('latitude', 0.0))
        longitude = float(data.get('longitude', 0.0))
        input_values = np.array([dc, temp, dec, mar]).reshape(1, -1)
    except KeyError as e:
        return jsonify({'error': f'Missing input for: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    scaled_input = scaler.transform(input_values)
    prediction = int(model.predict(scaled_input)[0])
    save_prediction_to_db(dc, temp, dec, mar, prediction, latitude, longitude)

    message = "⚠️ High fire risk detected!" if prediction == 1 else "✅ Low fire risk."
    return jsonify({'fire_risk': prediction, 'message': message})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/indicators')
def indicators():
    return render_template('indicators.html')

@app.route('/our-models')
def models():
    return render_template('ourmodel.html')

@app.route('/data')
def view_data():
    conn = sqlite3.connect('wildfire_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    data = cursor.fetchall()
    conn.close()
    return render_template('data.html', data=data)

@app.route('/api/predictions')
def get_predictions():
    conn = sqlite3.connect('wildfire_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT dc, temp, month_dec, month_mar, prediction, latitude, longitude FROM predictions')
    rows = cursor.fetchall()
    conn.close()
    data = [{'dc': row[0], 'temp': row[1], 'month_dec': row[2], 'month_mar': row[3], 'prediction': row[4], 'latitude': row[5], 'longitude': row[6]} for row in rows]
    return jsonify(data)

@app.route('/map')
def map_view():
    return render_template('map.html')

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
