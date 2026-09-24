import os
import webbrowser
import threading
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import requests  # ✅ Correct import
from flask_mysqldb import MySQL

# Load CSV data
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load model
model = CNN.CNN(39)
model.load_state_dict(torch.load("Leaf_disease_model_1_latest.pt"))
model.eval()

# Ensure upload directory exists
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

# Initialize Flask app
app = Flask(__name__)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'tamil'
app.config['MYSQL_PASSWORD'] = 'tamil'
app.config['MYSQL_DB'] = 'testdb'

mysql = MySQL(app)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)

        # Get prediction for the uploaded image
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        # Insert into MySQL
        try:
            cur = mysql.connection.cursor()
            cur.execute("""
                INSERT INTO predictions (image_name, prediction_label, prediction_confidence)
                VALUES (%s, %s, %s)
            """, (filename, title, '95.0%'))
            mysql.connection.commit()
            cur.close()
        except Exception as e:
            app.logger.error(f"Database Insert Error: {e}")

        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name,
                               simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html',
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']),
                           buy=list(supplement_info['buy link']))

@app.route('/test_db')
def test_db():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT VERSION()")
        version = cur.fetchone()
        return f"<h2>✅ MySQL Connected!</h2><p>Version: {version[0]}</p>"
    except Exception as e:
        return f"<h2>❌ Error connecting to MySQL:</h2><p>{str(e)}</p>"

# ✅ Weather Route
@app.route('/weather', methods=['GET', 'POST'])
def weather():
    weather_data = None
    if request.method == 'POST':
        city = request.form['city']
        api_key = '892d082c07d2d113324f6d608930b1a4'  # Use your actual OpenWeather API key
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather_data = {
                'city': city,
                'temperature': data['main']['temp'],
                'description': data['weather'][0]['description'].capitalize(),
                'icon': data['weather'][0]['icon']
            }
        else:
            weather_data = {'error': 'City not found or API error.'}
    return render_template('weather.html', weather=weather_data)

# Auto-open browser on first run only
def open_browser():
    webbrowser.open_new("http://localhost:5000")

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.0, open_browser).start()

    app.run(host='localhost', port=5000, debug=True)
