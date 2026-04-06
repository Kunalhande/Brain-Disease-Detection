from flask import Flask, request, render_template, redirect, url_for, session, jsonify, flash
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import pandas as pd
import json
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

app.secret_key = 'VentureSystems'

UPLOAD_FOLDER = './static/uploads'
TUMOR_MODEL_PATH = './model/tumor_best_model.keras'
STROKE_MODEL_PATH = './model/stroke_best_model.keras'
ALZHEIMER_MODEL_PATH = './model/updated_alzheimer_best_model.keras'  # Add Alzheimer model path

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure MySQL connection
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Hk@12233'
app.config['MYSQL_DB'] = 'brain_disease_db'

mysql = MySQL(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
tumor_model = load_model(TUMOR_MODEL_PATH)
stroke_model = load_model(STROKE_MODEL_PATH)
alzheimer_model = load_model(ALZHEIMER_MODEL_PATH)  # Load Alzheimer model

# Load Excel file
EXCEL_PATH = "hospitalList.xlsx"
xls = pd.ExcelFile(EXCEL_PATH)
sheet_names = xls.sheet_names  # Get city names

# Load Excel file for Do's and Don'ts
INSTRUCTION_PATH = "instructionList.xlsx"
instruction_xls = pd.ExcelFile(INSTRUCTION_PATH)

# Load hospital data from Excel
def load_hospitals():
    file_path = "hospitalList.xlsx"  # Ensure this file is in the project folder
    df = pd.read_excel(file_path)
    return df.to_dict(orient="records")  # Convert DataFrame to a list of dictionaries

@app.route('/')
def home():
    print(sheet_names)
    return render_template('index.html', tumor=None, stroke=None, alzheimer=None, image=None, error=None, cities=sheet_names)

def get_instructions(disease):
    """Fetch Do's and Don'ts from instructionList.xlsx for the detected disease."""
    if disease in instruction_xls.sheet_names:
        df = pd.read_excel(instruction_xls, sheet_name=disease)

        # Check for variations of "Do's" and "Don'ts"
        dos_col = next((col for col in df.columns if "Do" in col), None)
        donts_col = next((col for col in df.columns if "Dont" in col), None)

        if dos_col and donts_col:
            return df[dos_col].dropna().tolist(), df[donts_col].dropna().tolist()

    return [], []

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', tumor=None, stroke=None, alzheimer=None, image=None, error="No file uploaded.", cities=sheet_names)

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', tumor=None, stroke=None, alzheimer=None, image=None, error="No file selected.")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        img = cv2.imread(filepath)
        img = cv2.resize(img, (224, 224))
        img_array = preprocess_input(np.expand_dims(img, axis=0))

        # Model Predictions
        tumor_pred = tumor_model.predict(img_array)
        stroke_pred = stroke_model.predict(img_array)
        alzheimer_pred = alzheimer_model.predict(img_array)

        # Interpret Results
        tumor_result = "Yes" if np.argmax(tumor_pred, axis=1)[0] == 1 else "No"
        stroke_result = "Yes" if np.argmax(stroke_pred, axis=1)[0] == 1 else "No"
        alzheimer_result = "Yes" if np.argmax(alzheimer_pred, axis=1)[0] == 1 else "No"

        # Save results to history if user is logged in
        if 'user_id' in session:
            prediction_data = {
                "tumor": tumor_result,
                "stroke": stroke_result,
                "alzheimer": alzheimer_result
            }
            cur = mysql.connection.cursor()
            cur.execute(
                "INSERT INTO predictions (user_id, image_path, result) VALUES (%s, %s, %s)",
                (session['user_id'], filepath, json.dumps(prediction_data))
            )
            mysql.connection.commit()
            cur.close()

        # Get Do's and Don'ts
        detected_disease = None
        if tumor_result == "Yes":
            detected_disease = "Tumor"
        elif stroke_result == "Yes":
            detected_disease = "Stroke"
        elif alzheimer_result == "Yes":
            detected_disease = "Alzheimer"

        do_list, dont_list = get_instructions(detected_disease) if detected_disease else ([], [])

        return render_template(
            'index.html',
            tumor=tumor_result,
            stroke=stroke_result,
            alzheimer=alzheimer_result,
            image=file.filename,
            dos=do_list,
            donts=dont_list,
            error=None,
            cities=sheet_names
        )
    except Exception as e:
        print(e)
        return render_template('index.html', tumor=None, stroke=None, alzheimer=None, image=None, error=str(e))

@app.route('/get_hospitals', methods=['GET'])
def get_hospitals():
    city = request.args.get('city')  # Extract city from query parameters
    if city and city in sheet_names:
        df = pd.read_excel(xls, sheet_name=city)
        hospitals = df.to_dict(orient='records')
        return jsonify(hospitals)
    return jsonify([])

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        age = request.form['age']
        gender = request.form['gender']
        contact = request.form['contact']
        hashed_password = generate_password_hash(password)

        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO users (name, email, password_hash, mobile_number, age, gender) VALUES (%s, %s, %s, %s, %s, %s)",
            (name, email, hashed_password, contact, age, gender)
        )
        mysql.connection.commit()
        cur.close()

        flash('Signup successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT id, name, password_hash FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[2], password):  # user[2] is password_hash
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

def save_prediction(user_id, image_path, disease, dos, donts):
    """Save predictions to MySQL database."""
    try:
        cursor = mysql.connection.cursor()
        result_data = {
            "disease": disease or "No Disease Detected",
            "dos": dos,
            "donts": donts
        }
        cursor.execute(
            "INSERT INTO predictions (user_id, image_path, result, prediction_date) VALUES (%s, %s, %s, NOW())",
            (user_id, image_path, json.dumps(result_data))
        )
        mysql.connection.commit()
        cursor.close()
    except Exception as e:
        print("Error saving prediction:", str(e))

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Please login to view your history', 'danger')
        return redirect(url_for('login'))
    
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT image_path, result, prediction_date 
        FROM predictions 
        WHERE user_id = %s 
        ORDER BY prediction_date DESC
    """, (session['user_id'],))
    predictions = cur.fetchall()
    cur.close()
    
    # Convert tuple results to dictionaries for easier template handling
    prediction_list = []
    for pred in predictions:
        prediction_list.append({
            'image_path': pred[0],
            'result': json.loads(pred[1]),  # Convert JSON string to dict
            'date': pred[2]
        })
    
    return render_template('history.html', predictions=prediction_list)

@app.context_processor
def inject_user():
    return dict(user_name=session.get('user_name'))

if __name__ == "__main__":
    app.run(debug=True)
