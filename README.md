# 🧠 Brain Disease Detection System using AI

A full-stack AI-powered web application that detects **Brain Tumor, Stroke, and Alzheimer’s disease** from MRI images using Deep Learning models.
The system also provides **medical guidance, hospital recommendations, and user history tracking**.

---

## 🚀 Features

* 🧠 Detects:

  * Brain Tumor
  * Stroke
  * Alzheimer’s

* 🤖 AI-based image classification using deep learning

* 🌐 Web interface built with Flask

* 🏥 Hospital recommendations based on city

* 📋 Do’s and Don’ts for detected disease

* 🔐 User authentication (Login/Signup)

* 📊 Prediction history stored in database

---

## 🛠️ Tech Stack

### 🔹 Backend

* Python (Flask)
* TensorFlow / Keras
* OpenCV
* MySQL

### 🔹 Frontend

* HTML
* CSS
* JavaScript

### 🔹 Database

* MySQL

---

## 🧠 AI Models

* Tumor Detection Model (CNN)
* Stroke Detection Model (CNN)
* Alzheimer Detection Model (EfficientNet / Transfer Learning)

👉 Models are loaded in Flask app


---

## 📂 Project Structure

```bash
Brain-Disease-Detection/
│
├── app.py                  # Main Flask application
├── model/                 # Trained AI models (.keras)
├── static/uploads/        # Uploaded images
├── templates/             # HTML files
├── hospitalList.xlsx      # Hospital data
├── instructionList.xlsx   # Do's & Don'ts
├── Brain_Disease_Prediction.sql  # Database schema
├── requirement.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Brain-Disease-Detection.git
cd Brain-Disease-Detection
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirement.txt
```

👉 Required libraries: TensorFlow, OpenCV, Scikit-learn 

---

### 3️⃣ Setup Database (MySQL)

1. Open MySQL
2. Create database:

```sql
CREATE DATABASE brain_disease_db;
```

3. Import SQL file:

```bash
Brain_Disease_Prediction.sql
```

👉 Tables include:

* users
* predictions 

---

### 4️⃣ Configure Database in `app.py`

```python
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'your_password'
app.config['MYSQL_DB'] = 'brain_disease_db'
```

---

### 5️⃣ Run Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🧠 How It Works

1. User uploads MRI image
2. Image is preprocessed using OpenCV
3. Passed to trained AI models
4. Model predicts disease (Yes/No)
5. Results displayed with:

   * Diagnosis
   * Do’s & Don’ts
   * Nearby hospitals

---

## 📊 Model Training

* Transfer Learning using VGG16 / EfficientNet
* Image augmentation & preprocessing
* Early stopping and model checkpoints
* Multi-model architecture for different diseases

👉 Example training pipeline:


---

## 📌 Future Improvements

* 🧠 Add more diseases
* 📱 Mobile app integration
* ☁️ Deploy on cloud (AWS / Render)
* 📊 Improve model accuracy
* 🗂️ Add report download feature

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and improve the project.

---

## 📜 License

MIT License

---

## 📬 Contact

**Kunal Hande**
Let’s build impactful AI solutions 🚀
