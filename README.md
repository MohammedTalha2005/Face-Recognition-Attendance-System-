# Attendance Monitoring System

A modern, web-based automatic attendance monitoring system. This version has been fully upgraded from the legacy LBPH & OpenCV cascade classifiers to a production-grade architecture utilizing **Google MediaPipe Tasks (BlazeFace)** for robust face detection, **Keras-FaceNet** for 512-D face embeddings, and **PostgreSQL** for secure, database-backed embedding storage.

---

## 🚀 Key Improvements & Features

- 🧠 **MediaPipe & FaceNet Pipeline**: Upgraded to Google MediaPipe Tasks (`blaze_face_short_range.tflite`) for superior detection accuracy under various lighting/angles, and FaceNet for generating high-fidelity 512-dimensional embeddings.
- ⚡ **Instant Student Registration**: **Zero retraining required.** Unlike traditional classifier models (e.g., LBPH) that require model rebuilding, newly enrolled students are instantly registered. Face comparison is computed in real-time using Cosine Similarity.
- 🗄️ **PostgreSQL Database Storage**: All student records, authentication details, attendance logs, and face embeddings are saved in a unified PostgreSQL database (replacing local file-based `data/*.jpg` and `ml_models/classifier.xml`).
- 🔐 **Secure Administration**: Admin login/registration dashboard built with Flask-Login.
- 📊 **Attendance Logs**: View attendance, filter records by date or department, and export them directly as CSV sheets.
- 🎨 **Modern Responsive UI**: Clean, responsive styling with dark-mode aesthetic.

---

## 🛠️ Prerequisites

- **Python**: version `3.8` to `3.11` (recommended)
- **PostgreSQL**: Installed and running locally or remotely
- **Webcam**: Required for student registration and live attendance marking

---

## ⚙️ Installation & Setup

### 1. Set Up the Repository
Clone or download the project folder, then navigate into it:
```bash
cd "e:/Projects/Attendence System"
```

### 2. Create and Activate Virtual Environment
Create a virtual environment to manage dependencies:
```bash
python -m venv venv
```

**Activate on Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Activate on Linux/macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the package manager:
```bash
pip install -r requirements.txt
```

### 4. Database Setup
1. Log into your PostgreSQL instance (e.g., via `pgAdmin` or `psql` shell).
2. Create a new database named `a_s` (or any name you prefer):
   ```sql
   CREATE DATABASE a_s;
   ```
3. The application will automatically initialize the database schema and create all required tables (including `students`, `users`, `attendance`, and `face_embeddings`) upon the first startup.

### 5. Configure Environment Variables
1. Copy the template file to create your configuration file:
   ```bash
   copy .env.example .env
   ```
2. Open the newly created `.env` file in your editor and configure your PostgreSQL connection:
   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_USER=postgres
   DB_PASSWORD=your_postgresql_password
   DB_NAME=a_s
   SECRET_KEY=generate-a-secure-random-key-here
   FACE_SAMPLES_COUNT=50
   FACE_SIMILARITY_THRESHOLD=0.6
   ```

### 6. Run the Application
Start the Flask dev server using the provided helper batch file or run directly with Python:
```bash
.\run.bat
```
*or*
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

## 👥 Default Credentials

- **Username**: `admin`
- **Password**: `admin`

> ⚠️ **Security Warning**: Please change the default credentials or register a new admin user after your first successful login.

---

## 📂 Project Structure

```
Attendence System/
├── app.py                     # Main Flask entrypoint & app factory
├── config.py                  # PostgreSQL & FaceNet configurations
├── models.py                  # SQLAlchemy Database Models (Student, Attendance, FaceEmbedding)
├── face_detection.py          # Wrapper for Google MediaPipe Tasks face detection
├── face_recognition_engine.py # Embedding generator using Keras-FaceNet
├── requirements.txt           # Project dependencies
├── run.bat                    # Windows startup script
├── routes/                    # Blueprint routes
│   ├── auth.py                # Admin user auth (Login, Register, Logout)
│   ├── students.py            # Student profiles management
│   ├── attendance.py          # Attendance logging, viewing, and exporting
│   └── face_recognition.py    # Face scan registration & attendance detection routes
├── ml_models/                 # Stores model files
│   └── blaze_face_short_range.tflite # MediaPipe Face Detection model
├── static/                    # Front-end static assets
│   ├── css/style.css          # Main responsive stylesheet
│   └── js/
│       ├── main.js            # General UI helpers
│       └── camera.js          # WebRTC webcam utility logic
└── templates/                 # Jinja2 HTML templates
```

---

## 🔍 Troubleshooting

### Camera Not Opening
- Ensure you have granted the browser permissions to access your webcam.
- If using Chrome/Edge on remote deployment, camera access requires **HTTPS** protocols.

### Database Connection Crashes
- Double-check the `DB_USER`, `DB_PASSWORD`, and `DB_PORT` values in your `.env` file.
- Verify that your local PostgreSQL service is running. On Windows, you can check this under Windows Services (`services.msc`).
- If you have special characters (e.g. `@`, `#`) in your PostgreSQL password, the application automatically handles URL-encoding, but double-check details if connection fails.

---

## 💡 Technologies Used

- **Web Framework**: Flask, Werkzeug
- **ORM & Database**: Flask-SQLAlchemy, PostgreSQL (`psycopg2-binary`)
- **Face Detection**: Google MediaPipe (Tasks Vision API)
- **Face Embeddings**: Keras-FaceNet (TensorFlow back-end)
- **Front-end**: Vanilla JS, WebRTC, custom CSS
