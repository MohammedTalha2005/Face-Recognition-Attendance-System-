import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
    MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models')
    EMBEDDINGS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)

    # Database configuration (PostgreSQL)
    DB_HOST = os.environ.get('DB_HOST') or 'localhost'
    DB_PORT = os.environ.get('DB_PORT') or '5432'
    DB_USER = os.environ.get('DB_USER') or 'postgres'
    DB_PASSWORD = os.environ.get('DB_PASSWORD') or ''
    DB_NAME = os.environ.get('DB_NAME') or 'a_s'

    # URL-encode password to handle special characters (e.g. @ in Talha@786)
    try:
        from urllib.parse import quote_plus
        _db_password_encoded = quote_plus(DB_PASSWORD)
    except Exception:
        _db_password_encoded = DB_PASSWORD

    SQLALCHEMY_DATABASE_URI = (
        f"postgresql+psycopg2://{DB_USER}:{_db_password_encoded}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    # FaceNet recognition settings
    FACE_SAMPLES_COUNT = int(os.environ.get('FACE_SAMPLES_COUNT') or 50)
    FACE_SIMILARITY_THRESHOLD = float(os.environ.get('FACE_SIMILARITY_THRESHOLD') or 0.6)
    ANTI_SPOOF_THRESHOLD = float(os.environ.get('ANTI_SPOOF_THRESHOLD') or 0.65)

    # Ensure required directories exist
    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.MODEL_FOLDER, exist_ok=True)
        os.makedirs(Config.EMBEDDINGS_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(Config.UPLOAD_FOLDER, 'students'), exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
