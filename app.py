import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, redirect, url_for, flash, session
from flask_login import LoginManager, current_user
from models import db, User
from config import config

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def create_app(config_name='default'):
    """Application factory"""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    
    # Initialize app-specific config
    config[config_name].init_app(app)
    
    # Register blueprints
    from routes.auth import auth_bp
    from routes.students import students_bp
    from routes.attendance import attendance_bp
    from routes.face_recognition import face_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(students_bp, url_prefix='/students')
    app.register_blueprint(attendance_bp, url_prefix='/attendance')
    app.register_blueprint(face_bp, url_prefix='/face')
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Main routes
    @app.route('/')
    def index():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        return redirect(url_for('auth.login'))
    
    @app.route('/dashboard')
    def dashboard():
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))
        
        from models import Student, Attendance
        from datetime import datetime
        
        # Get statistics
        total_students = Student.query.count()
        today = datetime.now().strftime('%Y-%m-%d')
        today_attendance = Attendance.query.filter_by(date=today).count()
        
        return render_template('dashboard.html', 
                             total_students=total_students,
                             today_attendance=today_attendance,
                             user=current_user)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return render_template('500.html'), 500
    
    return app

if __name__ == '__main__':
    app = create_app(os.environ.get('FLASK_ENV', 'development'))
    app.run(host='0.0.0.0', port=5001, debug=True)
