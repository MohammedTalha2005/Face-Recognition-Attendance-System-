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


def get_or_create_ssl_context():
    """Generate or retrieve persistent self-signed SSL certificate with SAN for HTTPS."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cert_path = os.path.join(base_dir, 'cert.pem')
    key_path = os.path.join(base_dir, 'key.pem')

    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        try:
            import datetime
            import ipaddress
            import socket
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa

            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, u"192.168.31.200"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Attendance System"),
            ])

            san_list = [
                x509.DNSName(u"localhost"),
                x509.IPAddress(ipaddress.IPv4Address(u"127.0.0.1")),
                x509.IPAddress(ipaddress.IPv4Address(u"192.168.31.200")),
            ]
            try:
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                if local_ip not in ("127.0.0.1", "192.168.31.200"):
                    san_list.append(x509.IPAddress(ipaddress.IPv4Address(local_ip)))
            except Exception:
                pass

            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
                .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3650))
                .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
                .sign(key, hashes.SHA256())
            )

            with open(key_path, "wb") as f:
                f.write(key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                ))

            with open(cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))

        except Exception:
            return 'adhoc'

    return (cert_path, key_path)


if __name__ == '__main__':
    import threading
    from werkzeug.serving import run_simple

    app = create_app(os.environ.get('FLASK_ENV', 'development'))
    use_ssl = os.environ.get('USE_SSL', 'true').lower() in ('1', 'true', 'yes')
    ssl_ctx = get_or_create_ssl_context() if use_ssl else None

    # Run plain HTTP on port 5000 in background daemon thread (Zero SSL warnings on localhost)
    def run_http():
        run_simple('0.0.0.0', 5000, app, use_reloader=False, threaded=True)

    http_thread = threading.Thread(target=run_http, daemon=True)
    http_thread.start()

    print("\n" + "=" * 60)
    print(" [DUAL SERVER RUNNING] Attendance System")
    print("=" * 60)
    print(" >> Localhost (HTTP - NO WARNING):   http://127.0.0.1:5000")
    print("                                     http://localhost:5000")
    if ssl_ctx:
        print(" >> Network Phone (HTTPS with SSL):  https://192.168.31.200:5001")
        print("                                     https://127.0.0.1:5001")
    print("=" * 60 + "\n")

    if ssl_ctx:
        # Run HTTPS on port 5001 in main thread
        run_simple('0.0.0.0', 5001, app, ssl_context=ssl_ctx, use_reloader=False, threaded=True)
    else:
        http_thread.join()
