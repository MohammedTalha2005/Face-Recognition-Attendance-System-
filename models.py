from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    contact = db.Column(db.String(20))
    security_question = db.Column(db.String(200))
    security_answer = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Student(db.Model):
    """Student model"""
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    name = db.Column(db.String(200), nullable=False)
    roll_no = db.Column(db.String(50), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    course = db.Column(db.String(100))
    year = db.Column(db.String(20))
    semester = db.Column(db.String(20))
    division = db.Column(db.String(10))
    gender = db.Column(db.String(10))
    dob = db.Column(db.String(20))
    email = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    teacher_name = db.Column(db.String(200))
    photo_sample = db.Column(db.String(10), default='No')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    attendance_records = db.relationship('Attendance', backref='student', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert student object to dictionary"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'name': self.name,
            'roll_no': self.roll_no,
            'department': self.department,
            'course': self.course,
            'year': self.year,
            'semester': self.semester,
            'division': self.division,
            'gender': self.gender,
            'dob': self.dob,
            'email': self.email,
            'phone': self.phone,
            'address': self.address,
            'teacher_name': self.teacher_name,
            'photo_sample': self.photo_sample,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }
    
    def __repr__(self):
        return f'<Student {self.name} ({self.student_id})>'

class Attendance(db.Model):
    """Attendance model"""
    __tablename__ = 'attendance'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), db.ForeignKey('students.student_id'), nullable=False, index=True)
    roll_no = db.Column(db.String(50))
    name = db.Column(db.String(200))
    department = db.Column(db.String(100))
    time = db.Column(db.String(20))
    date = db.Column(db.String(20), index=True)
    status = db.Column(db.String(20), default='Present')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert attendance object to dictionary"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'roll_no': self.roll_no,
            'name': self.name,
            'department': self.department,
            'time': self.time,
            'date': self.date,
            'status': self.status,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }

    def __repr__(self):
        return f'<Attendance {self.name} - {self.date} {self.time}>'
    
