from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from models import db, Student
import os
from config import Config

students_bp = Blueprint('students', __name__)

@students_bp.route('/')
@login_required
def index():
    """Display all students"""
    students = Student.query.order_by(Student.created_at.desc()).all()
    return render_template('students.html', students=students)

@students_bp.route('/add', methods=['GET', 'POST'])
@login_required
def add():
    """Add new student"""
    if request.method == 'POST':
        # Get form data
        student_id = request.form.get('student_id')
        name = request.form.get('name')
        roll_no = request.form.get('roll_no')
        department = request.form.get('department')
        course = request.form.get('course')
        year = request.form.get('year')
        semester = request.form.get('semester')
        division = request.form.get('division')
        gender = request.form.get('gender')
        dob = request.form.get('dob')
        email = request.form.get('email')
        phone = request.form.get('phone')
        address = request.form.get('address')
        teacher_name = request.form.get('teacher_name')
        
        # Validation
        if not all([student_id, name, roll_no, department]):
            flash('Student ID, Name, Roll No, and Department are required', 'error')
            return render_template('student_form.html')
        
        # Check if student already exists
        if Student.query.filter_by(student_id=student_id).first():
            flash('Student ID already exists', 'error')
            return render_template('student_form.html')
        
        # Create new student
        new_student = Student(
            student_id=student_id,
            name=name,
            roll_no=roll_no,
            department=department,
            course=course,
            year=year,
            semester=semester,
            division=division,
            gender=gender,
            dob=dob,
            email=email,
            phone=phone,
            address=address,
            teacher_name=teacher_name,
            photo_sample='No'
        )
        
        try:
            db.session.add(new_student)
            db.session.commit()
            flash(f'Student {name} added successfully!', 'success')
            return redirect(url_for('students.index'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred while adding the student', 'error')
            print(f"Add student error: {e}")
    
    return render_template('student_form.html', student=None)

@students_bp.route('/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit(id):
    """Edit existing student"""
    student = Student.query.get_or_404(id)
    
    if request.method == 'POST':
        # Update student data
        student.name = request.form.get('name')
        student.roll_no = request.form.get('roll_no')
        student.department = request.form.get('department')
        student.course = request.form.get('course')
        student.year = request.form.get('year')
        student.semester = request.form.get('semester')
        student.division = request.form.get('division')
        student.gender = request.form.get('gender')
        student.dob = request.form.get('dob')
        student.email = request.form.get('email')
        student.phone = request.form.get('phone')
        student.address = request.form.get('address')
        student.teacher_name = request.form.get('teacher_name')
        
        try:
            db.session.commit()
            flash(f'Student {student.name} updated successfully!', 'success')
            return redirect(url_for('students.index'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred while updating the student', 'error')
            print(f"Update student error: {e}")
    
    return render_template('student_form.html', student=student)

@students_bp.route('/delete/<int:id>', methods=['POST'])
@login_required
def delete(id):
    """Delete student and associated face data."""
    student = Student.query.get_or_404(id)

    try:
        # Delete student photo folder if it exists (legacy grayscale JPEGs)
        data_folder = Config.DATA_FOLDER
        student_folder = os.path.join(data_folder, f"student_{student.id}")
        if os.path.exists(student_folder):
            import shutil
            shutil.rmtree(student_folder, ignore_errors=True)

        # FaceEmbedding rows are cascade-deleted by DB FK (ondelete='CASCADE')
        db.session.delete(student)
        db.session.commit()
        flash(f'Student {student.name} deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash('An error occurred while deleting the student', 'error')
        print(f"Delete student error: {e}")

    return redirect(url_for('students.index'))

@students_bp.route('/api/search')
@login_required
def search():
    """Search students API"""
    query = request.args.get('q', '')
    search_by = request.args.get('by', 'name')
    
    if search_by == 'roll_no':
        students = Student.query.filter(Student.roll_no.like(f'%{query}%')).all()
    else:
        students = Student.query.filter(Student.name.like(f'%{query}%')).all()
    
    return jsonify([student.to_dict() for student in students])

@students_bp.route('/api/all')
@login_required
def get_all():
    """Get all students API"""
    students = Student.query.all()
    return jsonify([student.to_dict() for student in students])
