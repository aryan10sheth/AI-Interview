from flask import Blueprint, request, jsonify
from .recruiter_evaluator import RecruiterEvaluator
from .utils.pdf_processor import extract_text_from_pdf
import os
import tempfile

interview_bp = Blueprint('interview', __name__)
recruiter = RecruiterEvaluator()

@interview_bp.route('/start-interview', methods=['POST'])
def start_interview():
    if 'cv_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No CV file provided'}), 400
        
    cv_file = request.files['cv_file']
    job_description = request.form.get('job_description')
    
    if not cv_file or not job_description:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
        
    if not cv_file.filename.lower().endswith('.pdf'):
        return jsonify({'status': 'error', 'message': 'Only PDF files are accepted'}), 400
    
    # Save the uploaded file temporarily
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, 'temp_cv.pdf')
    cv_file.save(temp_path)
    
    try:
        # Extract text from the PDF
        cv_text = extract_text_from_pdf(temp_path)
        if cv_text is None:
            return jsonify({'status': 'error', 'message': 'Failed to extract text from PDF'}), 400
        
        # Initialize the interview
        recruiter.initialize_interview(cv_text, job_description)
        next_question = recruiter.get_next_question()
        total_questions = recruiter.get_total_questions()
        
        return jsonify({
            'status': 'success',
            'next_question': next_question,
            'total_questions': total_questions
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path) 