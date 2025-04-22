from flask import Blueprint, request, jsonify, send_file, render_template, session, current_app
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import json
import PyPDF2
from analyzers.recruiter_evaluator import RecruiterEvaluator
from analyzers.video_analyzer import VideoAnalyzer
from analyzers.audio_analyzer import AudioAnalyzer
from analyzers.response_analyzer import ResponseAnalyzer
import tempfile
import mimetypes
import whisper

from utils.file_processor import extract_text_from_pdf, save_uploaded_pdf
from config.settings import UPLOAD_FOLDER, ALLOWED_EXTENSIONS

interview_bp = Blueprint('interview', __name__)

# Initialize models
whisper_model = whisper.load_model("small")
video_analyzer = VideoAnalyzer()
response_analyzer = ResponseAnalyzer()
recruiter_evaluator = RecruiterEvaluator()

ALLOWED_EXTENSIONS = {'pdf'}
ALLOWED_VIDEO_TYPES = {'video/webm', 'video/mp4', 'video/webm;codecs=vp8,opus', 'video/webm;codecs=vp9,opus', 'video/webm;codecs=h264,opus'}

def allowed_file(filename, content_type=None):
    """Check if the file is allowed based on extension or content type."""
    if content_type:
        base_type = content_type.split(';')[0]  # Handle codecs in content type
        return base_type in {'video/webm', 'video/mp4'} or content_type in ALLOWED_VIDEO_TYPES
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return text

@interview_bp.route('/')
def index():
    """Serve the main interview page."""
    return render_template('index.html')

@interview_bp.route('/start-interview', methods=['POST'])
def start_interview():
    """Initialize a new interview session."""
    try:
        # Accept multipart/form-data for file upload
        if 'cv' not in request.files or 'job_description' not in request.form:
            return jsonify({'status': 'error', 'message': 'CV (PDF) and job description are required'}), 400

        cv_file = request.files['cv']
        job_description = request.form['job_description']

        if cv_file.filename == '' or not cv_file.filename.lower().endswith('.pdf'):
            return jsonify({'status': 'error', 'message': 'A valid PDF file is required for the CV'}), 400

        # Save the uploaded PDF to the uploads folder
        filename = secure_filename(cv_file.filename)
        pdf_path = os.path.join(UPLOAD_FOLDER, f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
        cv_file.save(pdf_path)

        # Generate questions (3 resume-based, 2 personal)
        questions = recruiter_evaluator.generate_interview_questions(pdf_path, job_description)

        # Store in session
        session['questions'] = questions
        session['current_question_index'] = 0
        session['responses'] = []

        return jsonify({
            'status': 'success',
            'message': 'Interview started',
            'total_questions': len(questions),
            'next_question': questions[0]
        })
    except Exception as e:
        print(f"Error in start_interview: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@interview_bp.route('/submit-response', methods=['POST'])
def submit_response():
    """Handle video response submission."""
    print("Received submit-response request")
    
    # Get current question from session
    questions = session.get('questions', [])
    current_index = session.get('current_question_index', 0)
    
    if not questions or current_index >= len(questions):
        return jsonify({
            'status': 'error',
            'message': 'No active question found'
        }), 400
    
    current_question = questions[current_index]
    
    if 'video' not in request.files:
        print("No video file in request")
        return jsonify({'status': 'error', 'message': 'No video file provided'}), 400
    
    video_file = request.files['video']
    print(f"Received file: {video_file.filename}, Content-Type: {video_file.content_type}")
    
    if video_file.filename == '':
        print("Empty filename")
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    if not video_file.content_type or not video_file.content_type.startswith('video/'):
        print(f"Invalid content type: {video_file.content_type}")
        return jsonify({'status': 'error', 'message': f'Invalid content type: {video_file.content_type}'}), 400

    if not allowed_file(video_file.filename, video_file.content_type):
        print(f"File type not allowed: {video_file.content_type}")
        return jsonify({'status': 'error', 'message': 'Invalid file format. Allowed formats: WebM, MP4'}), 400

    try:
        # Create a unique filename using timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{secure_filename(video_file.filename)}"
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Ensure upload directory exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Save the video file
        video_file.save(video_path)
        print(f"Video saved to: {video_path}")
        
        if not os.path.exists(video_path):
            raise Exception("Failed to save video file")
        
        file_size = os.path.getsize(video_path)
        print(f"Saved file size: {file_size} bytes")
        
        if file_size == 0:
            raise Exception("Saved file is empty")

        # Transcribe the video
        print("Starting transcription...")
        result = whisper_model.transcribe(video_path)
        transcription = result["text"]
        print(f"Transcription completed: {transcription[:100]}...")

        # Analyze video and response
        print("Analyzing video...")
        video_analysis = video_analyzer.analyze_video(video_path)
        print("Analyzing response...")
        response_analysis = response_analyzer.analyze_response(transcription, current_question)
        
        # Evaluate the response
        print("Evaluating response...")
        evaluation = recruiter_evaluator.evaluate(
            question=current_question,
            transcription=transcription,
            video_analysis=video_analysis
        )
        
        # Store response data in session
        responses = session.get('responses', [])
        responses.append({
            'question': current_question,
            'transcription': transcription,
            'video_analysis': video_analysis,
            'response_analysis': response_analysis,
            'evaluation': evaluation,
            'timestamp': timestamp
        })
        session['responses'] = responses
        
        # Increment question index
        session['current_question_index'] = current_index + 1
        
        return jsonify({
            'status': 'success',
            'transcription': transcription,
            'evaluation': evaluation
        })

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        # Clean up the video file if it was saved
        if 'video_path' in locals() and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Cleaned up video file: {video_path}")
            except Exception as cleanup_error:
                print(f"Error cleaning up video file: {str(cleanup_error)}")
        
        return jsonify({
            'status': 'error',
            'message': f'Error processing video: {str(e)}'
        }), 500

@interview_bp.route('/next-question', methods=['GET'])
def get_next_question():
    """Get the next interview question."""
    try:
        questions = session.get('questions', [])
        current_index = session.get('current_question_index', 0)
        
        if current_index >= len(questions):
            return jsonify({
                'status': 'completed',
                'message': 'Interview completed'
            })
        
        return jsonify({
            'status': 'success',
            'question_number': current_index + 1,
            'total_questions': len(questions),
            'question': questions[current_index]
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@interview_bp.route('/interview-summary', methods=['GET'])
def get_interview_summary():
    """Get the complete interview summary and analysis."""
    try:
        responses = session.get('responses', [])
        if not responses:
            return jsonify({'status': 'error', 'message': 'No interview data found'}), 404
        
        # Generate comprehensive report
        report = recruiter_evaluator.generate_report()
        
        return jsonify({
            'status': 'success',
            'summary': json.loads(report)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@interview_bp.route('/reset-interview', methods=['POST'])
def reset_interview():
    """Reset the interview session."""
    try:
        session.clear()
        return jsonify({
            'status': 'success',
            'message': 'Interview session reset successfully'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@interview_bp.route('/download_report', methods=['GET'])
def download_report():
    try:
        # Generate report using the recruiter evaluator
        report_path = recruiter_evaluator.generate_report()
        
        # Send the report file
        return send_file(
            report_path,
            as_attachment=True,
            download_name='interview_report.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@interview_bp.route('/generate-final-report', methods=['POST'])
def generate_final_report():
    """Generate final interview report after all questions are answered."""
    try:
        # Get all responses from session
        responses = session.get('responses', [])
        
        if len(responses) < 5:
            return jsonify({
                'success': False,
                'error': f'Not enough responses. Expected 5, got {len(responses)}'
            }), 400
        
        # Generate comprehensive final report
        final_report = recruiter_evaluator.generate_final_report()
        
        if 'error' in final_report:
            return jsonify({
                'success': False,
                'error': final_report['error']
            }), 400
            
        return jsonify({
            'success': True,
            'report': final_report
        })
        
    except Exception as e:
        print(f"Error generating final report: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500 