from flask import Flask
from dotenv import load_dotenv
import os
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize Flask app
app = Flask(__name__)

# Set a secret key for session management
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')  # Use environment variable or fallback

# Load configuration
from config.settings import UPLOAD_FOLDER, MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Register blueprints
from routes.interview_routes import interview_bp
app.register_blueprint(interview_bp)

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure proper permissions for upload directory
try:
    os.chmod(app.config['UPLOAD_FOLDER'], 0o777)
except Exception as e:
    print(f"Warning: Could not set permissions for upload directory: {e}")

if __name__ == '__main__':
    load_dotenv()
    app.run(debug=True) 