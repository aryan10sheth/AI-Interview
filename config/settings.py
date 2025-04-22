import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory of the application
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Upload folder for storing files
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'webm', 'mp4'}

# Flask Configuration
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# API Keys
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

# Audio Configuration
SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# Video Configuration
FRAME_RATE = 30
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Analysis Weights
ANALYSIS_WEIGHTS = {
    'behavioral': 0.3,
    'communication': 0.3,
    'content': 0.4
}

# Scoring Thresholds
SCORE_THRESHOLDS = {
    'strongly_recommend': 8.5,
    'recommend': 7.0,
    'consider': 5.5
}

# Movement Analysis
MOVEMENT_THRESHOLDS = {
    'stable_threshold': 70,
    'moderate_movement': 50,
    'high_movement': 30
}

# Communication Analysis
COMMUNICATION_METRICS = {
    'filler_word_threshold': 0.1,
    'vocabulary_diversity_threshold': 0.3,
    'min_words_per_minute': 120,
    'max_words_per_minute': 160
}

# Sentiment Analysis
SENTIMENT_THRESHOLDS = {
    'very_positive': 0.3,
    'positive': 0,
    'very_negative': -0.3,
    'negative': 0
}

# Eye Contact Analysis
EYE_CONTACT_THRESHOLDS = {
    'excellent': 0.8,
    'good': 0.6,
    'needs_improvement': 0.4
}

# Report Generation
REPORT_SECTIONS = [
    'behavioral_assessment',
    'communication_assessment',
    'content_assessment',
    'recommendation',
    'detailed_feedback'
] 