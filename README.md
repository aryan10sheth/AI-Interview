# AI-Powered Interview System

An intelligent interview system that conducts and evaluates video interviews using AI technologies. The system analyzes responses using speech recognition, natural language processing, and advanced language models to provide comprehensive feedback.

## Features

- Video interview recording and processing
- Speech-to-text transcription using Whisper
- Natural language processing with spaCy
- Intelligent response evaluation
- PDF resume parsing
- Automated question generation based on resume and job description
- Comprehensive interview reports

## Tech Stack

- Python 3.x
- Flask (Web Framework)
- OpenAI Whisper (Speech Recognition)
- spaCy (NLP)
- PyPDF2 (PDF Processing)
- NLTK (Natural Language Processing)
- HTML/CSS/JavaScript (Frontend)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

4. Create a `.env` file in the root directory with your configuration:
```
SECRET_KEY=your-secret-key
```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Project Structure

- `/analyzers` - Analysis modules for video, audio, and text processing
- `/config` - Configuration settings
- `/routes` - Flask route handlers
- `/templates` - HTML templates
- `/utils` - Utility functions
- `/uploads` - Directory for uploaded files (created automatically)

## Usage

1. Upload a resume (PDF) and provide a job description
2. Answer the generated interview questions via video
3. Receive real-time feedback and evaluations
4. Get a comprehensive interview report

## Note

This is a development version. For production deployment:
- Use a production WSGI server
- Secure the application properly
- Configure proper storage for uploads
- Set up proper logging 