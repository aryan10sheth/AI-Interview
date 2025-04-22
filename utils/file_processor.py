import os
from werkzeug.utils import secure_filename
import PyPDF2
from typing import Optional

def save_uploaded_pdf(file, upload_folder: str) -> Optional[str]:
    """Save uploaded PDF file and return the path."""
    try:
        # Create upload folder if it doesn't exist
        os.makedirs(upload_folder, exist_ok=True)
        
        # Secure the filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        
        # Save the file
        file.save(filepath)
        
        return filepath
    except Exception as e:
        print(f"Error saving PDF: {str(e)}")
        return None

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """Extract text content from PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None
    finally:
        # Clean up the file
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
        except Exception as cleanup_error:
            print(f"Error cleaning up PDF: {str(cleanup_error)}")

def save_audio_file(file, upload_folder: str) -> Optional[str]:
    """Save uploaded audio file and return the path."""
    try:
        # Create upload folder if it doesn't exist
        os.makedirs(upload_folder, exist_ok=True)
        
        # Secure the filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        
        # Save the file
        file.save(filepath)
        
        return filepath
    except Exception as e:
        print(f"Error saving audio file: {str(e)}")
        return None

def cleanup_files(files: list) -> None:
    """Clean up temporary files."""
    for file_path in files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up file {file_path}: {str(e)}") 