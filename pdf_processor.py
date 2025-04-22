import PyPDF2
import os
from typing import Optional

def extract_text_from_pdf(pdf_file_path: str) -> Optional[str]:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_file_path (str): Path to the PDF file
        
    Returns:
        Optional[str]: Extracted text from the PDF, or None if extraction fails
    """
    try:
        with open(pdf_file_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None

def save_uploaded_pdf(file, upload_folder: str) -> Optional[str]:
    """
    Save an uploaded PDF file to the specified folder.
    
    Args:
        file: FileStorage object from Flask
        upload_folder (str): Path to the folder where PDFs should be saved
        
    Returns:
        Optional[str]: Path to the saved PDF file, or None if save fails
    """
    try:
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            
        filename = file.filename
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        return filepath
    except Exception as e:
        print(f"Error saving PDF file: {str(e)}")
        return None 