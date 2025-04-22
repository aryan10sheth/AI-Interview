from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF, or None if extraction fails
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
        return text.strip() if text else None
        
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None 