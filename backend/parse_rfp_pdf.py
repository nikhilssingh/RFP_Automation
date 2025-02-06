#parse_rfp_pdf.py

import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# If Tesseract-OCR is not installed at default location, set path manually:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows Example

def parse_rfp_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF, including images, charts, and graphs using OCR.
    Uses pdfplumber first (best for structured text), then PyMuPDF (fallback for extracted text),
    and finally pytesseract OCR for any images.
    
    Returns extracted text or an error message if processing fails.
    """
    extracted_text = ""

    # ✅ Step 1: Try extracting structured text with pdfplumber (best for formatted documents)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"

                # ✅ Extract text from images, graphs, and charts using OCR
                image = page.to_image(resolution=300)  # Convert PDF page to image
                img = image.original
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    extracted_text += "\n[OCR Extracted from Image]\n" + ocr_text + "\n"

        if extracted_text.strip():
            logging.info(f"✅ Successfully extracted text from {pdf_path} using pdfplumber & OCR.")
            return extracted_text.strip()
    except Exception as e:
        logging.warning(f"⚠️ pdfplumber & OCR failed on {pdf_path}: {e}")

    # ✅ Step 2: Try PyMuPDF (Better for text-based PDFs)
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            extracted_text += page.get_text("text") + "\n"

        if extracted_text.strip():
            logging.info(f"✅ Successfully extracted text from {pdf_path} using PyMuPDF.")
            return extracted_text.strip()
    except Exception as e:
        logging.warning(f"⚠️ PyMuPDF failed on {pdf_path}: {e}")

    # ✅ Step 3: If Both Methods Fail
    logging.error(f"❌ Unable to extract text from {pdf_path}. The PDF may be highly graphical or corrupted.")
    return "Error: Unable to process the PDF."
