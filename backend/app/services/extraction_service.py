import base64
import os
import tempfile
import time
from typing import Dict, List, Optional, Any
import json

import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def extract_invoice_data(extraction_id: str, pdf_base64: str, extraction_tasks: Dict):
    """
    Process a PDF invoice and extract relevant information using OCR and OpenAI.
    
    Args:
        extraction_id: The unique ID for this extraction task
        pdf_base64: The PDF file encoded as base64
        extraction_tasks: The dictionary to store extraction results
    """
    try:

        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), http_client=None)
        TEMPERATURE = os.getenv("TEMPERATURE") or 0.1

        # Update status
        extraction_tasks[extraction_id]["status"] = "processing"
        extraction_tasks[extraction_id]["progress"] = 0.1
        
        # Decode PDF from base64
        pdf_bytes = base64.b64decode(pdf_base64)
        
        # Convert PDF to images
        extraction_tasks[extraction_id]["progress"] = 0.2
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name
        
        try:
            # Convert PDF to images
            images = convert_from_bytes(pdf_bytes)
            extraction_tasks[extraction_id]["progress"] = 0.4
            
            # Extract text using OCR
            full_text = ""
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                full_text += f"\n--- PAGE {i+1} ---\n{text}"
            
            extraction_tasks[extraction_id]["progress"] = 0.6
            
            # Use OpenAI to extract structured data
            prompt = f"""
            Extract the following information from this invoice:
            1. Vendor name
            2. Items (including name, quantity, and price per unit)
            3. Total tax amount
            4. Total invoice amount
            5. Invoice date

            Format the response as a JSON object with the following structure:
            {{
                "vendor_name": "Vendor Name",
                "items": [
                    {{
                        "name": "Item 1",
                        "quantity": 1,
                        "price": 10.99
                    }},
                    {{
                        "name": "Item 2",
                        "quantity": 2,
                        "price": 20.50
                    }}
                ],
                "total_tax": 10.50,
                "total_amount": 150.00,
                "date": "YYYY-MM-DD"
            }}

            Try to extract all fields, but if a field is not found in the invoice, set it to null.
            For items, if price is not available, set it to null but still include other item information.
            The price field should represent the price per unit, not the total price for the item quantity.
            
            Invoice text:
            {full_text}
            """
            
            extraction_tasks[extraction_id]["progress"] = 0.8
            
            # Call OpenAI API
            logger.info(f"Calling OpenAI API with prompt: {prompt}")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that extracts structured data from invoice text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=1000
            )
            logger.info(f"OpenAI API response: {response}")
            
            # Parse the response
            extracted_data = json.loads(response.choices[0].message.content)
            
            # Update extraction task with results
            extraction_tasks[extraction_id]["status"] = "completed"
            extraction_tasks[extraction_id]["progress"] = 1.0
            extraction_tasks[extraction_id]["results"] = extracted_data
        
        except Exception as e:
            logger.error(f"Error processing invoice: {e}")
            extraction_tasks[extraction_id]["status"] = "failed"
            extraction_tasks[extraction_id]["error"] = str(e)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
                
    except Exception as e:
        # Update status to failed
        extraction_tasks[extraction_id]["status"] = "failed"
        extraction_tasks[extraction_id]["error"] = str(e)

def get_extraction_status(extraction_id: str, extraction_tasks: Dict) -> Optional[Dict]:
    """
    Get the status of an extraction task.
    
    Args:
        extraction_id: The ID of the extraction task
        extraction_tasks: The dictionary containing extraction tasks
        
    Returns:
        Dict or None: The status information or None if not found
    """
    if extraction_id not in extraction_tasks:
        return None
    
    task = extraction_tasks[extraction_id]
    return {
        "extraction_id": extraction_id,
        "status": task["status"],
        "progress": task.get("progress", 0.0)
    }

def get_extraction_results(extraction_id: str, extraction_tasks: Dict) -> Optional[Dict]:
    """
    Get the results of an extraction task.
    
    Args:
        extraction_id: The ID of the extraction task
        extraction_tasks: The dictionary containing extraction tasks
        
    Returns:
        Dict or None: The extraction results or None if not found
    """
    if extraction_id not in extraction_tasks:
        return None
    
    task = extraction_tasks[extraction_id]
    
    if task["status"] == "completed" and task.get("results") is not None:
        results = task["results"]
        return {
            "extraction_id": extraction_id,
            "vendor_name": results.get("vendor_name"),
            "items": results.get("items", []),
            "total_tax": results.get("total_tax"),
            "date": results.get("date"),
            "total_amount": results.get("total_amount"),
            "status": task["status"]
        }
    
    return {
        "extraction_id": extraction_id,
        "status": task["status"],
        "error": task.get("error")
    } 