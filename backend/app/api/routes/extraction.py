from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uuid
import base64
from app.services.extraction_service import extract_invoice_data, get_extraction_status, get_extraction_results

router = APIRouter(tags=["extraction"])

class ExtractionRequest(BaseModel):
    pdf_base64: str
    
class ExtractionResponse(BaseModel):
    extraction_id: str
    status: str

class ExtractionStatusResponse(BaseModel):
    extraction_id: str
    status: str
    progress: Optional[float] = None
    
class Item(BaseModel):
    name: str = Field(..., description="Name of the item")
    quantity: float = Field(..., description="Quantity of the item")
    price: Optional[float] = Field(None, description="Price per unit")

class ExtractionResultsResponse(BaseModel):
    extraction_id: str
    vendor_name: Optional[str] = Field(None, description="Name of the vendor")
    items: Optional[List[Item]] = Field(None, description="List of items in the invoice")
    total_tax: Optional[float] = Field(None, description="Total tax amount")
    total_amount: Optional[float] = Field(None, description="Total invoice amount")
    date: Optional[str] = Field(None, description="Invoice date")
    status: str = Field(..., description="Status of the extraction")

# In-memory storage for extraction tasks (in production, use a database)
extraction_tasks = {}

@router.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(request: ExtractionRequest, background_tasks: BackgroundTasks):
    """
    Start an extraction task for a PDF invoice.
    
    Args:
        request: The extraction request containing the PDF as base64
        
    Returns:
        ExtractionResponse: The extraction task ID and status
    """
    try:
        # Generate a unique ID for this extraction
        extraction_id = str(uuid.uuid4())
        
        # Store initial status
        extraction_tasks[extraction_id] = {
            "status": "processing",
            "progress": 0.0,
            "results": None
        }
        
        # Add extraction task to background tasks
        background_tasks.add_task(
            extract_invoice_data,
            extraction_id,
            request.pdf_base64,
            extraction_tasks
        )
        
        return {"extraction_id": extraction_id, "status": "processing"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@router.get("/status/{extraction_id}", response_model=ExtractionStatusResponse)
async def get_status(extraction_id: str):
    """
    Get the status of an extraction task.
    
    Args:
        extraction_id: The ID of the extraction task
        
    Returns:
        ExtractionStatusResponse: The status of the extraction task
    """
    status = get_extraction_status(extraction_id, extraction_tasks)
    if status is None:
        raise HTTPException(status_code=404, detail="Extraction task not found")
    
    return status

@router.get("/results/{extraction_id}", response_model=ExtractionResultsResponse)
async def get_results(extraction_id: str):
    """
    Get the results of a completed extraction task.
    
    Args:
        extraction_id: The ID of the extraction task
        
    Returns:
        ExtractionResultsResponse: The extraction results
    """
    results = get_extraction_results(extraction_id, extraction_tasks)
    if results is None:
        raise HTTPException(status_code=404, detail="Extraction task not found")
    
    if results["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Extraction not completed yet. Current status: {results['status']}")
    
    return results 