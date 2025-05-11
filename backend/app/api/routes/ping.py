from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/ping", summary="Health check endpoint")
async def ping():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        dict: A simple response with status "ok"
    """
    return {"status": "pong"} 