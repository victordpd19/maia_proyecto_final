from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .api.routes import ping, inference #, extraction
from .api.routes.inference import router as inference_router

app = FastAPI(
    title="Invoice Extraction API",
    description="API for extracting data from PDF invoices",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ping.router)
#app.include_router(extraction.router, prefix="/extraction")
app.include_router(inference_router, prefix="/inference")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 