import os
from dotenv import load_dotenv
from pydantic import BaseSettings
import pytesseract

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Invoice Extraction API"

    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Tesseract settings
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "tesseract")

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings object
settings = Settings()

# Configure pytesseract

pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
