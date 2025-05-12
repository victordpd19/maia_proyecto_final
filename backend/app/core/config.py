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
    PROJECT_NAME: str = "CBAM Herdnet"
    PROJECT_DESCRIPTION: str = "Localization and clasifications of african wildlife"

    # Other animaloc and herdent configurations

    

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings object
settings = Settings()
