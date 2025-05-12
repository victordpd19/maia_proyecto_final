import base64
import os
import tempfile
import time
from typing import Dict, List, Optional, Any
import json

import pytesseract
from PIL import Image
import logging

# Load environment variables
load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def save_images_to_tmp_folder() -> str:

    "path to str file"
    pass

async def infer_images() -> str:

    "output folder results with inferred images"
    pass
