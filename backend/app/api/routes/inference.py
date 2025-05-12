from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uuid
import base64
import os
import subprocess
import glob
import shutil
router = APIRouter(tags=["inference"])

# --- Constants (Consider moving to a configuration file) ---
MODEL_PTH_PATH = "app/herdnet/full_model.pth"  # Placeholder: Ensure this path is correct
BASE_IMAGES_DIR = "app/images"  # Base directory for storing inference images
INFER_SCRIPT_PATH = "app/herdnet/infer-custom.py"  # Path to the inference script
# --- End Constants ---

class InferenceRequest(BaseModel):
    image: str  # Changed from images: List[str] to a single image

class InferenceResponse(BaseModel):
    image: str

class InferenceStatusResponse(BaseModel):
    inference_id: str
    status: str
    progress: Optional[float] = None

class CleanupResponse(BaseModel):
    message: str

# Helper function to run the inference script and get the processed image
def run_herdnet_inference(
    image_input_dir: str, # Directory containing the single image, e.g. backend/images/inf_id/
    original_image_filename: str # Filename of the input image, e.g., <uuid>.jpg
) -> Optional[str]:
    """
    Runs the HerdNet inference script synchronously and returns the base64 encoded output image.
    """
    if not os.path.exists(MODEL_PTH_PATH):
        print(f"ERROR: Model file not found at {MODEL_PTH_PATH}")
        return None

    if not os.path.exists(INFER_SCRIPT_PATH):
        print(f"ERROR: Inference script not found at {INFER_SCRIPT_PATH}")
        return None

    command = [
        "python",
        INFER_SCRIPT_PATH,
        image_input_dir,  # 'root' argument for infer-custom.py
        MODEL_PTH_PATH,   # 'pth' argument
        "-device", "cuda",
        # Add other arguments like -size, -over if needed
    ]

    try:
        print(f"Starting synchronous inference for image in {image_input_dir}. Command: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True, check=False)

        if process.returncode == 0:
            print(f"Inference for image in {image_input_dir} completed successfully.")
            print(f"Stdout: {process.stdout}")

            # Find the results directory created by infer-custom.py
            # It creates a dir like 'YYYYMMDD-HHMMSS_HerdNet_results'
            search_pattern = os.path.join(image_input_dir, "*_HerdNet_results")
            results_dirs = glob.glob(search_pattern)

            if not results_dirs:
                print(f"ERROR: No HerdNet results directory found in {image_input_dir}")
                return None
            
            # Assume the first one found is the correct one if multiple (should ideally be one)
            herdnet_results_dir = results_dirs[0]
            
            # Path to the plotted image (should have the same name as original_image_filename)
            output_image_path = os.path.join(herdnet_results_dir, "plots", original_image_filename)

            if os.path.exists(output_image_path):
                with open(output_image_path, "rb") as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                # Optionally, clean up:
                # import shutil
                # shutil.rmtree(image_input_dir, ignore_errors=True) # Removes the whole inference dir
                return encoded_image
            else:
                print(f"ERROR: Output image not found at {output_image_path}")
                return None
        else:
            print(f"Error during inference for image in {image_input_dir}. Return code: {process.returncode}")
            print(f"Stdout: {process.stdout}")
            print(f"Stderr: {process.stderr}")
            return None
            
    except FileNotFoundError:
        print(f"ERROR: Python interpreter or script not found. Ensure Python is in PATH and script exists.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during inference: {str(e)}")
        return None

@router.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    """
    Processes an image synchronously and returns the processed image.
    """
    unique_id = str(uuid.uuid4()) # Still useful for unique temp directory naming
    
    current_inference_image_dir = os.path.join(BASE_IMAGES_DIR, unique_id)
    
    try:
        os.makedirs(current_inference_image_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {current_inference_image_dir}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not create image directory: {str(e)}")

    image_filename = f"{unique_id}.jpg" 
    image_path = os.path.join(current_inference_image_dir, image_filename)
    
    try:
        image_data = base64.b64decode(request.image)
        with open(image_path, "wb") as f:
            f.write(image_data)
    except base64.binascii.Error:
        # import shutil # Consider cleanup
        # shutil.rmtree(current_inference_image_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")
    except Exception as e:
        # import shutil # Consider cleanup
        # shutil.rmtree(current_inference_image_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")

    # Run inference synchronously
    processed_image_base64 = run_herdnet_inference(
        image_input_dir=current_inference_image_dir,
        original_image_filename=image_filename
    )
    
    # --- Optional: Cleanup ---
    # import shutil
    # try:
    #     shutil.rmtree(current_inference_image_dir) # remove the temp dir after processing
    #     print(f"Cleaned up directory: {current_inference_image_dir}")
    # except Exception as e:
    #     print(f"Error cleaning up directory {current_inference_image_dir}: {e}")
    # --- End Optional: Cleanup ---

    if processed_image_base64:
        return InferenceResponse(image=processed_image_base64)
    else:
        # If cleanup was done, the temp dir might be gone.
        # If not, it remains with the original image and any partial results.
        raise HTTPException(status_code=500, detail="Inference failed or output image not found.")

# The get_inference_status and get_inference_results endpoints are less relevant
# for a synchronous flow that directly returns the image.
# They are kept here but would need rethinking if this is the primary mode.

# Clean up the inference directory after the inference is done

@router.post("/cleanup", response_model=CleanupResponse)
def clean_up_inference_directory():
    inference_dir = os.path.join(BASE_IMAGES_DIR)
    if os.path.exists(inference_dir):
        shutil.rmtree(inference_dir)
    return CleanupResponse(message="Inference directory cleaned up")






