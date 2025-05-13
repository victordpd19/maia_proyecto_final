from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uuid
import base64
import os
import subprocess
import glob
import shutil
import json
import pandas as pd
import logging

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

router = APIRouter(tags=["inference"])

# --- Constants (Consider moving to a configuration file) ---
MODEL_PTH_PATH = "app/herdnet/full_model.pth"  # Placeholder: Ensure this path is correct
BASE_IMAGES_DIR = "app/images"  # Base directory for storing inference images
INFER_SCRIPT_PATH = "app/herdnet/infer-custom.py"  # Path to the inference script
# --- End Constants ---

class InferenceRequest(BaseModel):
    image: str  # Changed from images: List[str] to a single image

class PointData(BaseModel): # Optional: Keep if needed elsewhere or for nested validation
    id: Optional[int] = None
    x: float
    y: float

class InferenceResponse(BaseModel):
    image: str
    detections: Optional[List[Dict[str, Any]]] = None

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
) -> Dict[str, Any]: # Return a dictionary containing image and detections
    """
    Runs the HerdNet inference script synchronously and returns the base64 encoded output image
    and detections data from the CSV (with added point IDs).
    """
    output = {"image": None, "detections": None} # Initialize output dictionary (removed points)

    if not os.path.exists(MODEL_PTH_PATH):
        logger.error(f"Model file not found at {MODEL_PTH_PATH}")
        return output

    if not os.path.exists(INFER_SCRIPT_PATH):
        logger.error(f"Inference script not found at {INFER_SCRIPT_PATH}")
        return output

    command = [
        "python",
        INFER_SCRIPT_PATH,
        image_input_dir,  # 'root' argument for infer-custom.py
        MODEL_PTH_PATH,   # 'pth' argument
        "-device", "cuda",
        # Add other arguments like -size, -over if needed
    ]

    try:
        logger.info(f"Starting synchronous inference for image in {image_input_dir}. Command: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True, check=False)

        if process.returncode == 0:
            logger.info(f"Inference for image in {image_input_dir} completed successfully.")
            if process.stdout:
                logger.info(f"Stdout: {process.stdout.strip()}")
            if process.stderr: # Also log stderr even on success, as it might contain warnings
                logger.warning(f"Stderr (on success): {process.stderr.strip()}")

            # Find the results directory created by infer-custom.py
            # It creates a dir like 'YYYYMMDD-HHMMSS_HerdNet_results'
            search_pattern = os.path.join(image_input_dir, "*_HerdNet_results")
            results_dirs = glob.glob(search_pattern)

            if not results_dirs:
                logger.error(f"No HerdNet results directory found in {image_input_dir} after successful script run.")
                return output
            
            # Assume the first one found is the correct one if multiple (should ideally be one)
            herdnet_results_dir = results_dirs[0]
            
            # --- Find and Read Detections CSV --- 
            csv_search_pattern = os.path.join(herdnet_results_dir, "*_detections.csv")
            csv_files = glob.glob(csv_search_pattern)
            if csv_files:
                detections_csv_path = csv_files[0] # Assume the first one is correct
                try:
                    df_detections = pd.read_csv(detections_csv_path)
                    # Add point_id column (index + 1)
                    df_detections['point_id'] = df_detections.index + 1
                    # Optional: Reorder columns to put point_id first
                    # --- Select only the desired columns --- 
                    desired_columns = ['point_id', 'scores', 'x', 'y', 'species']
                    # Check which desired columns actually exist in the DataFrame to avoid errors
                    columns_to_keep = [col for col in desired_columns if col in df_detections.columns]
                    if len(columns_to_keep) < len(desired_columns):
                        logger.warning(f"Not all desired columns ({desired_columns}) found in CSV. Keeping only: {columns_to_keep}")
                    df_detections = df_detections[columns_to_keep]
                    # --- End column selection ---
                    
                    # Convert NaN to None for JSON compatibility if necessary
                    df_detections = df_detections.where(pd.notnull(df_detections), None)
                    output["detections"] = df_detections.to_dict(orient='records')
                    logger.info(f"Successfully loaded detections data from {detections_csv_path}")
                except pd.errors.EmptyDataError:
                    logger.warning(f"Detections CSV file is empty: {detections_csv_path}")
                    output["detections"] = [] # Return empty list for empty CSV
                except Exception as e:
                    logger.error(f"Failed to read or process detections CSV {detections_csv_path}: {e}")
            else:
                logger.warning(f"Detections CSV file not found in {herdnet_results_dir}")
            # --- End Find and Read Detections CSV ---

            # Path to the plotted image (should have the same name as original_image_filename)
            output_image_path = os.path.join(herdnet_results_dir, "plots", original_image_filename)

            if os.path.exists(output_image_path):
                with open(output_image_path, "rb") as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                output["image"] = encoded_image
                return output
            else:
                logger.error(f"Output image not found at {output_image_path}")
                return output
        else:
            logger.error(f"Error during inference for image in {image_input_dir}. Return code: {process.returncode}")
            if process.stdout:
                logger.error(f"Stdout (on error): {process.stdout.strip()}")
            if process.stderr:
                logger.error(f"Stderr (on error): {process.stderr.strip()}")
            return output
            
    except FileNotFoundError:
        logger.critical(f"Python interpreter or {INFER_SCRIPT_PATH} not found. Ensure Python is in PATH and script exists.")
        return output
    except Exception as e:
        logger.exception(f"An unexpected error occurred during inference: {str(e)}")
        return output

@router.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    """
    Processes an image synchronously and returns the processed image and detections (with point IDs).
    """
    unique_id = str(uuid.uuid4()) # Still useful for unique temp directory naming
    
    current_inference_image_dir = os.path.join(BASE_IMAGES_DIR, unique_id)
    
    try:
        os.makedirs(current_inference_image_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating directory {current_inference_image_dir}: {e}")
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
    inference_result = run_herdnet_inference(
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

    if inference_result["image"]: # Check if image processing was successful
        return InferenceResponse(
            image=inference_result["image"],
            detections=inference_result.get("detections") # Get detections if available
            )
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
        try:
            shutil.rmtree(inference_dir)
            logger.info(f"Successfully cleaned up inference directory: {inference_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up inference directory {inference_dir}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to clean up directory: {str(e)}")
        return CleanupResponse(message="Inference directory cleaned up")
    else:
        logger.info(f"Inference directory {inference_dir} not found, no cleanup needed.")
        return CleanupResponse(message="Inference directory not found, no cleanup needed")






