import streamlit as st
import requests
import json
import base64
import os
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import io

# Load environment variables
load_dotenv()

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Image Inference",
    page_icon="🖼️",
    layout="wide",
)

def draw_detections(image, detections, score_threshold):
    # Create a copy of the image to draw on
    img_with_dots = image.copy()
    draw = ImageDraw.Draw(img_with_dots)
    
    # Filter detections by score threshold
    filtered_detections = [d for d in detections if d['scores'] >= score_threshold]
    
    # Draw dots for each detection
    for detection in filtered_detections:
        x, y = detection['x'], detection['y']
        draw.ellipse([x-25, y-25, x+25, y+25], fill='red')
        draw.text((x-10, y-10), f"ID: {detection['point_id']}", fill='white')
    
    return img_with_dots

def main():
    # Initialize session state variables if they don't exist
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'detections_data' not in st.session_state:
        st.session_state.detections_data = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None

    # Create three columns: left sidebar, main content, and right panel
    left_sidebar, main_content, right_panel = st.columns([1, 3, 1])
    
    with right_panel:
        st.header("About")
        st.info("This paper presents an extension of the HerdNet architecture for multi-species wildlife detection and counting in aerial imagery, " \
        "achieved by integrating the Convolutional Block Attention Module (CBAM) and applying the Hard Negative Patching (HNP) technique within a two-stage training scheme. " \
        "Although the HerdNet+CBAM model did not outperform results reported in earlier studies, it nevertheless improved on the baseline implemented " \
        "here—particularly in per-class precision and validation stability. Moreover, the use of HNP helped reduce false positives and enhance discrimination between ambiguous" \
        " background regions and true animal instances. These findings position HerdNet+CBAM as a viable and promising alternative for automated wildlife monitoring in complex environments, " \
        "underscoring the importance of tailoring attention mechanisms to the target domain and complementing architectures with robust training strategies. " \
        "This is the first documented implementation of HerdNet combined with CBAM in this field, providing a solid foundation for future research on ecological monitoring systems using computer vision."
            "\n\nAuthors: Victor Pérez, Jordi Sanchez, Maryi Carvajal, Simón Aristizábal"
            "\n\nUniversidad de los Andes - MAIA"
        )

        # adding repository link with github icon
        st.markdown("[![Repository](https://img.shields.io/badge/Repository-GitHub-blue)](https://github.com/victordpd19/maia_proyecto_final)")

    with left_sidebar:
        st.header("Model Information")
        st.markdown("""
        ### HerdNet CBAM Model
        - **Architecture**: Convolutional Neural Network with CBAM attention, based on the original work by [@Alexandre-Delplanque](https://github.com/Alexandre-Delplanque/HerdNet)
        - **Training Data**: [Dataset from Multispecies detection and identification of African mammals in aerial imagery using convolutional neural networks](https://dataverse.uliege.be/dataset.xhtml?persistentId=doi:10.58119/ULG/MIRUU5)
        """)
        
        # Performance Metrics Table
        st.markdown("### Performance Metrics")
        metrics_data = {
            "Metric": ["MAE", "MSE", "RMSE", "Accuracy", "F1 Score", "mAP", "Precision", "Recall"],
            "Value": [3.79, 62.88, 7.93, 0.932, 0.704, 0.565, 0.634, 0.792]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df.style.format({"Value": "{:.3f}"}))
        
        st.markdown("### Use Cases")
        st.markdown("""
        - Wildlife population monitoring
        - Animal detection in natural habitats
        - Conservation research
        - Ecological studies
        """)
        
        # Technical Details
        st.markdown("""
        ### Technical Details
        - **Input Size**: Any size of the image
        - **Window Size**: 512x512 px
        - **Overlap**: 160 px
        - **Output Classes**: 6 animal species
        - **Model Parameters**: TBD
        """)

    with main_content:
        st.title("Image Inference with HerdNet CBAM")
        st.markdown("""
        Upload an image to perform inference using the modified HerdNet CBAM model.
        The processed image will be displayed below.
        """)

        # File uploader for images
        uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display uploaded image preview in a collapsible section
            with st.expander("Uploaded Image Preview", expanded=False):
                image = Image.open(uploaded_file)
                st.session_state.original_image = image
                st.image(image, caption="Uploaded Image.", use_column_width=True)

            # Convert image to base64
            img_bytes = uploaded_file.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            # Process button
            if st.button("Process Image"):
                # Call the API to perform inference
                with st.spinner("Processing image..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/inference/infer",
                            json={"image": img_base64}
                        )

                        if response.status_code == 200:
                            inference_data = response.json()
                            processed_image_base64 = inference_data.get("image")
                            detections_data = inference_data.get("detections")

                            if processed_image_base64:
                                st.success("Image processed successfully!")
                                st.session_state.processed_image = base64.b64decode(processed_image_base64)
                                st.session_state.detections_data = detections_data

                            else:
                                st.error("Inference completed, but no processed image was returned.")

                        elif response.status_code == 400:
                            error_detail = response.json().get("detail", "Bad request.")
                            st.error(f"Failed to process image (Bad Request): {error_detail}")
                        elif response.status_code == 500:
                            error_detail = response.json().get("detail", "Internal server error.")
                            st.error(f"Failed to process image (Server Error): {error_detail}")
                        else:
                            st.error(f"Failed to process image. Status code: {response.status_code}. Response: {response.text}")
                    
                    except requests.exceptions.RequestException as e:
                        st.error(f"Could not connect to the API: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

        # Display processed image and detections if they exist in session state
        if st.session_state.processed_image is not None:
            with st.expander("Processed Image", expanded=True):
                st.subheader("Processed Image")
                st.image(st.session_state.processed_image, caption="Processed Image.", use_column_width=True)

                # Download button for processed image
                st.download_button(
                    label="Download Processed Image",
                    data=st.session_state.processed_image,
                    file_name="processed_image.jpg",
                    mime="image/jpeg"
                )

                if st.session_state.detections_data:
                    # Add score threshold slider
                    score_threshold = st.slider(
                        "Score Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.01,
                        help="Filter detections by confidence score"
                    )
                    
                    # Filter detections by score threshold
                    filtered_detections = [d for d in st.session_state.detections_data if d['scores'] >= score_threshold]
                    
                    # Display filtered detections count
                    st.write(f"Total detections: {len(filtered_detections)} (filtered from {len(st.session_state.detections_data)} total)")
                    
                    # Recreate image with dots
                    # if len(filtered_detections) > 0 and st.session_state.original_image is not None:
                    st.subheader("Detections Visualization")
                    img_with_dots = draw_detections(st.session_state.original_image, filtered_detections, score_threshold)
                    st.image(img_with_dots, caption="Detections Visualization", use_column_width=True)
                    # Add table with detections data
                    st.subheader("Detections Data")
                    st.dataframe(st.session_state.detections_data, use_container_width=True)

if __name__ == "__main__":
    main() 