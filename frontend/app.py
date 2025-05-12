import streamlit as st
import requests
import json
import base64
import os
import time
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv()

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000") # Assuming the backend runs on port 8000

st.set_page_config(
    page_title="Image Inference",
    page_icon="🖼️",
    layout="wide",
)

def main():
    st.title("Image Inference with HerdNet")
    st.markdown("""
    Upload an image to perform inference using the HerdNet model.
    The processed image will be displayed below.
    """)

    # File uploader for images
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image preview
        st.subheader("Uploaded Image Preview")
        # To display the image, we can use st.image directly with the uploaded file object
        # or read bytes and then display. PIL can be used for more control if needed.
        image = Image.open(uploaded_file)
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
                        f"{API_URL}/inference/infer", # Updated to the inference endpoint
                        json={"image": img_base64} # Sending a single image as per backend
                    )

                    if response.status_code == 200:
                        inference_data = response.json()
                        processed_image_base64 = inference_data.get("image")

                        if processed_image_base64:
                            st.success("Image processed successfully!")
                            st.subheader("Processed Image")
                            
                            # Decode the base64 string to bytes
                            processed_image_bytes = base64.b64decode(processed_image_base64)
                            # Display the processed image
                            st.image(processed_image_bytes, caption="Processed Image.", use_column_width=True)
                            
                            # Optional: Add a download button for the processed image
                            st.download_button(
                                label="Download Processed Image",
                                data=processed_image_bytes,
                                file_name=f"processed_{uploaded_file.name}",
                                mime=uploaded_file.type # Use the mime type of the uploaded file
                            )
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
    
    st.sidebar.header("About")
    st.sidebar.info(
        "This application demonstrates image inference using a backend API. "
        "Upload an image, and the system will process it and display the result."
        "\n\nAuthors: Victor Pérez, Jordi Sanchez, Maryi Carvajal, Simón Aristizábal"
        "\n\nUniversidad de los Andes - MAIA"
    )

if __name__ == "__main__":
    main() 