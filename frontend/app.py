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
    # Create three columns: left sidebar, main content, and right panel
    left_sidebar, main_content, right_panel = st.columns([1, 3, 1])
    
    with right_panel:
        st.header("About")
        st.info(
            "An extension of the HerdNet model is presented by incorporating a CBAM (Convolutional Block Attention Module) with the aim "
            "of improving the detection and counting of animals in dense herds from aerial images. CBAM allows the model to focus its "
            "attention on both spatially and channel relevant regions, enhancing the ability to localize and classify species. "
            "The results show improvements in the call and increased robustness in scenarios with high occlusion and non-uniform distributions."
            "\n\nThis application demonstrates image inference using the modified HerdNet CBAM model through a FastAPI backend. The project is the deployment of the Master in Artificial Intelligence final project."
            
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

                                # Display processed image
                                with st.expander("Processed Image", expanded=True):
                                    # st.subheader("Processed Image")
                                    processed_image_bytes = base64.b64decode(processed_image_base64)
                                    st.image(processed_image_bytes, caption="Processed Image.", use_column_width=True)

                                # Download button for processed image
                                st.download_button(
                                    label="Download Processed Image",
                                    data=processed_image_bytes,
                                    file_name=f"processed_{uploaded_file.name}",
                                    mime=uploaded_file.type
                                )

                                # Display detections data below the image
                                if detections_data:
                                    st.subheader("Detections Data")
                                    try:
                                        detections_df = pd.DataFrame(detections_data).sort_values(by='scores', ascending=False)
                                        st.dataframe(detections_df)
                                        st.write(f"Total detections: {len(detections_df)}")
                                    except Exception as e:
                                        st.warning(f"Could not display detections data as table: {e}")
                                        st.json(detections_data)
                                else:
                                    st.info("No detections data was returned by the backend.")

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

if __name__ == "__main__":
    main() 