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
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Invoice Data Extractor",
    page_icon="📊",
    layout="wide",
)

def main():
    st.title("📊 Invoice Data Extractor")
    st.markdown("""
    Upload a PDF invoice and extract key information using AI.
    
    This application extracts:
    - Vendor name
    - Items and quantities
    - Total tax amount
    - Total invoice amount
    - Invoice date
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your invoice (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        # Display PDF preview
        st.subheader("Invoice Preview")
        
        # Convert PDF to base64
        pdf_bytes = uploaded_file.getvalue()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        
        # Display PDF download link
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=uploaded_file.name,
            mime="application/pdf"
        )
        
        # Process button
        if st.button("Extract Data"):
            # Call the API to start extraction
            with st.spinner("Starting extraction process..."):
                response = requests.post(
                    f"{API_URL}/extraction/extract",
                    json={"pdf_base64": pdf_base64}
                )
                
                if response.status_code == 200:
                    extraction_data = response.json()
                    extraction_id = extraction_data["extraction_id"]
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Poll for status until completed
                    completed = False
                    while not completed:
                        status_response = requests.get(f"{API_URL}/extraction/status/{extraction_id}")
                        
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            progress = status_data.get("progress", 0)
                            status = status_data.get("status", "")
                            
                            progress_bar.progress(progress)
                            status_text.text(f"Status: {status.capitalize()} - Progress: {int(progress * 100)}%")
                            
                            if status == "completed":
                                completed = True
                            elif status == "failed":
                                st.error("Extraction failed. Please try again.")
                                break
                        else:
                            st.error("Failed to get extraction status.")
                            break
                        
                        time.sleep(1)
                    
                    # If completed, get and display results
                    if completed:
                        results_response = requests.get(f"{API_URL}/extraction/results/{extraction_id}")
                        
                        if results_response.status_code == 200:
                            results = results_response.json()
                            
                            st.success("Extraction completed successfully!")
                            
                            # Display results in a nice format
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Invoice Details")
                                st.write(f"**Vendor:** {results.get('vendor_name', 'N/A')}")
                                st.write(f"**Date:** {results.get('date', 'N/A')}")
                                st.write(f"**Total Tax:** ${results.get('total_tax', 'N/A')}")
                                st.write(f"**Total Amount:** ${results.get('total_amount', 'N/A')}")
                            
                            with col2:
                                st.subheader("Items")
                                if results.get("items"):
                                    # Convert items to DataFrame for better display
                                    items_df = pd.DataFrame(results["items"])
                                    st.dataframe(items_df)
                                else:
                                    st.write("No items found.")
                        else:
                            st.error("Failed to get extraction results.")
                else:
                    st.error(f"Failed to start extraction: {response.text}")

if __name__ == "__main__":
    main() 