import streamlit as st
from streamlit.logger import get_logger
from PIL import Image
import time, requests

LOGGER = get_logger(__name__)

def run():
    # Custom page configurations
    st.set_page_config(
        page_title="Meddy-Buddy",
        page_icon="😷",
        layout="wide",  # You can change this to "centered" or "wide"
        initial_sidebar_state="collapsed",
    )
    
    image = Image.open('img/img1.png')  
    st.image(image)

    st.write("# Diabetes Prediction Using Machine Learning")

    #st.sidebar.success("")

    st.markdown(
        """
        AIML MINI-PROJECT
    """
    )

    st.write("# Information about the Diseases: ")

    st.write("""
Diabetes is a chronic health condition that affects how the body regulates blood sugar (glucose) levels. Glucose is vital for providing energy to cells, and insulin is a hormone produced by the pancreas that helps glucose enter cells from the bloodstream. In diabetes, either the body does not produce enough insulin or cannot effectively use the insulin it produces, leading to elevated levels of glucose in the blood.""")
    

    st.write("**Symptoms:**")
    st.write("- Increased thirst")
    st.write("- Frequent urination")
    st.write("- Extreme hunger")
    st.write("- Unexplained weight loss")
    st.write("- Blurred vision")

if __name__ == "__main__":
    run()
