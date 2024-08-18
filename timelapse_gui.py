import streamlit as st
import torch
from PIL import Image
import numpy as np
import io
import sys
import os

# Add the directory containing your StyleGAN code to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your StyleGAN classes
from first import StyleGANGenerator

# Set up the Streamlit page
st.set_page_config(page_title="StyleGAN Timelapse Generator", layout="wide")
st.title("StyleGAN Timelapse Generator")

# Sidebar for model parameters
st.sidebar.header("Model Parameters")
latent_dim = st.sidebar.slider("Latent Dimension", min_value=10, max_value=500, value=100, step=10)
num_frames = st.sidebar.slider("Number of Frames", min_value=1, max_value=20, value=5, step=1)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Generate Timelapse")
    if st.button("Generate"):
        # Initialize the generator
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = StyleGANGenerator(latent_dim, num_frames).to(device)
        
        # Load the trained model (you'll need to update this path)
        generator.load_state_dict(torch.load('"C:\coding\ANN_CP\generator_epoch_400.pth"', map_location=device))
        generator.eval()

        # Generate a sample
        with torch.no_grad():
            z = torch.randn(1, latent_dim, device=device)
            fake_sequence = generator(z)

        # Convert to images and display
        images = []
        for i in range(num_frames):
            img = fake_sequence[0, i].cpu().numpy()
            img = (img * 0.5 + 0.5) * 255
            img = img.transpose(1, 2, 0).astype('uint8')
            images.append(Image.fromarray(img))

        # Display the generated images
        st.image(images, caption=[f"Frame {i+1}" for i in range(num_frames)], width=200)

with col2:
    st.header("About")
    st.write("""
    This application uses a StyleGAN model to generate timelapse sequences.
    
    To use:
    1. Adjust the parameters in the sidebar if desired.
    2. Click the 'Generate' button to create a new timelapse sequence.
    3. The generated frames will appear on the left.
    
    Note: This is a demo GUI and does not include the full functionality of training or saving the model.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("StyleGAN Timelapse Generator v1.0")
