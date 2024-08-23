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

# Sidebar for model parameters and input
st.sidebar.header("Input Parameters")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
start_time = st.sidebar.time_input("Start Time")
end_time = st.sidebar.time_input("End Time")
season = st.sidebar.selectbox("Select Season", ["Spring", "Summer", "Autumn", "Winter"])

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Generate Timelapse")
    if st.button("Generate") and uploaded_file is not None:
        # Load and display the input image
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Input Image", use_column_width=True)

        # Initialize the generator
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = StyleGANGenerator(latent_dim=100, num_frames=10).to(device)  # Adjust parameters as needed

        # Load the trained model (you'll need to update this path)
        generator.load_state_dict(torch.load("C:/coding/ANN_CP/generator_epoch_400.pth", map_location=device))
        generator.eval()

        # Generate timelapse (placeholder logic - replace with actual StyleGAN logic)
        with torch.no_grad():
            z = torch.randn(1, 100, device=device)
            fake_sequence = generator(z)

        # Convert to images and display
        images = []
        for i in range(10):  # Assuming 10 frames
            img = fake_sequence[0, i].cpu().numpy()
            img = (img * 0.5 + 0.5) * 255
            img = img.transpose(1, 2, 0).astype('uint8')
            images.append(Image.fromarray(img))

        # Display the generated images
        st.image(images, caption=[f"Frame {i+1}" for i in range(10)], width=200)

with col2:
    st.header("About")
    st.write("""
    This application uses a StyleGAN model to generate timelapse sequences based on an input image.
    
    To use:
    1. Upload an image using the sidebar.
    2. Set the start and end times for the timelapse.
    3. Select the season for the timelapse.
    4. Click the 'Generate' button to create a new timelapse sequence.
    5. The generated frames will appear on the left.
    
    Note: This is a demo GUI and the actual StyleGAN transformation logic needs to be implemented.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("StyleGAN Timelapse Generator v1.0")
