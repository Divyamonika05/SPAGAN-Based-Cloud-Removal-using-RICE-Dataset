import os
import yaml
import torch
import numpy as np
import streamlit as st
from PIL import Image
from attrdict import AttrMap
from torchvision import transforms
from cloud_removal.models.gen.SPANet import Generator

# =========================
# CONFIG & MODEL LOADING
# =========================
CONFIG_PATH = "cloud_removal/results/pretrained_models/RICE1/config.yml"
MODEL_PATH = "cloud_removal/results/pretrained_models/RICE1/gen_model_epoch_200.pth"

# Load YAML config
with open(CONFIG_PATH, "r", encoding="UTF-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = AttrMap(config)

# =========================
# DEVICE SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.gpu_ids = [0] if torch.cuda.is_available() else []

# Load generator model
generator = Generator(gpu_ids=config.gpu_ids)
state_dict = torch.load(MODEL_PATH, map_location=device)
generator.load_state_dict(state_dict)
generator.to(device)
generator.eval()

# =========================
# HELPER FUNCTIONS
# =========================
def preprocess_image(img):
    """Convert PIL image to normalized tensor"""
    transform = transforms.Compose([
        transforms.Resize((config.width, config.width)),
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor

def postprocess_image(tensor):
    """Convert tensor back to PIL image"""
    tensor = tensor.squeeze(0).detach().cpu().numpy()
    tensor = np.clip(tensor, 0, 1)
    tensor = np.transpose(tensor, (1, 2, 0))
    img = Image.fromarray((tensor * 255).astype(np.uint8))
    return img

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="🌤️ Cloud Removal - SpaGAN", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 2rem;
    border-radius: 15px;
}
.title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: bold;
    color: #0d47a1;
    margin-bottom: 0.5rem;
}
.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #1565c0;
    margin-bottom: 2rem;
}
.stButton>button {
    background-color: #1976d2;
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-size: 1rem;
    font-weight: 500;
}
.stButton>button:hover {
    background-color: #0d47a1;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Info
st.sidebar.header("Project Info")
st.sidebar.markdown("""
**Project:** GAN-Based Cloud Removal for Enhanced Satellite Imagery  
**Model:** SpaGAN (Spatial Attention GAN)  
**Dataset:** RICE
""")

# Title Section
st.markdown('<div class="title">🌤️ GAN-Based Cloud Removal</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a cloudy satellite image to get a clean, cloud-free version using SpaGAN.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload a cloudy image", type=["png", "jpg", "jpeg"])

# =========================
# INFERENCE & DISPLAY
# =========================
if uploaded_file is not None:
    input_img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(input_img, caption="☁️ Input Image", use_container_width=True)

    if st.button("✨ Generate Cloud-Free Image"):
        with st.spinner("Running SpaGAN model..."):
            input_tensor = preprocess_image(input_img)
            with torch.no_grad():
                result = generator(input_tensor)
                if isinstance(result, tuple):
                    att, output = result
                else:
                    output = result
                output_img = postprocess_image(output)

        with col2:
            st.image(output_img, caption="🌤️ Cloud-Free Output", use_container_width=True)

        output_path = "cloud_free_output.png"
        output_img.save(output_path)

        with open(output_path, "rb") as file:
            st.download_button(
                label="⬇️ Download Cloud-Free Image",
                data=file,
                file_name="cloud_free_output.png",
                mime="image/png"
            )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>SpaGAN for Cloud Removal</p>",
    unsafe_allow_html=True
)
