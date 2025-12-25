import streamlit as st
import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from PIL import Image
import io

# Constants from your notebook
ENCODER = 'timm-efficientnet-b0'
IMAGE_SIZE = 320
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the model class (must match your trained model architecture)
class SegmentationModel(torch.nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.arc = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=None,  # No pretrained weights needed when loading state_dict
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, images, masks=None):
        logits = self.arc(images)
        if masks is not None:
            # Losses not needed for inference, but included for completeness
            from segmentation_models_pytorch.losses import DiceLoss
            loss1 = DiceLoss(mode='binary')(logits, masks)
            loss2 = torch.nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1 + loss2
        return logits

# Load the saved model
@st.cache_resource
def load_model():
    model = SegmentationModel()
    model.load_state_dict(torch.load('saved_model/best_model.pt', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

st.title("Human Image Segmentation App")
st.write("Upload an image to segment the human figures (binary mask output).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess: Resize to 320x320
    resized_image = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    # Prepare for model: Transpose to (C, H, W), normalize, to tensor
    input_tensor = np.transpose(resized_image, (2, 0, 1)).astype(np.float32) / 255.0
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        pred_mask = torch.sigmoid(logits)
        pred_mask = (pred_mask > 0.5).float().cpu().squeeze(0).squeeze(0).numpy()

    # Display results side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(Image.fromarray(resized_image), use_container_width=True)
    with col2:
        st.subheader("Predicted Mask")
        st.image(pred_mask * 255, use_container_width=True, clamp=True)  # Scale to 0-255 for display as grayscale
