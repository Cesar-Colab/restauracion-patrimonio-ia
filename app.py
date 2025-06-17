import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt

from torchvision.transforms import Compose
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

st.set_page_config(page_title="Restauración y Levantamiento Patrimonial", layout="centered")

@st.cache_resource
def load_depth_model():
    model_path = "models/midas_v21_small-70d6b9c8.pt"
    model = MidasNet_small(
        model_path,
        features=64,
        backbone="efficientnet_lite3",
        exportable=True,
        non_negative=True
    )
    model.eval()
    return model

model = load_depth_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = Compose([
    Resize(384, 384, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32, resize_method="minimal"),
    NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    PrepareForNet()
])

st.title("Restauración y Levantamiento de Fachadas con IA")
st.markdown("Sube una fotografía antigua de una fachada patrimonial para restaurarla y estimar su profundidad.")

uploaded_file = st.file_uploader("Sube una imagen JPG/PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="Imagen original", use_container_width=True)

    restored = cv2.detailEnhance(image_np, sigma_s=10, sigma_r=0.15)
    st.image(restored, caption="Imagen restaurada (simulada)", use_container_width=True)

    input_image = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB) / 255.0
    input_sample = transform({"image": input_image})
    input_tensor = torch.from_numpy(input_sample["image"]).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=image_np.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    plt.figure(figsize=(8, 6))
    plt.imshow(depth_normalized, cmap='inferno')
    plt.axis('off')
    st.pyplot(plt)

    st.markdown("### Resultado: Mapa de profundidad estimado")
    st.success("Proceso completado con éxito.")
else:
    st.info("Esperando imagen...")