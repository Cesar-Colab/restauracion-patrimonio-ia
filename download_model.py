import os
import urllib.request

os.makedirs("models", exist_ok=True)
model_url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small-70d6b9c8.pt"
save_path = "models/midas_v21_small-70d6b9c8.pt"

if not os.path.exists(save_path):
    print("Descargando modelo MiDaS...")
    urllib.request.urlretrieve(model_url, save_path)
    print("Modelo descargado correctamente.")
