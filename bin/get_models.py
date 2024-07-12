import os
import requests

def download_config():
    config_url = "https://raw.githubusercontent.com/roboflow/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    config_path = "./models/GroundingDINO_SwinT_OGC.py"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    if not os.path.exists(config_path):
        response = requests.get(config_url)
        with open(config_path, "wb") as f:
            f.write(response.content)

def download_model():
    model_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    model_path = "./models/groundingdino_swint_ogc.pth"

    if not os.path.exists(model_path):
        response = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(response.content)


if __name__ == "__main__":
    download_config()
    download_model()
    print("Models downloaded successfully.")