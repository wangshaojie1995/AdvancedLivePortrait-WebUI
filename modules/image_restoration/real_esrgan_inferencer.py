import os.path
import gradio as gr
import torch
from PIL import Image
import numpy as np
from typing import Optional
from RealESRGAN import RealESRGAN

from modules.utils.paths import *
from .model_downloader import *


class RealESRGANInferencer:
    def __init__(self,
                 model_dir: str = MODELS_REAL_ESRGAN_DIR,
                 output_dir: str = OUTPUTS_DIR):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.device = self.get_device()
        self.model = None
        self.available_models = list(MODELS_REALESRGAN_URL.keys())

    def load_model(self,
                   model_name: Optional[str] = None,
                   scale: int = 1,
                   progress: gr.Progress = gr.Progress()):
        if model_name is None:
            model_name = "realesr-general-x4v3"
        if not model_name.endswith(".pth"):
            model_name += ".pth"
        model_path = os.path.join(self.model_dir, model_name)

        if not os.path.exists(model_path):
            progress(0, f"Downloading RealESRGAN model to : {model_path}")
            name, ext = os.path.splitext(model_name)
            download_resrgan_model(model_path, MODELS_REALESRGAN_URL[name])

        if self.model is None:
            self.model = RealESRGAN(self.device, scale=scale)
            self.model.load_weights(model_path=model_path, download=False)

    def restore_image(self,
                      img_path: str,
                      overwrite: bool = True):
        if self.model is None:
            self.load_model()

        try:
            img = Image.open(img_path).convert('RGB')
            sr_img = self.model.predict(img)
            if overwrite:
                output_path = img_path
            else:
                output_path = get_auto_incremental_file_path(self.output_dir, extension="png")
            sr_img.save(output_path)
        except Exception as e:
            raise

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
