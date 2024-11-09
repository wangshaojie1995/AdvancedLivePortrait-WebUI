from PIL import Image, ImageChops
import requests
import os
import torch
import functools
import numpy as np

from modules.utils.paths import *


TEST_IMAGE_URL = "https://github.com/microsoft/onnxjs-demo/raw/master/src/assets/EmotionSampleImages/sad_baby.jpg"
TEST_VIDEO_URL = "https://github.com/jhj0517/sample-medias/raw/master/vids/human-face/expression01_short.mp4"
TEST_IMAGE_PATH = os.path.join(PROJECT_ROOT_DIR, "tests", "test.png")
TEST_VIDEO_PATH = os.path.join(PROJECT_ROOT_DIR, "tests", "test_expression.mp4")
TEST_EXPRESSION_OUTPUT_PATH = os.path.join(PROJECT_ROOT_DIR, "tests", "edited_expression.png")
TEST_EXPRESSION_AAA = 100


def download_image(url, path):
    if os.path.exists(path):
       return

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Image successfully downloaded to {path}")
    else:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")


def are_images_different(image1_path: str, image2_path: str):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    diff = ImageChops.difference(image1, image2)

    if diff.getbbox() is None:
        return False
    else:
        return True


@functools.lru_cache
def is_cuda_available():
    return torch.cuda.is_available()

