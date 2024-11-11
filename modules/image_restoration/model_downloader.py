from modules.live_portrait.model_downloader import download_model

MODELS_REALESRGAN_URL = {
    "realesr-general-x4v3": "https://huggingface.co/jhj0517/realesr-general-x4v3/resolve/main/realesr-general-x4v3.pth",
}


def download_resrgan_model(file_path, url):
    return download_model(file_path, url)
