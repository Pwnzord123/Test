"""
Превращает фото товара (или вырезанный пользователем фрагмент) в вектор
для поиска похожих товаров.

Используем CLIP, дообученный на фото одежды (fashion-clip) — в отличие от
обычного CLIP он лучше различает силуэт, фасон и материал одежды, а не
только общий сюжет фото.

Модель скачивается один раз при первом запуске и кешируется локально в
~/.cache/huggingface.
"""

from io import BytesIO

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "patrickjohncyh/fashion-clip"


class FashionEmbedder:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.dim = self.model.config.projection_dim

    @torch.inference_mode()
    def embed_image(self, image_bytes: bytes) -> list[float]:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output = self.model.get_image_features(**inputs)
        if torch.is_tensor(output):
            features = output
        else:
            # некоторые версии transformers оборачивают результат в объект
            # вида BaseModelOutputWithPooling вместо голого тензора
            features = getattr(output, "image_embeds", None)
            if features is None:
                features = getattr(output, "pooler_output", None)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features[0].cpu().tolist()
