"""
Детекция предметов одежды на фото.

Используем готовую модель сегментации одежды (Segformer, дообученный на
датасете разметки частей гардероба). Модель размечает каждый пиксель фото
меткой класса (футболка, брюки, платье, обувь и т.д.), из масок мы
достаём bounding box для каждого найденного предмета.

Модель скачивается один раз при первом запуске (~350 МБ) и кешируется
локально в ~/.cache/huggingface.
"""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

MODEL_NAME = "mattmdjaga/segformer_b2_clothes"

# Метки модели -> человекочитаемые категории на русском.
# id 0 (background) и части тела без одежды (волосы, лицо, кожа) отбрасываем.
LABEL_MAP = {
    1: "верх (майка/футболка)",
    4: "верхняя одежда (куртка/пальто)",
    5: "платье",
    6: "пальто",
    7: "носки",
    8: "брюки",
    9: "джинсы",
    10: "перчатки",
    11: "юбка",
    12: "капюшон",
    16: "чулки",
    17: "сумка",
    18: "шарф",
}

MIN_AREA_FRACTION = 0.01  # игнорируем совсем мелкие/шумные маски


@dataclass
class DetectedItem:
    label: str
    class_id: int
    bbox: tuple  # (x1, y1, x2, y2) в пикселях исходного фото
    area_fraction: float


class ClothesDetector:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def detect(self, image_bytes: bytes) -> list[DetectedItem]:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = image.size

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits  # (1, num_classes, H', W')
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        seg = upsampled.argmax(dim=1)[0].cpu().numpy()  # (h, w) карта классов

        total_pixels = h * w
        items: list[DetectedItem] = []
        for class_id, label in LABEL_MAP.items():
            mask = seg == class_id
            area = int(mask.sum())
            if area == 0 or area / total_pixels < MIN_AREA_FRACTION:
                continue
            ys, xs = np.where(mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            items.append(
                DetectedItem(
                    label=label,
                    class_id=class_id,
                    bbox=bbox,
                    area_fraction=round(area / total_pixels, 4),
                )
            )

        items.sort(key=lambda i: i.area_fraction, reverse=True)
        return items

    @staticmethod
    def crop(image_bytes: bytes, bbox: tuple) -> bytes:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        x1, y1, x2, y2 = bbox
        # небольшой отступ вокруг предмета, чтобы не обрезать край
        pad = 10
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(image.width, x2 + pad), min(image.height, y2 + pad)
        cropped = image.crop((x1, y1, x2, y2))
        buf = BytesIO()
        cropped.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
