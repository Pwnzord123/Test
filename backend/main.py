"""
Backend сайта.

Запуск: uvicorn main:app --reload --port 8000

Эндпоинты:
  POST /detect  — принимает фото, возвращает найденные предметы одежды
                  (категория + рамка на фото), НИЧЕГО не ищет.
  POST /search  — принимает фото + bbox выбранного предмета + регион,
                  возвращает похожие товары со ссылками.
"""

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from detection import ClothesDetector
from embeddings import FashionEmbedder
from vectorstore import VectorStore

app = FastAPI(title="Clothing Finder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для прод-версии сузьте до домена вашего сайта
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели грузятся один раз при старте сервера, а не на каждый запрос —
# иначе каждый запрос будет ждать загрузку модели с диска.
detector = ClothesDetector()
embedder = FashionEmbedder()
store = VectorStore(vector_size=embedder.dim)


@app.post("/detect")
async def detect(photo: UploadFile = File(...)):
    image_bytes = await photo.read()
    items = detector.detect(image_bytes)
    return {
        "items": [
            {
                "label": item.label,
                "class_id": item.class_id,
                "bbox": item.bbox,
                "area_fraction": item.area_fraction,
            }
            for item in items
        ]
    }


@app.post("/search")
async def search(
    photo: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
    category: str = Form(...),
    region: str = Form("ru"),  # "ru" или "world"
    limit: int = Form(12),
):
    image_bytes = await photo.read()
    crop_bytes = ClothesDetector.crop(image_bytes, (x1, y1, x2, y2))
    vector = embedder.embed_image(crop_bytes)
    results = store.search(vector, region=region, category=category, limit=limit)
    return {"results": results}


@app.get("/health")
async def health():
    return {"status": "ok", "items_in_catalog": store.count()}
