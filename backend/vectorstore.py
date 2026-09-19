"""
Хранилище каталога товаров с поиском по похожести.

Используем Qdrant в embedded-режиме (библиотека qdrant-client хранит
данные в локальной папке, без отдельного сервера/Docker) — этого достаточно
для MVP на десятки-сотни тысяч товаров.
"""

import uuid
from dataclasses import asdict, dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

COLLECTION = "products"
DATA_DIR = "./qdrant_data"


@dataclass
class Product:
    id: str
    title: str
    price: float
    currency: str
    url: str
    image_url: str
    shop: str
    region: str
    category: str


def _point_id(raw_id: str) -> str:
    """Qdrant требует id точки в виде int или UUID — превращаем произвольную
    строку вида 'wb_12345' в стабильный UUID (один и тот же товар всегда
    получает один и тот же id, повторный ingest его просто обновит)"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))


class VectorStore:
    def __init__(self, vector_size: int, path: str = DATA_DIR):
        self.client = QdrantClient(path=path)
        self.vector_size = vector_size
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=vector_size, distance=qmodels.Distance.COSINE
                ),
            )

    def upsert(self, products: list[Product], vectors: list[list[float]]) -> None:
        points = [
            qmodels.PointStruct(id=_point_id(product.id), vector=vector, payload=asdict(product))
            for product, vector in zip(products, vectors)
        ]
        self.client.upsert(collection_name=COLLECTION, points=points)

    def search(
        self, vector: list[float], region: str, category: str, limit: int = 12
    ) -> list[dict]:
        must = [qmodels.FieldCondition(key="category", match=qmodels.MatchValue(value=category))]
        if region != "world":
            must.append(qmodels.FieldCondition(key="region", match=qmodels.MatchValue(value=region)))

        hits = self.client.search(
            collection_name=COLLECTION,
            query_vector=vector,
            query_filter=qmodels.Filter(must=must),
            limit=limit,
        )
        return [hit.payload for hit in hits]

    def count(self) -> int:
        return self.client.count(collection_name=COLLECTION).count
