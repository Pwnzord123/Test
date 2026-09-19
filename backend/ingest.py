"""
Оффлайн-наполнение каталога товаров.

Запуск: python ingest.py

Проходит по списку (поисковый запрос -> категория), тянет товары через
парсеры магазинов, считает эмбеддинг картинки каждого товара через
FashionEmbedder и складывает всё в векторную базу. Запускайте регулярно
(например по крону) — иначе цены/наличие в каталоге устареют.
"""

import requests

from embeddings import FashionEmbedder
from parsers import ozon, wildberries
from vectorstore import VectorStore

# (поисковый запрос для магазина, категория из detection.LABEL_MAP)
QUERIES = [
    ("джинсовая куртка мужская", "верхняя одежда (куртка/пальто)"),
    ("пальто женское", "пальто"),
    ("платье", "платье"),
    ("джинсы мужские", "джинсы"),
    ("брюки классические", "брюки"),
    ("юбка", "юбка"),
    ("кроссовки", "обувь"),
    ("сумка женская", "сумка"),
]

ITEMS_PER_QUERY = 60


def main() -> None:
    embedder = FashionEmbedder()
    store = VectorStore(vector_size=embedder.dim)

    for query, category in QUERIES:
        for parser in (wildberries, ozon):
            products = parser.search(query, category, max_items=ITEMS_PER_QUERY)

            kept, vectors = [], []
            for product in products:
                try:
                    image_bytes = requests.get(product.image_url, timeout=10).content
                    vectors.append(embedder.embed_image(image_bytes))
                    kept.append(product)
                except Exception as exc:
                    print(f"пропуск {product.id}: {exc}")

            if kept:
                store.upsert(kept, vectors)
                print(f"{parser.__name__.split('.')[-1]}: добавлено {len(kept)} товаров ({query})")

    print(f"Готово. Всего в каталоге: {store.count()}")


if __name__ == "__main__":
    main()
