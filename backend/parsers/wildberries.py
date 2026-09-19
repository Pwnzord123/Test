"""
Парсер Wildberries.

WB отдаёт результаты поиска через свой внутренний JSON-API (тот же,
которым пользуется сайт). Это не официальный публичный API — WB
периодически меняет версию эндпоинта и структуру ответа, так что при
поломке парсера первым делом проверяйте актуальный URL через вкладку
Network в браузере на wildberries.ru при поиске.

Работает без Selenium — обычный requests, что быстрее и стабильнее.
"""

import requests
from tqdm import tqdm

from vectorstore import Product

SEARCH_URL = "https://search.wb.ru/exactmatch/ru/common/v9/search"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    )
}


def search(query: str, category: str, max_items: int = 100) -> list[Product]:
    """Ищет товары по текстовому запросу (например 'джинсовая куртка мужская')
    и помечает их переданной категорией (из detection.LABEL_MAP)."""

    products: list[Product] = []
    page = 1
    per_page = 100

    with tqdm(total=max_items, desc=f"WB: {query}") as pbar:
        while len(products) < max_items:
            params = {
                "ab_testing": "false",
                "appType": "1",
                "curr": "rub",
                "dest": "-1257786",  # Москва; влияет на цены/наличие
                "page": page,
                "query": query,
                "resultset": "catalog",
                "sort": "popular",
                "spp": "30",
                "suppressSpellcheck": "false",
            }
            resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            items = data.get("products", [])
            if not items:
                break

            for item in items:
                price_kopeks = (
                    item.get("sizes", [{}])[0]
                    .get("price", {})
                    .get("product", 0)
                )
                nm_id = item["id"]
                # схема формирования картинки по id товара (vol/part вычисляются из id)
                vol = nm_id // 100000
                part = nm_id // 1000
                image_url = (
                    f"https://basket-{_basket_host(vol)}.wbbasket.ru/vol{vol}/"
                    f"part{part}/{nm_id}/images/c516x688/1.webp"
                )
                products.append(
                    Product(
                        id=f"wb_{nm_id}",
                        title=item.get("name", ""),
                        price=price_kopeks / 100,
                        currency="RUB",
                        url=f"https://www.wildberries.ru/catalog/{nm_id}/detail.aspx",
                        image_url=image_url,
                        shop="wildberries",
                        region="ru",
                        category=category,
                    )
                )
                pbar.update(1)
                if len(products) >= max_items:
                    break

            page += 1
            if page > 20:  # защита от бесконечного цикла
                break

    return products


def _basket_host(vol: int) -> str:
    """WB раскладывает картинки по серверам basket-01..basket-24 в
    зависимости от диапазона vol. Таблица периодически расширяется —
    актуальные границы смотрите в любом открытом WB-парсере на GitHub."""
    ranges = [
        (0, 143, "01"), (144, 287, "02"), (288, 431, "03"), (432, 719, "04"),
        (720, 1007, "05"), (1008, 1061, "06"), (1062, 1115, "07"), (1116, 1169, "08"),
        (1170, 1313, "09"), (1314, 1601, "10"), (1602, 1655, "11"), (1656, 1919, "12"),
        (1920, 2045, "13"), (2046, 2189, "14"), (2190, 2405, "15"), (2406, 2621, "16"),
        (2622, 2837, "17"), (2838, 3053, "18"), (3054, 3269, "19"), (3270, 3485, "20"),
    ]
    for lo, hi, host in ranges:
        if lo <= vol <= hi:
            return host
    return "21"
