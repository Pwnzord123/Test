"""
Парсер Wildberries.

WB отдаёт результаты поиска через свой внутренний JSON-API (тот же,
которым пользуется сайт). Это не официальный публичный API — WB
периодически меняет версию эндпоинта и структуру ответа, так что при
поломке парсера первым делом проверяйте актуальный URL через вкладку
Network в браузере на wildberries.ru при поиске.

Работает без Selenium — обычный requests, что быстрее и стабильнее.
"""

import time

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

DELAY_BETWEEN_PAGES_SECONDS = 1.0
MAX_RATE_LIMIT_RETRIES = 4


def _get_with_backoff(url: str, params: dict) -> requests.Response:
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        wait = float(resp.headers.get("Retry-After", 2 * (attempt + 1)))
        time.sleep(wait)
    resp.raise_for_status()
    return resp


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
            resp = _get_with_backoff(SEARCH_URL, params)
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
                host = _resolve_basket_host(vol, part, nm_id)
                image_url = (
                    f"https://basket-{host}.wbbasket.ru/vol{vol}/"
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
            time.sleep(DELAY_BETWEEN_PAGES_SECONDS)

    return products


# Какой basket-NN.wbbasket.ru обслуживает конкретный vol — не документировано
# и периодически меняется (WB добавляет новые сервера). Вместо зашитой
# таблицы диапазонов, которая регулярно устаревает, определяем рабочий сервер
# пробой один раз на каждый уникальный vol и кешируем результат — так парсер
# не ломается, когда WB в очередной раз расширяет список серверов.
_BASKET_HOSTS = [f"{i:02d}" for i in range(1, 61)]
_basket_host_cache: dict[int, str] = {}


def _resolve_basket_host(vol: int, part: int, nm_id: int) -> str:
    if vol in _basket_host_cache:
        return _basket_host_cache[vol]

    for host in _BASKET_HOSTS:
        url = f"https://basket-{host}.wbbasket.ru/vol{vol}/part{part}/{nm_id}/images/c516x688/1.webp"
        try:
            resp = requests.head(url, timeout=3)
        except requests.RequestException:
            continue
        if resp.status_code == 200:
            _basket_host_cache[vol] = host
            return host

    # ни один сервер не отозвался — берём первый как запасной вариант,
    # а не падаем; embed-шаг в ingest.py и так пропустит нерабочую картинку
    _basket_host_cache[vol] = _BASKET_HOSTS[0]
    return _BASKET_HOSTS[0]
