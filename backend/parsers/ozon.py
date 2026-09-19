"""
Парсер Ozon.

У Ozon нет открытого поискового API, как у Wildberries, а карточки товаров
дорисовываются через JS — обычный requests их не увидит. Поэтому используем
headless Chrome через Selenium и разбираем уже отрисованную страницу.

Хрупкое место: вёрстка Ozon меняется без предупреждения, так что CSS-
селекторы ниже могут потребовать обновления. Если парсер перестал находить
товары — откройте страницу поиска в браузере и сверьте селекторы через
DevTools.
"""

import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

from vectorstore import Product

SEARCH_URL = "https://www.ozon.ru/search/?text={query}&page={page}"
PAGE_LOAD_PAUSE_SECONDS = 3  # даём JS время дорисовать карточки товаров
MAX_PAGES = 10


def _make_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def search(query: str, category: str, max_items: int = 100) -> list[Product]:
    """Ищет товары по текстовому запросу и помечает их переданной категорией."""

    products: list[Product] = []
    seen_ids: set[str] = set()
    driver = _make_driver()

    try:
        page = 1
        with tqdm(total=max_items, desc=f"Ozon: {query}") as pbar:
            while len(products) < max_items and page <= MAX_PAGES:
                driver.get(SEARCH_URL.format(query=query, page=page))
                time.sleep(PAGE_LOAD_PAUSE_SECONDS)

                cards = driver.find_elements(
                    By.CSS_SELECTOR, "[data-widget='searchResultsV2'] a[href*='/product/']"
                )
                if not cards:
                    break

                new_on_page = 0
                for card in cards:
                    try:
                        url = card.get_attribute("href").split("?")[0]
                        product_id = url.rstrip("/").split("-")[-1]
                        if product_id in seen_ids:
                            continue

                        title = card.get_attribute("aria-label") or card.text
                        image_url = card.find_element(By.TAG_NAME, "img").get_attribute("src")
                        price_text = card.find_element(
                            By.XPATH, ".//span[contains(text(), '₽')]"
                        ).text
                        price = float("".join(c for c in price_text if c.isdigit()))
                    except Exception:
                        continue  # карточка без цены/картинки/ссылки — пропускаем

                    seen_ids.add(product_id)
                    new_on_page += 1
                    products.append(
                        Product(
                            id=f"ozon_{product_id}",
                            title=title,
                            price=price,
                            currency="RUB",
                            url=url,
                            image_url=image_url,
                            shop="ozon",
                            region="ru",
                            category=category,
                        )
                    )
                    pbar.update(1)
                    if len(products) >= max_items:
                        break

                if new_on_page == 0:
                    break  # страница не дала новых товаров — дальше листать бессмысленно
                page += 1
    finally:
        driver.quit()

    return products
