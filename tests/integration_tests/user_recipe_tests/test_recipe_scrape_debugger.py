from types import SimpleNamespace

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from mealie.services.scraper.scraper_strategies import RecipeScraperPackage
from tests.utils import api_routes
from tests.utils.fixture_schemas import TestUser

SCRAPER_FIELDS = {
    "title": "Reisnudeln mit glasierten Hackbällchen",
    "ingredients": ["300 g Breite Reisnudeln", "1 Bund Frühlingszwiebeln"],
    "instructions_list": ["Nudeln kochen.", "Hackbällchen braten."],
}


def stub_scraper(monkeypatch: MonkeyPatch, schema_data: dict) -> None:
    """Site-specific recipe-scrapers classes (e.g. picnic.app) parse the page themselves and leave
    the schema.org data empty; schema based scrapers fill it and to_json() is never needed."""

    async def fake_scrape_url(self):
        return SimpleNamespace(schema=SimpleNamespace(data=schema_data), to_json=lambda: SCRAPER_FIELDS)

    monkeypatch.setattr(RecipeScraperPackage, "scrape_url", fake_scrape_url)


def test_scrape_debugger_returns_schema_data(api_client: TestClient, unique_user: TestUser, monkeypatch: MonkeyPatch):
    schema_data = {"@type": "Recipe", "name": "Schema Recipe", "recipeIngredient": ["1 egg"]}
    stub_scraper(monkeypatch, schema_data)

    response = api_client.post(
        api_routes.recipes_test_scrape_url,
        json={"url": "https://example.com/recipe", "useOpenAI": False},
        headers=unique_user.token,
    )

    assert response.status_code == 200
    assert response.json() == schema_data


def test_scrape_debugger_falls_back_to_scraper_fields(
    api_client: TestClient, unique_user: TestUser, monkeypatch: MonkeyPatch
):
    """Regression: the debugger showed `{}` for sites handled by a site-specific scraper although
    importing the same URL works, because it only ever returned the (empty) schema.org data."""
    stub_scraper(monkeypatch, schema_data={})

    response = api_client.post(
        api_routes.recipes_test_scrape_url,
        json={"url": "https://picnic.app/de/go/2bzq6bj", "useOpenAI": False},
        headers=unique_user.token,
    )

    assert response.status_code == 200
    assert response.json() == SCRAPER_FIELDS
