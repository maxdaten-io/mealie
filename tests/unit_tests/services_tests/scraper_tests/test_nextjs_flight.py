import json

import pytest

from mealie.lang.providers import get_locale_provider
from mealie.services.scraper import nextjs_flight
from mealie.services.scraper.scraper_strategies import RecipeScraperPackage
from tests import data as test_data

PICNIC_URL = "https://picnic.app/de/rezepte/6a33cd0ee601c856bcf59f7e"

RECIPE = {
    "@context": "https://schema.org",
    "@type": "Recipe",
    "name": "Flight Cake",
    "recipeIngredient": ["1 cup flour", "2 eggs"],
    "recipeInstructions": [{"@type": "HowToStep", "text": "Mix"}, {"@type": "HowToStep", "text": "Bake"}],
}


def flight_html(*rows: str) -> str:
    """Wrap decoded flight rows the way Next.js App Router streams them into the page."""
    scripts = "".join(f"<script>self.__next_f.push([1,{json.dumps(row)}])</script>" for row in rows)
    return f"<html><head><title>t</title></head><body>{scripts}</body></html>"


def test_extract_from_text_row_spanning_chunks():
    """Large strings become T-rows whose raw JSON may be split across several push() calls."""
    ld = json.dumps(RECIPE)
    cut = len(ld) // 2
    html = flight_html('1:I[123,[],""]\n', f"2:T{len(ld):x},", ld[:cut], ld[cut:], '3:["$","script",null,{}]\n')

    assert json.loads(nextjs_flight.extract_recipe_ld_json(html)) == RECIPE


def test_extract_from_inline_json_string():
    """Short strings stay inline as an escaped JSON string prop instead of a T-row."""
    row = "4:" + json.dumps(["$", "script", None, {"dangerouslySetInnerHTML": {"__html": json.dumps(RECIPE)}}]) + "\n"

    assert json.loads(nextjs_flight.extract_recipe_ld_json(flight_html(row))) == RECIPE


def test_extract_skips_non_recipe_objects():
    row = "5:" + json.dumps({"@type": "WebSite", "name": "x", "nested": {"@type": "Recipe", "name": "inner"}}) + "\n"

    assert json.loads(nextjs_flight.extract_recipe_ld_json(flight_html(row))) == {"@type": "Recipe", "name": "inner"}


@pytest.mark.parametrize(
    "html",
    [
        "<html><body><p>no scripts</p></body></html>",
        flight_html('1:["$","div",null,{"children":"no recipe here"}]\n'),
        '<html><body><script>self.__next_f.push([1,"broken \\x escape"])</script></body></html>',
    ],
)
def test_extract_returns_none_without_recipe(html):
    assert nextjs_flight.extract_recipe_ld_json(html) is None


def test_inject_ld_json_into_head():
    html = nextjs_flight.inject_ld_json("<html><head><title>t</title></head><body></body></html>", "{}")

    assert '<script type="application/ld+json">{}</script></head>' in html
    assert nextjs_flight.inject_ld_json("<p>no head</p>", "{}").startswith('<script type="application/ld+json">')


@pytest.mark.asyncio
async def test_package_strategy_recovers_recipe_from_flight_data():
    """picnic.app renders its JSON-LD from a client component, so the server HTML has no ld+json script.
    Without a site-specific scraper recipe-scrapers finds nothing; the strategy must fall back to the flight data."""
    html = test_data.html_reisnudeln_mit_glasierten_hackballchen.read_text(encoding="utf-8")
    strategy = RecipeScraperPackage("https://example.com/recipe", get_locale_provider(), None, raw_html=html)  # type: ignore[arg-type]

    recipe, _ = await strategy.parse()

    assert recipe is not None
    assert recipe.name == "Reisnudeln mit glasierten Hackbällchen"
    assert len(recipe.recipe_ingredient) == 18
    steps = [step.text for step in recipe.recipe_instructions]
    assert steps[0] == "Schritt 1"  # recipe-scrapers keeps the HowToStep names as their own lines
    assert steps[1].startswith("In einem Topf Salzwasser zum Kochen bringen")
    assert steps[-1].startswith("Ein wenig Sriracha macht aus dem Gericht")
    assert recipe.total_time == "20 minutes"
    assert recipe.image and recipe.image.startswith("https://storefront-prod.de.picnicinternational.com/")


@pytest.mark.asyncio
async def test_package_strategy_retries_in_wild_mode_with_recovered_schema(monkeypatch):
    """A site-specific scraper that already failed on the page must not shadow the recovered JSON-LD."""
    html = flight_html(f"1:T{len(json.dumps(RECIPE)):x},", json.dumps(RECIPE))
    strategy = RecipeScraperPackage(PICNIC_URL, get_locale_provider(), None, raw_html=html)  # type: ignore[arg-type]
    calls: list[tuple[str, bool]] = []

    def fake_scrape_schema(recipe_html: str, site_specific: bool = True):
        calls.append((recipe_html, site_specific))
        return "schema" if len(calls) == 2 else None

    monkeypatch.setattr(strategy, "scrape_schema", fake_scrape_schema)

    assert await strategy.scrape_url() == "schema"
    assert [site_specific for _, site_specific in calls] == [True, False]
    assert '<script type="application/ld+json">' not in calls[0][0]
    assert json.loads(nextjs_flight.extract_recipe_ld_json(calls[1][0])) == RECIPE
    assert '<script type="application/ld+json">{"@context"' in calls[1][0]
