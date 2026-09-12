"""Recover schema.org Recipe data from Next.js App Router pages.

Next.js streams React Server Component data to the browser as "flight" rows
embedded in ``<script>self.__next_f.push([1,"..."])</script>`` tags. Sites that
render their ``<script type="application/ld+json">`` from a client component
(e.g. picnic.app) ship the JSON-LD inside those rows instead of the server
rendered HTML, so recipe-scrapers finds no schema on the page although the
data is right there.

Large strings travel as text rows (``<id>:T<hex length>,<raw text>``), which
may be split across several ``push()`` calls; short strings stay inline as
escaped JSON. ``extract_recipe_ld_json`` handles both and returns the first
schema.org ``Recipe`` object as JSON-LD so the regular scraper can be re-run
on ``inject_ld_json(html, ld_json)``.
"""

import json
import re
from collections.abc import Iterator

# One push() call; the payload is a JavaScript string literal that JSON can decode.
FLIGHT_CHUNK = re.compile(r'self\.__next_f\.push\(\[1,\s*"((?:[^"\\]|\\.)*)"\s*\]\)', re.DOTALL)
RECIPE_TYPE = re.compile(r'"@type"\s*:\s*(?:"Recipe"|\[[^\]]*"Recipe")')
# A JSON string literal carrying an escaped "@type" key, i.e. JSON-LD passed around as a string.
ESCAPED_JSON_STRING = re.compile(r'"((?:[^"\\]|\\.)*?\\"@type\\"(?:[^"\\]|\\.)*)"')
MAX_BACKTRACK = 50

_decoder = json.JSONDecoder()


def extract_recipe_ld_json(html: str) -> str | None:
    """Return the first schema.org Recipe found in the page's flight data as JSON-LD, or None."""
    if "__next_f" not in html:
        return None

    recipe = _find_recipe("".join(_decode_chunks(html)), nested=False)
    return json.dumps(recipe, ensure_ascii=False) if recipe else None


def inject_ld_json(html: str, ld_json: str) -> str:
    """Add a JSON-LD script to the page head (or prepend it) so schema based scrapers can read it."""
    script = f'<script type="application/ld+json">{ld_json}</script>'
    head_end = html.find("</head>")
    if head_end == -1:
        return script + html
    return html[:head_end] + script + html[head_end:]


def _decode_chunks(html: str) -> Iterator[str]:
    for match in FLIGHT_CHUNK.finditer(html):
        try:
            yield json.loads(f'"{match.group(1)}"')
        except ValueError:
            continue


def _find_recipe(text: str, nested: bool) -> dict | None:
    for match in RECIPE_TYPE.finditer(text):
        recipe = _enclosing_recipe(text, match.start())
        if recipe is not None:
            return recipe

    if nested:
        return None

    for match in ESCAPED_JSON_STRING.finditer(text):
        try:
            inner = json.loads(f'"{match.group(1)}"')
        except ValueError:
            continue
        recipe = _find_recipe(inner, nested=True)
        if recipe is not None:
            return recipe

    return None


def _enclosing_recipe(text: str, pos: int) -> dict | None:
    """Walk back from a ``"@type": "Recipe"`` match to the ``{`` that opens its object."""
    start = pos
    for _ in range(MAX_BACKTRACK):
        start = text.rfind("{", 0, start)
        if start == -1:
            return None
        try:
            obj, _ = _decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(obj, dict) and _is_recipe(obj):
            return obj
    return None


def _is_recipe(obj: dict) -> bool:
    schema_type = obj.get("@type")
    return schema_type == "Recipe" or (isinstance(schema_type, list) and "Recipe" in schema_type)
