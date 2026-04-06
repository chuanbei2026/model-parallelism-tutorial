"""PlantUML rendering helper for Jupyter notebooks.

Renders PlantUML diagrams as inline images using the public PlantUML
server. Falls back to displaying raw PlantUML text if rendering fails.
"""

import zlib
import base64
import string
from IPython.display import Image, display, Markdown


# PlantUML uses a custom base64 encoding
_PLANTUML_ALPHABET = (
    string.digits + string.ascii_uppercase + string.ascii_lowercase + "-_"
)
_STD_ALPHABET = (
    string.ascii_uppercase + string.ascii_lowercase + string.digits + "+/"
)
_ENCODE_TABLE = str.maketrans(_STD_ALPHABET, _PLANTUML_ALPHABET)


def _plantuml_encode(text):
    """Encode PlantUML text into the URL-safe format used by the server."""
    compressed = zlib.compress(text.encode("utf-8"), 9)
    # Remove zlib header (first 2 bytes) and checksum (last 4 bytes)
    # Actually PlantUML uses deflate with full zlib wrapper
    encoded = base64.b64encode(compressed[2:-4]).decode("ascii")
    return encoded.translate(_ENCODE_TABLE)


def render_plantuml(source, fmt="svg", server="https://plantuml.corp.amazon.com/plantuml"):
    """Render a PlantUML diagram and display it in the notebook.

    Args:
        source: PlantUML source text (with or without @startuml/@enduml)
        fmt: output format — "svg" (default) or "png"
        server: PlantUML server URL

    Displays the diagram inline. Falls back to showing the raw source
    if rendering fails.
    """
    # Ensure @startuml/@enduml wrapper
    text = source.strip()
    if not text.startswith("@start"):
        text = f"@startuml\n{text}\n@enduml"

    encoded = _plantuml_encode(text)
    url = f"{server}/{fmt}/{encoded}"

    try:
        if fmt == "svg":
            # For SVG, fetch and display inline
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as resp:
                svg_data = resp.read()
            from IPython.display import SVG
            display(SVG(data=svg_data))
        else:
            display(Image(url=url))
    except Exception as e:
        # Fallback: show the source as a code block
        display(Markdown(f"**PlantUML rendering failed** ({e}). Source:\n```\n{text}\n```"))


def plantuml_url(source, fmt="svg", server="https://plantuml.corp.amazon.com/plantuml"):
    """Return the URL for a PlantUML diagram (without rendering).

    Useful for embedding in markdown cells or external tools.
    """
    text = source.strip()
    if not text.startswith("@start"):
        text = f"@startuml\n{text}\n@enduml"
    encoded = _plantuml_encode(text)
    return f"{server}/{fmt}/{encoded}"
