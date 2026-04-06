"""CJK font configuration for matplotlib.

Call ``configure_cjk_fonts()`` once before plotting to enable Chinese
character rendering.  The function prepends known CJK font families to
matplotlib's ``font.sans-serif`` list so that Chinese/Japanese/Korean
glyphs render correctly instead of showing as squares (tofu).
"""

import warnings

import matplotlib
import matplotlib.font_manager as fm

# Suppress "Glyph N missing from font(s)" warnings at import time.
# CJK fonts lack some symbol glyphs (subscript digits, check marks, etc.)
# and non-CJK fonts lack CJK glyphs — matplotlib renders them via fallback,
# but emits a noisy UserWarning for each missing glyph.
warnings.filterwarnings(
    "ignore", message="Glyph.*missing from font", category=UserWarning
)

# CJK font candidates in preference order (common across platforms).
_CJK_CANDIDATES = [
    # macOS
    "PingFang SC",
    "PingFang HK",
    "Hiragino Sans GB",
    "Hiragino Sans",
    "Heiti SC",
    "Heiti TC",
    "STHeiti",
    "Songti SC",
    # Windows
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    # Linux
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "WenQuanYi Micro Hei",
    # Cross-platform fallback
    "Arial Unicode MS",
]

_configured = False


def configure_cjk_fonts():
    """Prepend available CJK fonts to matplotlib's sans-serif list.

    Safe to call multiple times — only runs once.
    """
    global _configured
    if _configured:
        return

    available = {f.name for f in fm.fontManager.ttflist}
    cjk_found = [f for f in _CJK_CANDIDATES if f in available]

    if cjk_found:
        current = matplotlib.rcParams["font.sans-serif"]
        matplotlib.rcParams["font.sans-serif"] = cjk_found + current
        matplotlib.rcParams["axes.unicode_minus"] = False

    _configured = True
