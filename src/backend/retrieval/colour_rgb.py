"""Colour-name -> canonical colour_group via RGB nearest-neighbour. The right normaliser for
colour spans: MiniLM embeddings lack colour semantics (oxblood->Orange, saffron->Silver), but a
colour-name database (matplotlib XKCD 949 names + CSS4 + a few extras) gives real RGB, and the
canonical colour_group values map to RGB too, so nearest-by-distance generalises to unseen
colour words. Used by the tagger's stage-2 linker for field=colour_group.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import matplotlib.colors as mc

_XKCD = {k.split(":", 1)[1]: mc.to_rgb(v) for k, v in mc.XKCD_COLORS.items()}
_CSS = {k: mc.to_rgb(v) for k, v in mc.CSS4_COLORS.items()}

# surface words absent from the DBs
_SURFACE_EXTRA = {
    "oxblood": "#4a0000", "vermilion": "#e34234", "champagne": "#f7e7ce", "ecru": "#c2b280",
    "oatmeal": "#d3c7a2", "gunmetal": "#2a3439", "pewter": "#8a9a9a",
}
# canonical colour_group values not in CSS4
_CANON_EXTRA = {
    "off white": "#faf0e6", "light beige": "#f5e6c8", "dark beige": "#c8ad7f",
    "dark pink": "#e75480", "greenish khaki": "#8f9779", "yellowish brown": "#a0522d",
    "light purple": "#c3a6e0", "dark purple": "#4b0082", "other purple": "#9370db",
    "light turquoise": "#afeeee", "dark turquoise": "#00ced1", "light green": "#90ee90",
    "light red": "#ff7f7f", "light yellow": "#ffffe0", "dark yellow": "#b5a642",
    "light orange": "#ffb347", "bronze/copper": "#b87333", "greyish beige": "#bdb3a3",
    "other blue": "#4169e1", "other green": "#2e8b57", "other orange": "#ff8c00",
    "other pink": "#ff69b4", "other red": "#cd5c5c", "other yellow": "#ffd700",
    "other turquoise": "#40e0d0",
}
_WORD = re.compile(r"[a-z]+")


def _rgb_of(name: str) -> Optional[tuple]:
    n = name.lower().strip()
    if n in _SURFACE_EXTRA:
        return mc.to_rgb(_SURFACE_EXTRA[n])
    if n in _CANON_EXTRA:
        return mc.to_rgb(_CANON_EXTRA[n])
    if n in _XKCD:
        return _XKCD[n]
    flat = n.replace(" ", "")
    if flat in _CSS:
        return _CSS[flat]
    return None


def surface_rgb(surface: str) -> Optional[tuple]:
    """RGB of a colour span: whole string, else any single word (last colour word wins)."""
    rgb = _rgb_of(surface)
    if rgb:
        return rgb
    for w in reversed(_WORD.findall(surface.lower())):
        rgb = _rgb_of(w)
        if rgb:
            return rgb
    return None


class ColourRGBLinker:
    def __init__(self, valid_colours: List[str]):
        self.anchors: Dict[str, tuple] = {}
        for c in valid_colours:
            rgb = _rgb_of(c)
            if rgb:
                self.anchors[c] = rgb

    def link(self, surface: str) -> Optional[str]:
        rgb = surface_rgb(surface)
        if rgb is None or not self.anchors:
            return None
        return min(self.anchors, key=lambda c: sum((a - b) ** 2 for a, b in zip(self.anchors[c], rgb)))
