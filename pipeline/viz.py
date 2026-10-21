"""
Hexagon map visualization — local stand-in for the product site map.

Production paints scored H3 cells as colored polygons (High / Medium / Low).
This module writes the same idea to a standalone Folium HTML file you can
open in a browser without the full Froncort frontend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from utils import ARTIFACTS_DIR, ensure_dirs, logger

# Match product-ish risk coloring
_TIER_COLORS = {
    "High": "#c0392b",
    "Medium": "#e67e22",
    "Low": "#27ae60",
}


def _h3_boundary(h3_index: str) -> Optional[list]:
    """Return [[lat, lng], ...] ring for an H3 cell, or None if h3 unavailable."""
    try:
        import h3
    except ImportError:
        return None
    # h3-py v4 vs v3 API
    if hasattr(h3, "cell_to_boundary"):
        boundary = h3.cell_to_boundary(h3_index)
        # v4 returns (lat, lng) tuples
        return [[float(lat), float(lng)] for lat, lng in boundary]
    if hasattr(h3, "h3_to_geo_boundary"):
        boundary = h3.h3_to_geo_boundary(h3_index, geo_json=False)
        return [[float(lat), float(lng)] for lat, lng in boundary]
    return None


def _centroid_from_h3(h3_index: str) -> Optional[tuple]:
    """(lat, lng) centroid of a cell — used when recommendations lack lat/lng."""
    try:
        import h3
    except ImportError:
        return None
    if hasattr(h3, "cell_to_latlng"):
        return tuple(h3.cell_to_latlng(h3_index))
    if hasattr(h3, "h3_to_geo"):
        return tuple(h3.h3_to_geo(h3_index))
    return None


def attach_hex_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure lat/lng columns exist (from H3 if missing) for map / JSON export."""
    out = df.copy()
    need = ("lat" not in out.columns) or out["lat"].isna().all()
    if not need:
        return out
    lats, lngs = [], []
    for h in out["h3_index"].astype(str):
        c = _centroid_from_h3(h)
        if c is None:
            lats.append(None)
            lngs.append(None)
        else:
            lats.append(c[0])
            lngs.append(c[1])
    out["lat"] = lats
    out["lng"] = lngs
    return out


def render_hexagon_map(
    recommendations: pd.DataFrame,
    out_path: Optional[Path] = None,
    title: str = "TB hotspot recommendations (H3)",
    zoom_start: int = 12,
) -> Path:
    """
    Draw scored H3 hexagons as colored polygons (Folium HTML).

    Requires: pip install folium
    Open the returned .html file in a browser — same idea as the main site map.
    """
    try:
        import folium
    except ImportError as exc:
        raise ImportError(
            "folium is required for hexagon maps. Install with: pip install folium"
        ) from exc

    ensure_dirs()
    out_path = Path(out_path) if out_path else ARTIFACTS_DIR / "hotspot_map.html"
    recs = attach_hex_centroids(recommendations)

    # Map center: mean of available centroids
    valid = recs.dropna(subset=["lat", "lng"])
    if valid.empty:
        raise ValueError(
            "No lat/lng available to center the map. Install h3 so centroids "
            "can be derived from h3_index, or provide lat/lng on recommendations."
        )
    center = [float(valid["lat"].mean()), float(valid["lng"].mean())]

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")
    title_html = f'<h4 style="margin:8px;font-family:sans-serif">{title}</h4>'
    m.get_root().html.add_child(folium.Element(title_html))

    drawn = 0
    for _, row in recs.iterrows():
        h3_index = str(row["h3_index"])
        tier = str(row.get("risk_tier", "Low"))
        color = _TIER_COLORS.get(tier, "#7f8c8d")
        boundary = _h3_boundary(h3_index)
        popup = (
            f"<b>#{int(row.get('rank', 0))} {tier}</b><br>"
            f"h3: {h3_index}<br>"
            f"predicted: {float(row.get('predicted_cases', 0)):.2f}<br>"
            f"gate: {float(row.get('gate_probability', 0)):.2f}<br>"
            f"{row.get('rationale', '')}"
        )
        if boundary:
            folium.Polygon(
                locations=boundary,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.45,
                popup=folium.Popup(popup, max_width=320),
                tooltip=f"{tier} · rank {row.get('rank', '')}",
            ).add_to(m)
            drawn += 1
        elif pd.notna(row.get("lat")) and pd.notna(row.get("lng")):
            # Fallback marker if boundary API unavailable
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lng"])],
                radius=8,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=popup,
            ).add_to(m)
            drawn += 1

    m.save(str(out_path))
    logger.info("Hexagon map (%d cells) -> %s", drawn, out_path)
    return out_path


def render_aoi_outline(
    hexagons: pd.DataFrame,
    out_path: Optional[Path] = None,
    title: str = "AOI hexagon list (hexagons.csv)",
) -> Path:
    """
    Plot every cell in hexagons.csv so you can verify the region coverage
    before training — useful when bringing your own district AOI.
    """
    try:
        import folium
    except ImportError as exc:
        raise ImportError("pip install folium") from exc

    ensure_dirs()
    out_path = Path(out_path) if out_path else ARTIFACTS_DIR / "aoi_hexagons_map.html"
    hx = attach_hex_centroids(hexagons.rename(columns={"h3": "h3_index"}) if "h3" in hexagons.columns else hexagons)
    if "h3_index" not in hx.columns:
        raise ValueError("hexagons table needs an h3_index column")
    valid = hx.dropna(subset=["lat", "lng"])
    center = [float(valid["lat"].mean()), float(valid["lng"].mean())]
    m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")
    m.get_root().html.add_child(
        folium.Element(f'<h4 style="margin:8px;font-family:sans-serif">{title}</h4>')
    )
    for _, row in hx.iterrows():
        boundary = _h3_boundary(str(row["h3_index"]))
        if boundary:
            folium.Polygon(
                locations=boundary,
                color="#2980b9",
                weight=1,
                fill=True,
                fill_opacity=0.15,
                tooltip=str(row["h3_index"]),
            ).add_to(m)
    m.save(str(out_path))
    logger.info("AOI outline map (%d hexes) -> %s", len(hx), out_path)
    return out_path
