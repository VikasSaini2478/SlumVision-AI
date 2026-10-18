"""
GIS export and analytics utilities. Converts a predicted segmentation
mask into GeoTIFF / Shapefile / GeoJSON outputs, and computes area and
cluster statistics for planning/reporting.

If the source image had no CRS/transform (e.g. a plain PNG/JPEG upload
in the dashboard), a pixel-space identity transform + EPSG:4326 label is
used as a placeholder so files remain structurally valid; real-world
area/coordinate outputs require georeferenced input.
"""

from __future__ import annotations
import numpy as np
import cv2

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.features import shapes as rio_shapes
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import geopandas as gpd
    from shapely.geometry import shape as shapely_shape
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


def save_geotiff(mask: np.ndarray, out_path: str, transform=None, crs: str = "EPSG:4326"):
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for GeoTIFF export. `pip install rasterio`.")
    if transform is None:
        transform = Affine.identity()
    h, w = mask.shape
    with rasterio.open(
        out_path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype=mask.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(mask, 1)
    return out_path


def mask_to_geodataframe(mask: np.ndarray, transform=None, crs: str = "EPSG:4326",
                          min_cluster_pixels: int = 25):
    """Vectorize a binary/multi-class mask into polygons, one row per connected region."""
    if not (HAS_RASTERIO and HAS_GEOPANDAS):
        raise ImportError("rasterio and geopandas are required for vector GIS export.")
    if transform is None:
        transform = Affine.identity()

    mask_int = mask.astype(np.int32)
    records = []
    for geom, value in rio_shapes(mask_int, mask=mask_int > 0, transform=transform):
        poly = shapely_shape(geom)
        if poly.area < min_cluster_pixels:
            continue
        records.append({"class": int(value), "area_px": poly.area, "geometry": poly})

    gdf = gpd.GeoDataFrame(records, crs=crs)
    return gdf


def save_shapefile(mask: np.ndarray, out_path: str, transform=None, crs: str = "EPSG:4326",
                    min_cluster_pixels: int = 25):
    gdf = mask_to_geodataframe(mask, transform, crs, min_cluster_pixels)
    gdf.to_file(out_path, driver="ESRI Shapefile")
    return out_path


def save_geojson(mask: np.ndarray, out_path: str, transform=None, crs: str = "EPSG:4326",
                  min_cluster_pixels: int = 25):
    gdf = mask_to_geodataframe(mask, transform, crs, min_cluster_pixels)
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path


def compute_area_statistics(mask: np.ndarray, pixel_area_m2: float | None = None,
                             min_cluster_pixels: int = 25) -> dict:
    """
    Compute slum area and cluster statistics from a binary mask via
    connected-component analysis. If `pixel_area_m2` is known (from the
    source raster's ground sample distance), returns real-world area in
    m^2/hectares; otherwise reports pixel counts only.
    """
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    clusters = []
    for label_id in range(1, num_labels):  # 0 is background
        area_px = stats[label_id, cv2.CC_STAT_AREA]
        if area_px < min_cluster_pixels:
            continue
        entry = {"cluster_id": label_id, "area_px": int(area_px)}
        if pixel_area_m2:
            entry["area_m2"] = float(area_px * pixel_area_m2)
            entry["area_hectares"] = entry["area_m2"] / 10000.0
        clusters.append(entry)

    total_px = sum(c["area_px"] for c in clusters)
    result = {
        "num_clusters": len(clusters),
        "total_slum_area_px": total_px,
        "clusters": clusters,
    }
    if pixel_area_m2:
        result["total_slum_area_m2"] = total_px * pixel_area_m2
        result["total_slum_area_hectares"] = result["total_slum_area_m2"] / 10000.0
        if clusters:
            sizes = [c["area_m2"] for c in clusters]
            result["cluster_size_stats_m2"] = {
                "min": min(sizes), "max": max(sizes),
                "mean": sum(sizes) / len(sizes),
            }
    return result
