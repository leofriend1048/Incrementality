"""DMA boundary loading and polygon-based adjacency computation.

Computes geographic adjacency between all 210 Nielsen DMAs using actual
polygon boundaries from GeoJSON/TopoJSON files. This is critical for
accurate spillover risk assessment.

Two approaches:
1. Polygon-based (PRIMARY): Uses GeoPandas + Shapely to compute adjacency
   from DMA boundary polygons via geometry.touches(). Requires a GeoJSON file.

2. Hardcoded fallback: Uses the static adjacency map in spillover.py for
   the ~40 most common DMAs when GeoPandas/GeoJSON is unavailable.

Usage:
    # Download boundaries + compute adjacency
    incrementality boundaries download --output ./data/dma_boundaries.geojson
    incrementality boundaries compute --input ./data/dma_boundaries.geojson

    # Programmatic use
    from incrementality.design.dma_boundaries import load_adjacency, compute_adjacency_from_geojson
    adjacency = compute_adjacency_from_geojson("./data/dma_boundaries.geojson")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cache directory for computed adjacency
_DEFAULT_CACHE_DIR = Path("./data")
_ADJACENCY_CACHE_FILE = "dma_adjacency.json"


def compute_adjacency_from_geojson(
    geojson_path: str | Path,
    dma_code_field: str | None = None,
    buffer_miles: float = 1.0,
    cache_dir: str | Path | None = None,
) -> dict[str, list[str]]:
    """Compute DMA adjacency from a GeoJSON/TopoJSON boundary file.

    Uses GeoPandas to load the boundary polygons and Shapely to test
    whether polygons touch (share a border). A small buffer is applied
    to handle tiny gaps between polygon boundaries.

    Args:
        geojson_path: Path to GeoJSON or TopoJSON file with DMA boundaries.
        dma_code_field: Column name containing the DMA code. If None,
            auto-detects from common field names.
        buffer_miles: Buffer distance in miles for adjacency detection.
            Small buffer handles gaps between polygons. Default 1 mile.
        cache_dir: Directory to cache computed adjacency. If None, uses ./data/

    Returns:
        Dict mapping each DMA code to a list of adjacent DMA codes.
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "GeoPandas is required for polygon-based DMA adjacency. "
            "Install with: pip install 'incrementality[geo]'"
        )

    geojson_path = Path(geojson_path)
    if not geojson_path.exists():
        raise FileNotFoundError(
            f"DMA boundary file not found: {geojson_path}. "
            f"Download a DMA GeoJSON file first."
        )

    logger.info(f"Loading DMA boundaries from {geojson_path}")
    gdf = gpd.read_file(geojson_path)

    # Auto-detect DMA code column
    if dma_code_field is None:
        dma_code_field = _detect_dma_code_field(gdf)
    if dma_code_field not in gdf.columns:
        raise ValueError(
            f"DMA code field '{dma_code_field}' not found in GeoJSON. "
            f"Available fields: {list(gdf.columns)}"
        )

    # Normalize DMA codes to strings
    gdf["_dma_code"] = gdf[dma_code_field].astype(str).str.strip()

    # Reproject to a projected CRS for accurate distance calculations
    # Use Albers Equal Area (EPSG:5070) for contiguous US
    if gdf.crs and not gdf.crs.is_projected:
        gdf = gdf.to_crs(epsg=5070)
    elif gdf.crs is None:
        # Assume WGS84 and reproject
        gdf = gdf.set_crs(epsg=4326).to_crs(epsg=5070)

    # Convert buffer from miles to meters (1 mile ≈ 1609.34 meters)
    buffer_meters = buffer_miles * 1609.34

    logger.info(
        f"Computing adjacency for {len(gdf)} DMAs "
        f"(buffer={buffer_miles} mi / {buffer_meters:.0f} m)"
    )

    # Build spatial index for efficient queries
    gdf_buffered = gdf.copy()
    gdf_buffered["geometry"] = gdf.geometry.buffer(buffer_meters)
    sindex = gdf_buffered.sindex

    adjacency: dict[str, list[str]] = {}
    for idx, row in gdf_buffered.iterrows():
        dma_code = row["_dma_code"]
        # Query spatial index for candidates
        candidate_indices = list(sindex.intersection(row.geometry.bounds))
        neighbors = []
        for cidx in candidate_indices:
            if cidx == idx:
                continue
            candidate = gdf_buffered.iloc[cidx]
            if row.geometry.intersects(candidate.geometry):
                neighbors.append(candidate["_dma_code"])
        adjacency[dma_code] = sorted(set(neighbors))

    n_edges = sum(len(v) for v in adjacency.values()) // 2
    logger.info(
        f"Computed adjacency: {len(adjacency)} DMAs, {n_edges} border pairs"
    )

    # Cache the result
    cache_path = _get_cache_path(cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(adjacency, f, indent=2)
    logger.info(f"Adjacency cached to {cache_path}")

    return adjacency


def load_adjacency(cache_dir: str | Path | None = None) -> dict[str, list[str]] | None:
    """Load cached DMA adjacency from disk.

    Returns None if no cached adjacency exists.
    """
    cache_path = _get_cache_path(cache_dir)
    if not cache_path.exists():
        return None
    with open(cache_path) as f:
        adjacency = json.load(f)
    logger.info(f"Loaded cached DMA adjacency: {len(adjacency)} DMAs")
    return adjacency


def get_adjacency(
    geojson_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, list[str]] | None:
    """Get DMA adjacency, computing from GeoJSON if needed.

    Tries in order:
    1. Cached adjacency from disk
    2. Compute from GeoJSON if path provided
    3. Return None (caller should use hardcoded fallback)
    """
    # Try cache first
    cached = load_adjacency(cache_dir)
    if cached is not None:
        return cached

    # Try computing from GeoJSON
    if geojson_path is not None:
        try:
            return compute_adjacency_from_geojson(geojson_path, cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Failed to compute adjacency from GeoJSON: {e}")

    return None


def _detect_dma_code_field(gdf: Any) -> str:
    """Auto-detect the DMA code field name from common conventions."""
    candidates = [
        "dma", "DMA", "dma_code", "DMA_CODE", "dma_id", "DMA_ID",
        "nielsen_dma", "NIELSEN_DMA", "dma1", "DMA1",
        "id", "ID", "code", "CODE", "GEOID",
        "rank", "RANK", "dma_rank", "DMA_RANK",
    ]
    for candidate in candidates:
        if candidate in gdf.columns:
            return candidate

    # Try to find any column with plausible DMA codes (3-digit numbers)
    for col in gdf.columns:
        if col == "geometry":
            continue
        try:
            sample = gdf[col].dropna().head(10).astype(str)
            if all(s.strip().isdigit() and 100 <= int(s.strip()) <= 999 for s in sample):
                logger.info(f"Auto-detected DMA code field: '{col}'")
                return col
        except (ValueError, TypeError):
            continue

    raise ValueError(
        f"Could not auto-detect DMA code field. "
        f"Available columns: {list(gdf.columns)}. "
        f"Use dma_code_field parameter to specify explicitly."
    )


def _get_cache_path(cache_dir: str | Path | None) -> Path:
    """Get the path for the adjacency cache file."""
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    return Path(cache_dir) / _ADJACENCY_CACHE_FILE


# ---------------------------------------------------------------------------
# Complete DMA listing with names and approximate centroids
# Used for centroid-distance fallback when polygon data unavailable
# ---------------------------------------------------------------------------

# All 210 Nielsen DMAs with approximate lat/lon centroids
# Source: FCC DMA definitions, Census Bureau geographic centers
_ALL_DMA_CENTROIDS: dict[str, tuple[str, float, float]] = {
    "500": ("Portland-Auburn, ME", 43.66, -70.26),
    "501": ("New York, NY", 40.71, -74.01),
    "502": ("Binghamton, NY", 42.10, -75.91),
    "503": ("Macon, GA", 32.84, -83.63),
    "504": ("Philadelphia, PA", 39.95, -75.17),
    "505": ("Detroit, MI", 42.33, -83.05),
    "506": ("Boston, MA", 42.36, -71.06),
    "507": ("Savannah, GA", 32.08, -81.09),
    "508": ("Pittsburgh, PA", 40.44, -80.00),
    "509": ("Ft. Wayne, IN", 41.08, -85.14),
    "510": ("Cleveland, OH", 41.50, -81.69),
    "511": ("Washington, DC", 38.91, -77.04),
    "512": ("Baltimore, MD", 39.29, -76.61),
    "513": ("Flint-Saginaw, MI", 43.42, -83.95),
    "514": ("Buffalo, NY", 42.89, -78.88),
    "515": ("Cincinnati, OH", 39.10, -84.51),
    "516": ("Erie, PA", 42.13, -80.09),
    "517": ("Charlotte, NC", 35.23, -80.84),
    "518": ("Greensboro, NC", 36.07, -79.79),
    "519": ("Charleston, SC", 32.78, -79.93),
    "520": ("Augusta, GA", 33.47, -81.97),
    "521": ("Providence, RI", 41.82, -71.41),
    "522": ("Columbus, GA", 32.46, -84.99),
    "523": ("Burlington, VT", 44.48, -73.21),
    "524": ("Atlanta, GA", 33.75, -84.39),
    "525": ("Albany, GA", 31.58, -84.16),
    "526": ("Utica, NY", 43.10, -75.23),
    "527": ("Indianapolis, IN", 39.77, -86.16),
    "528": ("Miami-Ft. Lauderdale, FL", 25.76, -80.19),
    "529": ("Louisville, KY", 38.25, -85.76),
    "530": ("Tallahassee, FL", 30.44, -84.28),
    "531": ("Tri-Cities, TN-VA", 36.55, -82.19),
    "532": ("Albany-Schenectady, NY", 42.65, -73.76),
    "533": ("Hartford, CT", 41.76, -72.68),
    "534": ("Orlando, FL", 28.54, -81.38),
    "535": ("Columbus, OH", 39.96, -83.00),
    "536": ("Youngstown, OH", 41.10, -80.65),
    "537": ("Bangor, ME", 44.80, -68.77),
    "538": ("Rochester, NY", 43.16, -77.61),
    "539": ("Tampa-St. Petersburg, FL", 27.95, -82.46),
    "540": ("Traverse City, MI", 44.76, -85.62),
    "541": ("Lexington, KY", 38.05, -84.50),
    "542": ("Dayton, OH", 39.76, -84.19),
    "543": ("Springfield-Holyoke, MA", 42.10, -72.59),
    "544": ("Norfolk, VA", 36.85, -76.29),
    "545": ("Greenville-Spartanburg, SC", 34.85, -82.40),
    "546": ("Columbia, SC", 34.00, -81.03),
    "547": ("Toledo, OH", 41.65, -83.54),
    "548": ("West Palm Beach, FL", 26.72, -80.05),
    "549": ("Watertown, NY", 43.97, -75.91),
    "550": ("Wilmington, NC", 34.23, -77.94),
    "551": ("Lansing, MI", 42.73, -84.56),
    "552": ("Presque Isle, ME", 46.68, -68.02),
    "553": ("Marquette, MI", 46.54, -87.40),
    "554": ("Wheeling, WV", 40.06, -80.72),
    "555": ("Syracuse, NY", 43.05, -76.15),
    "556": ("Richmond, VA", 37.54, -77.44),
    "557": ("Knoxville, TN", 35.96, -83.92),
    "558": ("Lima, OH", 40.74, -84.11),
    "559": ("Bluefield-Beckley, WV", 37.27, -81.22),
    "560": ("Raleigh-Durham, NC", 35.78, -78.64),
    "561": ("Jacksonville, FL", 30.33, -81.66),
    "563": ("Grand Rapids, MI", 42.96, -85.66),
    "564": ("Charleston-Huntington, WV", 38.35, -81.63),
    "565": ("Elmira, NY", 42.09, -76.81),
    "566": ("Harrisburg, PA", 40.27, -76.88),
    "567": ("Greenville-New Bern, NC", 35.60, -77.37),
    "569": ("Harrisonburg, VA", 38.45, -78.87),
    "570": ("Florence-Myrtle Beach, SC", 34.20, -79.76),
    "571": ("Ft. Myers-Naples, FL", 26.64, -81.87),
    "573": ("Roanoke-Lynchburg, VA", 37.27, -79.94),
    "574": ("Johnstown-Altoona, PA", 40.33, -78.92),
    "575": ("Chattanooga, TN", 35.05, -85.31),
    "576": ("Salisbury, MD", 38.37, -75.60),
    "577": ("Wilkes-Barre, PA", 41.25, -75.88),
    "581": ("Terre Haute, IN", 39.47, -87.41),
    "582": ("Lafayette, IN", 40.42, -86.87),
    "583": ("Alpena, MI", 45.06, -83.43),
    "584": ("Charlottesville, VA", 38.03, -78.48),
    "588": ("South Bend, IN", 41.68, -86.25),
    "592": ("Gainesville, FL", 29.65, -82.32),
    "596": ("Zanesville, OH", 39.94, -82.01),
    "597": ("Parkersburg, WV", 39.27, -81.56),
    "598": ("Clarksburg-Weston, WV", 39.28, -80.34),
    "600": ("Corpus Christi, TX", 27.80, -97.40),
    "602": ("Chicago, IL", 41.88, -87.63),
    "603": ("Joplin-Pittsburg, MO", 37.08, -94.51),
    "604": ("Columbia-Jefferson City, MO", 38.95, -92.33),
    "605": ("Topeka, KS", 39.05, -95.68),
    "606": ("Dothan, AL", 31.22, -85.39),
    "609": ("St. Louis, MO", 38.63, -90.20),
    "610": ("Rockford, IL", 42.27, -89.09),
    "611": ("Rochester-Mason City, MN", 44.02, -92.47),
    "612": ("Shreveport, LA", 32.53, -93.75),
    "613": ("Minneapolis-St. Paul, MN", 44.98, -93.27),
    "616": ("Kansas City, MO", 39.10, -94.58),
    "617": ("Milwaukee, WI", 43.04, -87.91),
    "618": ("Houston, TX", 29.76, -95.37),
    "619": ("Springfield, MO", 37.22, -93.29),
    "622": ("New Orleans, LA", 29.95, -90.07),
    "623": ("Dallas-Ft. Worth, TX", 32.78, -96.80),
    "624": ("Sioux City, IA", 42.50, -96.40),
    "625": ("Waco-Temple-Bryan, TX", 31.55, -97.15),
    "626": ("Victoria, TX", 28.81, -96.99),
    "627": ("Wichita Falls-Lawton, TX", 33.91, -98.49),
    "628": ("Monroe-El Dorado, LA", 32.51, -92.12),
    "630": ("Birmingham, AL", 33.52, -86.81),
    "631": ("Ottumwa-Kirksville, IA", 41.02, -92.41),
    "632": ("Paducah-Cape Girardeau, KY", 37.08, -88.60),
    "633": ("Odessa-Midland, TX", 31.99, -102.08),
    "634": ("Amarillo, TX", 35.22, -101.83),
    "635": ("Austin, TX", 30.27, -97.74),
    "636": ("Harlingen-Weslaco, TX", 26.19, -97.70),
    "637": ("Cedar Rapids, IA", 42.03, -91.64),
    "638": ("St. Joseph, MO", 39.77, -94.85),
    "639": ("Jackson, TN", 35.61, -88.81),
    "640": ("Memphis, TN", 35.15, -90.05),
    "641": ("San Antonio, TX", 29.42, -98.49),
    "642": ("Lafayette, LA", 30.22, -92.02),
    "643": ("Lake Charles, LA", 30.23, -93.22),
    "644": ("Alexandria, LA", 31.31, -92.45),
    "647": ("Greenwood-Greenville, MS", 33.52, -90.18),
    "648": ("Champaign-Springfield, IL", 40.12, -88.24),
    "649": ("Evansville, IN", 37.97, -87.56),
    "650": ("Oklahoma City, OK", 35.47, -97.52),
    "651": ("Lubbock, TX", 33.57, -101.85),
    "652": ("Omaha, NE", 41.26, -95.94),
    "656": ("Panama City, FL", 30.16, -85.66),
    "657": ("Sherman-Ada, TX", 33.64, -96.61),
    "658": ("Green Bay-Appleton, WI", 44.51, -88.02),
    "659": ("Nashville, TN", 36.17, -86.78),
    "661": ("San Angelo, TX", 31.46, -100.44),
    "662": ("Abilene-Sweetwater, TX", 32.45, -99.73),
    "669": ("Madison, WI", 43.07, -89.40),
    "670": ("Ft. Smith-Fayetteville, AR", 35.39, -94.40),
    "671": ("Tulsa, OK", 36.15, -95.99),
    "673": ("Columbus-Tupelo, MS", 33.50, -88.43),
    "675": ("Peoria-Bloomington, IL", 40.69, -89.59),
    "676": ("Duluth-Superior, MN", 46.79, -92.10),
    "678": ("Wichita-Hutchinson, KS", 37.69, -97.34),
    "679": ("Des Moines, IA", 41.59, -93.62),
    "682": ("Davenport-Rock Island, IA", 41.52, -90.58),
    "686": ("Mobile-Pensacola, FL", 30.69, -88.04),
    "687": ("Minot-Bismarck, ND", 48.23, -101.29),
    "691": ("Huntsville-Decatur, AL", 34.73, -86.59),
    "692": ("Beaumont-Port Arthur, TX", 30.08, -94.10),
    "693": ("Little Rock-Pine Bluff, AR", 34.75, -92.29),
    "698": ("Montgomery-Selma, AL", 32.38, -86.31),
    "702": ("La Crosse-Eau Claire, WI", 43.81, -91.25),
    "705": ("Wausau-Rhinelander, WI", 44.96, -89.63),
    "709": ("Tyler-Longview, TX", 32.35, -95.30),
    "710": ("Hattiesburg-Laurel, MS", 31.33, -89.29),
    "711": ("Meridian, MS", 32.35, -88.70),
    "716": ("Baton Rouge, LA", 30.45, -91.19),
    "717": ("Quincy-Hannibal, IL", 39.94, -91.41),
    "718": ("Jackson, MS", 32.30, -90.18),
    "722": ("Lincoln-Hastings, NE", 40.81, -96.70),
    "724": ("Fargo-Valley City, ND", 46.88, -96.79),
    "725": ("Sioux Falls-Mitchell, SD", 43.55, -96.73),
    "734": ("Jonesboro, AR", 35.84, -90.70),
    "736": ("Bowling Green, KY", 36.99, -86.44),
    "737": ("Knoxville, TN", 35.96, -83.92),
    "740": ("North Platte, NE", 41.12, -100.77),
    "743": ("Anchorage, AK", 61.22, -149.90),
    "744": ("Honolulu, HI", 21.31, -157.86),
    "745": ("Fairbanks, AK", 64.84, -147.72),
    "746": ("Biloxi-Gulfport, MS", 30.40, -89.07),
    "747": ("Juneau, AK", 58.30, -134.42),
    "749": ("Laredo, TX", 27.51, -99.51),
    "751": ("Denver, CO", 39.74, -104.99),
    "752": ("Colorado Springs, CO", 38.83, -104.82),
    "753": ("Phoenix, AZ", 33.45, -112.07),
    "754": ("Butte-Bozeman, MT", 45.78, -111.03),
    "755": ("Great Falls, MT", 47.51, -111.29),
    "756": ("Billings, MT", 45.78, -108.50),
    "757": ("Boise, ID", 43.62, -116.21),
    "758": ("Idaho Falls-Pocatello, ID", 43.47, -112.04),
    "759": ("Cheyenne-Scottsbluff, WY", 41.14, -104.82),
    "760": ("Twin Falls, ID", 42.56, -114.46),
    "762": ("Missoula, MT", 46.87, -114.00),
    "764": ("Rapid City, SD", 44.08, -103.23),
    "765": ("El Paso, TX", 31.76, -106.49),
    "766": ("Helena, MT", 46.59, -112.04),
    "770": ("Salt Lake City, UT", 40.76, -111.89),
    "771": ("Yuma-El Centro, AZ", 32.69, -114.62),
    "773": ("Grand Junction, CO", 39.07, -108.55),
    "789": ("Tucson, AZ", 32.22, -110.97),
    "790": ("Albuquerque-Santa Fe, NM", 35.08, -106.65),
    "798": ("Glendive, MT", 47.11, -104.71),
    "800": ("Bakersfield, CA", 35.37, -119.02),
    "801": ("Eugene, OR", 44.05, -123.09),
    "802": ("Eureka, CA", 40.80, -124.16),
    "803": ("Los Angeles, CA", 34.05, -118.24),
    "804": ("Palm Springs, CA", 33.83, -116.55),
    "807": ("San Francisco-Oakland, CA", 37.77, -122.42),
    "810": ("Yakima-Pasco, WA", 46.60, -120.51),
    "811": ("Reno, NV", 39.53, -119.81),
    "813": ("Medford-Klamath Falls, OR", 42.33, -122.87),
    "819": ("Seattle-Tacoma, WA", 47.61, -122.33),
    "820": ("Portland, OR", 45.52, -122.68),
    "821": ("Bend, OR", 44.06, -121.31),
    "825": ("San Diego, CA", 32.72, -117.16),
    "828": ("Monterey-Salinas, CA", 36.60, -121.89),
    "839": ("Las Vegas, NV", 36.17, -115.14),
    "855": ("Santa Barbara, CA", 34.42, -119.70),
    "862": ("Sacramento-Stockton, CA", 38.58, -121.49),
    "866": ("Fresno-Visalia, CA", 36.74, -119.77),
    "868": ("Chico-Redding, CA", 39.73, -121.84),
    "881": ("Spokane, WA", 47.66, -117.43),
}


def compute_adjacency_from_centroids(
    max_distance_miles: float = 175.0,
) -> dict[str, list[str]]:
    """Compute DMA adjacency based on centroid distances.

    This is a reasonable fallback when polygon boundaries are unavailable.
    Uses great-circle distance between DMA centroids.

    Args:
        max_distance_miles: Maximum distance between centroids to consider
            DMAs adjacent. Default 175 miles captures most true adjacencies
            while avoiding false positives.
    """
    from math import radians, cos, sin, asin, sqrt

    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in miles."""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 2 * 3959 * asin(sqrt(a))  # 3959 = Earth radius in miles

    dma_codes = list(_ALL_DMA_CENTROIDS.keys())
    adjacency: dict[str, list[str]] = {}

    for i, code_i in enumerate(dma_codes):
        _, lat_i, lon_i = _ALL_DMA_CENTROIDS[code_i]
        neighbors = []
        for j, code_j in enumerate(dma_codes):
            if i == j:
                continue
            _, lat_j, lon_j = _ALL_DMA_CENTROIDS[code_j]
            dist = haversine(lat_i, lon_i, lat_j, lon_j)
            if dist <= max_distance_miles:
                neighbors.append(code_j)
        adjacency[code_i] = sorted(neighbors)

    return adjacency
