import json
import math
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point, box

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


UTM_CRS = "EPSG:32637"
WGS84_CRS = "EPSG:4326"

GRID_CELL_SIZE_M = 500


def load_destroyed_buildings() -> gpd.GeoDataFrame:
    path = RAW_DIR / "kahramanmaras_merkez_buildings.geojson"
    gdf = gpd.read_file(path)
    gdf = gdf.set_crs(WGS84_CRS, allow_override=True)
    print(f"[1/4] {len(gdf)} yıkılmış bina yüklendi.")
    return gdf


def compute_demand_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:


    gdf_utm = gdf.to_crs(UTM_CRS)
    centroids_wgs84 = gdf_utm.geometry.centroid.to_crs(WGS84_CRS)

    demand = gpd.GeoDataFrame(
        {
            "osm_id": gdf["osm_id"],
            "area_m2": gdf_utm.geometry.area,
        },
        geometry=centroids_wgs84,
        crs=WGS84_CRS,
    )


    total_area = demand["area_m2"].sum()
    demand["weight"] = demand["area_m2"] / total_area

    print(f"[2/4] {len(demand)} talep noktası oluşturuldu.")
    print(f"      Toplam yıkılmış bina alanı: {total_area:,.0f} m^2")
    print(f"      Ortalama bina alanı: {demand['area_m2'].mean():.1f} m^2")
    return demand


def compute_risk_grid(demand: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    demand_utm = demand.to_crs(UTM_CRS)

    minx, miny, maxx, maxy = demand_utm.total_bounds

    minx -= GRID_CELL_SIZE_M
    miny -= GRID_CELL_SIZE_M
    maxx += GRID_CELL_SIZE_M
    maxy += GRID_CELL_SIZE_M

    n_cols = math.ceil((maxx - minx) / GRID_CELL_SIZE_M)
    n_rows = math.ceil((maxy - miny) / GRID_CELL_SIZE_M)

    cells = []
    for i in range(n_cols):
        for j in range(n_rows):
            x0 = minx + i * GRID_CELL_SIZE_M
            y0 = miny + j * GRID_CELL_SIZE_M
            cells.append(
                {
                    "col": i,
                    "row": j,
                    "geometry": box(x0, y0, x0 + GRID_CELL_SIZE_M, y0 + GRID_CELL_SIZE_M),
                }
            )
    grid = gpd.GeoDataFrame(cells, crs=UTM_CRS)


    joined = gpd.sjoin(demand_utm, grid, how="left", predicate="within")
    counts = joined.groupby("index_right").agg(
        n_buildings=("osm_id", "count"), total_area_m2=("area_m2", "sum")
    )
    grid = grid.join(counts)
    grid["n_buildings"] = grid["n_buildings"].fillna(0).astype(int)
    grid["total_area_m2"] = grid["total_area_m2"].fillna(0.0)


    max_count = grid["n_buildings"].max()
    grid["risk_score"] = grid["n_buildings"] / max_count if max_count > 0 else 0.0


    grid_nonzero = grid[grid["n_buildings"] > 0].to_crs(WGS84_CRS)

    print(f"[3/4] Grid oluşturuldu: {n_cols}x{n_rows} hücre, {len(grid_nonzero)} tanesi dolu.")
    print(f"      En yoğun hücre: {max_count} bina")
    return grid_nonzero


def main():
    gdf = load_destroyed_buildings()
    demand = compute_demand_points(gdf)
    risk_grid = compute_risk_grid(demand)

    demand_out = PROC_DIR / "demand_points.geojson"
    grid_out = PROC_DIR / "risk_grid.geojson"

    demand.to_file(demand_out, driver="GeoJSON")
    risk_grid.to_file(grid_out, driver="GeoJSON")

    print(f"[4/4] Kaydedildi:")
    print(f"      {demand_out}")
    print(f"      {grid_out}")


if __name__ == "__main__":
    main()
