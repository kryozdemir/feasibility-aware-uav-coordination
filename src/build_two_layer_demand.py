import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, box

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

UTM_CRS = "EPSG:32637"
WGS84_CRS = "EPSG:4326"
GRID_CELL_SIZE_M = 500


REFERENCE_BBOX = dict(min_lat=37.55, max_lat=37.60, min_lon=36.90, max_lon=36.95)


MIN_BUILDINGS_FOR_POTENTIAL = 15
MAX_DESTROYED_FOR_POTENTIAL = 0


def parse_all_buildings():
    print("[1/6] OSM dosyasından node koordinatları okunuyor...")
    node_coords = {}
    for _, elem in ET.iterparse(RAW_DIR / "kahramanmaras_merkez.osm", events=("end",)):
        if elem.tag == "node":
            node_coords[elem.get("id")] = (
                float(elem.get("lon")),
                float(elem.get("lat")),
            )
            elem.clear()

    print(f"      {len(node_coords)} node okundu.")
    print("[2/6] Building way'leri parse ediliyor...")

    records = []
    skipped = 0
    for _, elem in ET.iterparse(RAW_DIR / "kahramanmaras_merkez.osm", events=("end",)):
        if elem.tag == "way":
            tags = {t.get("k"): t.get("v") for t in elem.findall("tag")}
            if "building" in tags:
                refs = [nd.get("ref") for nd in elem.findall("nd")]
                coords = [node_coords[r] for r in refs if r in node_coords]
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    try:
                        poly = Polygon(coords)
                        if poly.is_valid and poly.area > 0:
                            records.append({"osm_id": elem.get("id"), "geometry": poly})
                        else:
                            skipped += 1
                    except Exception:
                        skipped += 1
                else:
                    skipped += 1
            elem.clear()

    print(f"      {len(records)} geçerli bina poligonu, {skipped} atlandı (bozuk/açık poligon).")
    gdf = gpd.GeoDataFrame(records, crs=WGS84_CRS)
    return gdf


def main():
    all_buildings = parse_all_buildings()

    destroyed = gpd.read_file(RAW_DIR / "kahramanmaras_merkez_buildings.geojson")


    print("[2.5/6] Konumsal eşleştirme ile 'is_destroyed' belirleniyor (ID eşleştirmesi güvenilmez)...")
    joined_match = gpd.sjoin(
        all_buildings, destroyed[["geometry"]], how="left", predicate="intersects"
    )
    matched_idx = set(joined_match[joined_match["index_right"].notna()].index)
    all_buildings["is_destroyed"] = all_buildings.index.isin(matched_idx)
    print(f"       Konumsal eşleşme: {all_buildings['is_destroyed'].sum()} / {len(destroyed)} yıkılmış bina, mevcut OSM'de karşılığı bulundu.")

    all_buildings_utm = all_buildings.to_crs(UTM_CRS)
    all_buildings["area_m2"] = all_buildings_utm.geometry.area
    all_buildings["centroid_lon"] = all_buildings_utm.geometry.centroid.to_crs(WGS84_CRS).x
    all_buildings["centroid_lat"] = all_buildings_utm.geometry.centroid.to_crs(WGS84_CRS).y

    out_path = PROC_DIR / "all_buildings.geojson"
    all_buildings.drop(columns=["centroid_lon", "centroid_lat"]).to_file(out_path, driver="GeoJSON")
    print(f"[3/6] Kaydedildi: {out_path} ({len(all_buildings)} bina, {all_buildings['is_destroyed'].sum()} tanesi 'destroyed')")


    ref = REFERENCE_BBOX
    destroyed_utm = destroyed.to_crs(UTM_CRS)
    destroyed_centroids_wgs = destroyed_utm.geometry.centroid.to_crs(WGS84_CRS)
    destroyed_in_ref = (
        (destroyed_centroids_wgs.y >= ref["min_lat"]) & (destroyed_centroids_wgs.y <= ref["max_lat"]) &
        (destroyed_centroids_wgs.x >= ref["min_lon"]) & (destroyed_centroids_wgs.x <= ref["max_lon"])
    ).sum()

    ref_mask = (
        (all_buildings["centroid_lat"] >= ref["min_lat"]) & (all_buildings["centroid_lat"] <= ref["max_lat"]) &
        (all_buildings["centroid_lon"] >= ref["min_lon"]) & (all_buildings["centroid_lon"] <= ref["max_lon"])
    )
    ref_total = ref_mask.sum()
    destruction_rate = destroyed_in_ref / ref_total if ref_total > 0 else 0.0

    print(f"[4/6] Referans bölge (iyi haritalanmış şehir merkezi):")
    print(f"      Toplam bina (güncel OSM, 2026): {ref_total}")
    print(f"      Yıkılmış (destroyed dosyasından, 2023 anlık görüntü): {destroyed_in_ref}")
    print(f"      Oran: {destruction_rate:.4f} ({destruction_rate*100:.2f}%)")
    print(f"      [UYARI: Bu oran muhtemelen gerçek yıkım oranını olduğundan düşük gösteriyor --")
    print(f"       hem HOTOSM crowdsourced verisi eksik, hem de payda (2026) yeniden inşa sonrası bina sayısını yansıtıyor.]")


    minx, miny, maxx, maxy = all_buildings_utm.total_bounds
    n_cols = math.ceil((maxx - minx) / GRID_CELL_SIZE_M)
    n_rows = math.ceil((maxy - miny) / GRID_CELL_SIZE_M)

    cells = []
    for i in range(n_cols):
        for j in range(n_rows):
            x0 = minx + i * GRID_CELL_SIZE_M
            y0 = miny + j * GRID_CELL_SIZE_M
            cells.append({"col": i, "row": j, "geometry": box(x0, y0, x0 + GRID_CELL_SIZE_M, y0 + GRID_CELL_SIZE_M)})
    grid = gpd.GeoDataFrame(cells, crs=UTM_CRS)

    centroids_utm = gpd.GeoDataFrame(
        all_buildings[["osm_id", "area_m2", "is_destroyed"]],
        geometry=all_buildings_utm.geometry.centroid,
        crs=UTM_CRS,
    )
    joined = gpd.sjoin(centroids_utm, grid, how="left", predicate="within")
    agg = joined.groupby("index_right").agg(
        n_total=("osm_id", "count"),
        n_destroyed=("is_destroyed", "sum"),
    )
    grid = grid.join(agg)
    grid["n_total"] = grid["n_total"].fillna(0).astype(int)
    grid["n_destroyed"] = grid["n_destroyed"].fillna(0).astype(int)


    GAP_DISTANCE_THRESHOLD_M = 2000

    confirmed_utm_points = destroyed.to_crs(UTM_CRS).geometry.centroid
    grid_centroids = grid.geometry.centroid

    print("      Her hücre için en yakın confirmed noktaya mesafe hesaplanıyor...")

    grid_centroid_gdf = gpd.GeoDataFrame(geometry=grid_centroids, crs=UTM_CRS)
    confirmed_points_gdf = gpd.GeoDataFrame(geometry=confirmed_utm_points, crs=UTM_CRS)
    nearest = gpd.sjoin_nearest(grid_centroid_gdf, confirmed_points_gdf, distance_col="dist_to_confirmed_m")
    nearest = nearest[~nearest.index.duplicated(keep="first")]
    grid["dist_to_confirmed_m"] = nearest["dist_to_confirmed_m"]

    grid["is_potential_gap"] = (grid["n_total"] >= MIN_BUILDINGS_FOR_POTENTIAL) & (grid["dist_to_confirmed_m"] >= GAP_DISTANCE_THRESHOLD_M)

    n_gap_cells = grid["is_potential_gap"].sum()
    print(f"[5/6] {n_gap_cells} hücre 'veri-eksik ama yoğun yerleşik' olarak işaretlendi (>= {MIN_BUILDINGS_FOR_POTENTIAL} bina, {MAX_DESTROYED_FOR_POTENTIAL} confirmed yıkım).")

    grid_nonzero = grid[grid["n_total"] > 0].to_crs(WGS84_CRS)
    grid_nonzero.to_file(PROC_DIR / "coverage_grid.geojson", driver="GeoJSON")


    confirmed_demand = destroyed.copy()
    confirmed_demand["source"] = "confirmed_destroyed"
    confirmed_demand["confidence"] = 1.0
    confirmed_demand_utm = confirmed_demand.to_crs(UTM_CRS)
    confirmed_demand["weight"] = confirmed_demand_utm.geometry.area
    confirmed_demand["geometry"] = confirmed_demand_utm.geometry.centroid.to_crs(WGS84_CRS)
    confirmed_demand = confirmed_demand[["osm_id", "weight", "source", "confidence", "geometry"]]

    gap_cells = grid[grid["is_potential_gap"]].to_crs(WGS84_CRS).copy()
    gap_cells["osm_id"] = "grid_" + gap_cells["col"].astype(str) + "_" + gap_cells["row"].astype(str)


    avg_building_area = all_buildings.loc[all_buildings["is_destroyed"], "area_m2"].mean()
    gap_cells["weight"] = gap_cells["n_total"] * destruction_rate * avg_building_area
    gap_cells["source"] = "potential_unmapped"


    gap_cells["confidence"] = destruction_rate
    gap_cells_utm = gap_cells.to_crs(UTM_CRS)
    gap_cells["geometry"] = gap_cells_utm.geometry.centroid
    gap_cells = gap_cells.set_geometry("geometry", crs=UTM_CRS).to_crs(WGS84_CRS)
    gap_cells = gap_cells[["osm_id", "weight", "source", "confidence", "geometry"]]

    combined = pd.concat([confirmed_demand, gap_cells], ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs=WGS84_CRS)
    combined["weight_normalized"] = combined["weight"] / combined["weight"].sum()

    out_path2 = PROC_DIR / "demand_points_v2.geojson"
    combined.to_file(out_path2, driver="GeoJSON")

    print(f"[6/6] Kaydedildi: {out_path2}")
    print(f"      Confirmed talep noktası: {(combined['source']=='confirmed_destroyed').sum()}")
    print(f"      Potential (tahmini) talep noktası: {(combined['source']=='potential_unmapped').sum()}")
    print(f"      Toplam ağırlık payı -- confirmed: {combined.loc[combined['source']=='confirmed_destroyed','weight_normalized'].sum():.2%}, "
          f"potential: {combined.loc[combined['source']=='potential_unmapped','weight_normalized'].sum():.2%}")


if __name__ == "__main__":
    main()
