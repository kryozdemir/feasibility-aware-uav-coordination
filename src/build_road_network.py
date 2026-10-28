import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"

UTM_CRS = "EPSG:32637"
WGS84_CRS = "EPSG:4326"


DEPOT_LAT = 37.5753
DEPOT_LON = 36.9228


def build_road_graph() -> nx.MultiDiGraph:
    print("[1/5] OSM XML'den yol ağı grafı çıkarılıyor...")
    G = ox.graph_from_xml(RAW_DIR / "kahramanmaras_merkez.osm", simplify=True)
    print(f"      Ham graf: {G.number_of_nodes()} node, {G.number_of_edges()} edge")
    return G


def clean_and_connect(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    print("[2/5] Bağlı bileşenler kontrol ediliyor...")


    components = list(nx.weakly_connected_components(G))
    components.sort(key=len, reverse=True)
    print(f"      {len(components)} ayrı bağlı bileşen bulundu.")
    print(f"      Boyutları (ilk 5): {[len(c) for c in components[:5]]}")

    largest = components[0]
    coverage = len(largest) / G.number_of_nodes()
    print(f"      En büyük bileşen: {len(largest)} node (toplamın %{coverage*100:.1f}'i)")

    if coverage < 0.95:
        print(f"      [UYARI] En büyük bileşen node'ların sadece %{coverage*100:.1f}'ini kapsıyor.")
        print(f"      Küçük bileşenlerdeki noktalar depoya karadan erişilemez olacak.")

    G_clean = G.subgraph(largest).copy()
    print(f"      Temizlenmiş graf: {G_clean.number_of_nodes()} node, {G_clean.number_of_edges()} edge")
    return G_clean


def project_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    print("[3/5] Graf metrik projeksiyona (UTM 37N) çevriliyor...")


    G_proj = ox.project_graph(G, to_crs=UTM_CRS)
    return G_proj


def attach_depot_and_compute_distances(G_proj: nx.MultiDiGraph, demand_path: Path):
    print("[4/5] Depo konumu grafa bağlanıyor...")
    import pyproj

    transformer = pyproj.Transformer.from_crs(WGS84_CRS, UTM_CRS, always_xy=True)
    depot_x, depot_y = transformer.transform(DEPOT_LON, DEPOT_LAT)

    depot_node = ox.distance.nearest_nodes(G_proj, depot_x, depot_y)
    depot_node_data = G_proj.nodes[depot_node]
    dist_to_depot_node = ((depot_node_data["x"] - depot_x) ** 2 + (depot_node_data["y"] - depot_y) ** 2) ** 0.5
    print(f"      Depo, en yakın yol düğümüne {dist_to_depot_node:.0f}m mesafede bağlandı (node id: {depot_node}).")

    print("[5/5] Depo -> her talep noktasına en kısa yol mesafesi hesaplanıyor...")
    demand = gpd.read_file(demand_path).to_crs(UTM_CRS)


    lengths = nx.single_source_dijkstra_path_length(G_proj.to_undirected(), depot_node, weight="length")

    distances = []
    unreachable = 0
    for geom in demand.geometry:
        nearest = ox.distance.nearest_nodes(G_proj, geom.x, geom.y)
        d = lengths.get(nearest)
        if d is None:
            unreachable += 1
            distances.append(None)
        else:

            node_data = G_proj.nodes[nearest]
            snap_dist = ((node_data["x"] - geom.x) ** 2 + (node_data["y"] - geom.y) ** 2) ** 0.5
            distances.append(d + snap_dist)

    demand["dist_from_depot_m"] = distances
    print(f"      {unreachable} / {len(demand)} talep noktası depoya karadan ULAŞILAMAZ (ayrı bağlı bileşende).")
    if len(demand) - unreachable > 0:
        reachable = demand["dist_from_depot_m"].dropna()
        print(f"      Ulaşılabilir noktalar için mesafe: min={reachable.min():.0f}m, "
              f"medyan={reachable.median():.0f}m, max={reachable.max():.0f}m")

    return demand, depot_node, (depot_x, depot_y)


def main():
    G = build_road_graph()
    G = clean_and_connect(G)
    G_proj = project_graph(G)

    demand_out, depot_node, depot_xy = attach_depot_and_compute_distances(
        G_proj, PROC_DIR / "demand_points_v2.geojson"
    )


    with open(PROC_DIR / "road_network.pickle", "wb") as f:
        pickle.dump(
            {
                "graph": G_proj,
                "depot_node": depot_node,
                "depot_xy_utm": depot_xy,
                "depot_latlon": (DEPOT_LAT, DEPOT_LON),
                "crs": UTM_CRS,
            },
            f,
        )
    print(f"\nKaydedildi: {PROC_DIR / 'road_network.pickle'}")

    demand_out.to_crs(WGS84_CRS).to_file(PROC_DIR / "demand_points_with_distance.geojson", driver="GeoJSON")
    print(f"Kaydedildi: {PROC_DIR / 'demand_points_with_distance.geojson'}")


if __name__ == "__main__":
    main()
