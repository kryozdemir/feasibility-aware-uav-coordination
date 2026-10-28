# Feasibility-Aware Multi-UAV Coordination under Selective Humanitarian Visibility in Disaster Response

This repository contains the source code, trained model checkpoints, and evaluation results for a hierarchical multi-UAV coordination system for post-disaster humanitarian logistics. The system combines a non-learned, optimization-based manager that is responsible for feasibility pruning (PPES), value-density prioritization, and explicit coverage of under-observed demand with MAPPO-trained worker policies that execute dispatch at the individual-UAV level.

The core problem addressed is **selective visibility**: in real disaster settings, confirmed damage/demand data (e.g., from crowdsourced humanitarian mapping such as HOTOSM) is spatially clustered, not randomly missing. Large swaths of potentially-affected but unmapped area ("potential demand") can be systematically overlooked by naive dispatch policies. This work studies how far different coordination mechanisms — reward shaping, explicit coverage constraints, feasibility pruning, and fair allocation — go toward correcting that bias, and under what resource conditions each mechanism actually helps.

## Repository structure

```
src/        Source code (environments, PPES manager, MAPPO networks, training/eval scripts)
data/       Demand-point and road-network datasets (see Dataset below)
models/     Trained actor checkpoints (.pt) across conditions, fleet sizes, and seeds
results/    Raw evaluation result histories (.pkl) used to produce the paper's figures/tables
```

## Dataset

The experiments are built around a disaster-response scenario centered on a depot near **36.92°N, 37.58°E** (Kahramanmaraş region, Türkiye), using road-network and demand data projected to **UTM zone 37N (EPSG:32637)**.

| File | Contents | Records |
|---|---|---|
| `data/demand_points_v2.geojson` | Demand points with per-point humanitarian value weight, source label, and detection confidence | 778 points — 286 `confirmed_destroyed`, 492 `potential_unmapped` |
| `data/demand_points_random_mask.geojson` | Same 778 demand points, with visibility masking randomized (quantity-matched control condition used in the visibility-structure experiment) | 778 points |
| `data/road_network.pickle` | Road network graph (`networkx.MultiDiGraph`) used for UAV routing/feasibility, plus depot location | 24,786 nodes / 66,127 edges |

Each demand point carries:
- `osm_id` — source OpenStreetMap identifier
- `weight` — raw humanitarian-value weight
- `weight_normalized` — weight normalized across the dataset
- `source` — `confirmed_destroyed` (from verified/crowdsourced damage mapping) or `potential_unmapped` (in a damage-plausible area lacking confirmed mapping — an observational label, not a probability of damage)
- `confidence` — detection/verification confidence score

The road network pickle stores a dict with keys `graph` (the road graph), `depot_node`, `depot_xy_utm`, `depot_latlon`, and `crs` (`EPSG:32637`).

## Models

`models/` contains MAPPO actor checkpoints (`*_actor_final.pt`) organized by condition code, fleet size, and seed, e.g. `K4_SHORT10_n7_seed0_actor_final.pt`. Condition codes correspond to the ablation chain described below (Nearest, ValuePrio, ValuePrio+Cov, ValuePrio+Cov+Safe, Full) across the three fleet-time regimes (10-min feasibility floor, 15-min critical-scarcity, 30-min resource-sufficiency) and fleet sizes N=3–20.

## Results

Evaluation used a 5-condition ablation chain (**Nearest → ValuePrio → ValuePrio+Cov → ValuePrio+Cov+Safe (PPES) → Full**), evaluated across three fleet-time regimes and multiple fleet sizes, with 5-seed averaging (3-seed for some N=7 intermediate conditions).

### Selective visibility has a structural, not just quantitative, effect

Under real-clustered masking, a Nearest-dispatch policy served only **10.0%** of potential demand; with the same amount of data withheld but spread randomly instead of spatially clustered, coverage rose to **51.7%** (≈5.17× higher) — showing that *where* data is missing matters as much as *how much*.

### Coverage enforcement without feasibility pruning

| Condition | Confirmed service (%) |
|---|---|
| Nearest | 87.0 |
| Nearest+Cov | 21.3 |

*(30-minute resource-sufficiency regime.)* Explicit coverage pressure alone, without feasibility filtering, redirects fleet-time away from confirmed demand far more aggressively than intended.

### Battery consumption across the ablation chain (15-min critical-scarcity regime)

| Condition | N=3 | N=5 |
|---|---|---|
| Nearest | 4.0 ± 0.5 | 5.7 ± 0.5 |
| ValuePrio | 2.8 ± 0.3 | 5.6 ± 0.7 |
| ValuePrio+Cov | 18.8 ± 1.2 | 18.9 ± 0.9 |
| ValuePrio+Cov+Safe | 3.4 ± 1.3 | 5.3 ± 0.8 |
| Full | 3.9 ± 0.8 | 6.1 ± 0.4 |

*Values are mean ± std of battery % consumed per agent.* Feasibility pruning (PPES) brings battery consumption for the unconstrained coverage condition back down to baseline levels.

### Fairness and operational margin at scale (30-min resource-sufficiency regime)

| N | Condition | Jain's index | Min. battery (%) |
|---|---|---|---|
| 3 | Flat | 0.930 | 72.3 |
| 3 | Nearest | 0.995 | 89.2 |
| 3 | Nearest+Safe | 0.996 | 88.4 |
| 3 | Full | 0.995 | 86.2 |
| 5 | Flat | 0.896 | 71.7 |
| 5 | Nearest | 0.992 | 82.0 |
| 5 | Nearest+Safe | 0.989 | 81.3 |
| 5 | Full | 0.988 | 78.0 |
| 7 | Flat | 0.910 | 70.5 |
| 7 | Nearest | 0.989 | 76.1 |
| 7 | Nearest+Safe | 0.987 | 75.1 |
| 7 | Full | 0.991 | 71.8 |
| 10 | Flat | 0.918 | 71.8 |
| 10 | Nearest | 0.991 | 72.2 |
| 10 | Nearest+Safe | 0.981 | 70.6 |
| 10 | Full | 0.990 | 69.0 |
| 15 | Flat | 0.898 | 66.6 |
| 15 | Nearest | 0.987 | 69.8 |
| 15 | Nearest+Safe | 0.980 | 66.1 |
| 15 | Full | 0.993 | 58.2 |
| 20 | Flat | 0.846 | 61.1 |
| 20 | Nearest | 0.984 | 63.6 |
| 20 | Nearest+Safe | 0.976 | 67.2 |
| 20 | Full | 0.992 | 55.2 |

No battery-depletion incidents occurred in any tested condition; the only return-infeasibility incidents observed were for *Full* at N=20 (≈0.33 events/episode).

### Value-density prioritization and target-selection efficiency (15-min regime)

| N | Condition | Locations served | Confirmed value | Value / location |
|---|---|---|---|---|
| 3 | Nearest | 133.5 ± 2.7 | 0.220 ± 0.007 | 0.00165 |
| 3 | Full | 84.7 ± 4.8 | 0.263 ± 0.009 | 0.00310 |
| 5 | Nearest | 170.6 ± 3.9 | 0.316 ± 0.008 | 0.00186 |
| 5 | Full | 134.6 ± 2.6 | 0.334 ± 0.002 | 0.00248 |
| 7 | Nearest | 195.4 ± 2.9 | 0.379 ± 0.009 | 0.00194 |
| 7 | Full | 160.3 ± 0.8 | 0.367 ± 0.002 | 0.00229 |

*Full* consistently visits fewer locations but selects higher value-per-visit targets than *Nearest*; the selectivity advantage (ratio of value/location) shrinks from ≈1.88× at N=3 to ≈1.18× at N=7 as scarcity eases.

### Key takeaways

- **Reward shaping alone is insufficient**: a 100-fold range of shaping-bonus magnitudes left potential-location coverage within 4.5–6.8%.
- **Explicit coverage without feasibility pruning over-corrects**, diverting large amounts of fleet-time/battery to potential locations at the cost of confirmed-demand service.
- **PPES (feasibility pruning) is the key regularizer**: it recovers most of the lost confirmed-demand value while keeping some potential-location coverage, without the battery cost of unfiltered coverage.
- **The added fair-allocation/profile layer does not increase mean confirmed-demand value or coverage** over ValuePrio+Cov+Safe in the tested critical-scarcity conditions, though it shows some evidence of reducing coverage variance.
- **The value of the full coordination stack is regime-dependent**: under resource sufficiency (30-min regime), simple Nearest dispatch can match or exceed Full, especially at larger fleet sizes.

## Citation

If you use this code or data, please cite the corresponding paper (citation details to be added upon publication).
